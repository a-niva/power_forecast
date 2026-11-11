"""One gradient-boosted model per target. Load/solar/wind first, then price on top of their forecasts."""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, QuantileRegressor

from kpler_forecast import TARGETS
from kpler_forecast.features import fct_features__calendar, fct_features__market, fct_features__weather

CALENDAR = ["hour", "weekday", "doy_sin", "doy_cos", "is_holiday", "is_xmas", "is_august"]

FEATURES = {
    "FR_load_actual": CALENDAR + ["post_crisis", "temp", "temp_24h", "temp_72h", "ssrd_mean", "tcc_mean"],
    "FR_solar_actual": ["tiles", "sun_elev", "doy_sin", "doy_cos", "ssrd_mean", "ssrd_q25", "ssrd_q75", "tcc_mean", "temp"],
    "FR_wind_actual": ["tiles", "ws_mean", "ws_q90", "doy_sin", "doy_cos", "hour", "temp"],
    "FR_price_actual": CALENDAR + [
        "load", "residual_load", "residual_after_nuclear", "renewable_share",
        "fuel_cost", "EEX_GAS_PEG", "EEX_CARBON", "EEX_COAL",
        "FR_availability_nuclear", "FR_availability_hydro", "FR_availability_gas", "FR_availability_coal",
    ],
}

# Capacity keeps growing and trees can't extrapolate: solar and wind are learned as load factors.
CAPACITY = {"FR_solar_actual": "FR_capacity_solar", "FR_wind_actual": "FR_capacity_wind"}
FUELS = ["EEX_GAS_PEG", "EEX_CARBON", "EEX_COAL"]

XGB_PARAMS = dict(n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=50,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42)


def fct_models__base_features(weather):
    return fct_features__weather(weather).join(fct_features__calendar(weather.index))


def fct_models__booster(objective="l2"):
    return xgb.XGBRegressor(objective={"l2": "reg:squarederror", "l1": "reg:absoluteerror"}[objective], **XGB_PARAMS)


def fct_models__fuel_model(network, price):
    # what the three fuel prices alone say about the price: a median regression, learned from the data.
    # Fuel prices change once a day, so fitting on daily means gives the same answer much faster.
    daily = pd.concat([network[FUELS], price.rename("price")], axis=1).resample("D").mean().dropna()
    return QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(daily[FUELS], daily["price"])


class Forecaster:
    """One gradient-boosted model per target. Load/solar/wind first, then price on top of their forecasts."""
    
    def __init__(self):
        self.tile_weights = {}
        self.models = {}
        self.meta = {}

    @staticmethod
    def _tiles(weather, target):
        # one column per weather tile: solar radiation in W/m2 (ssrd is J/m2 per hour), wind speed at 100 m
        if target == "FR_solar_actual":
            return (weather.xs("ssrd", level="weather_var", axis=1) / 3600).clip(lower=0).fillna(0)
        return weather.xs("100ws", level="weather_var", axis=1)

    def fit(self, weather, network, targets):
        """Fit the forecaster on the given data."""
        X = fct_models__base_features(weather)
        
        # 1. Train models for physical targets (load, solar, wind)
        for target in ["FR_load_actual", "FR_solar_actual", "FR_wind_actual"]:
            y = targets[target]
            
            if target in CAPACITY:
                y = y / network[CAPACITY[target]]
                # Plants aren't spread evenly: a positive linear fit on the tiles finds where they are
                tiles = self._tiles(weather, target)
                ok = y.notna()
                
                self.tile_weights[target] = LinearRegression(positive=True).fit(tiles[ok], y[ok])
                X = X.assign(tiles=self.tile_weights[target].predict(tiles))
                
            self.models[target] = self._fit_one(X[FEATURES[target]], y)

        # 2. Train the electricity price model
        # - Trained on actual load/solar/wind, fed our forecasts at inference 
        #   (training on out-of-fold forecasts helped a little in one test, not adopted).
        # - Target is the spread over the fuel cost: fuel prices moved 10x in 2022 but the spread much less.
        # - L1 so the 2022 spikes don't drag the fit.
        self.fuel_model = fct_models__fuel_model(network, targets.FR_price_actual)
        
        market_features = fct_features__market(
            network, 
            targets.FR_load_actual, 
            targets.FR_solar_actual, 
            targets.FR_wind_actual
        )
        Xp = X.join(market_features)
        Xp["fuel_cost"] = self.fuel_model.predict(network[FUELS])
        
        spread = targets.FR_price_actual - Xp.fuel_cost
        self.models["FR_price_actual"] = self._fit_one(
            Xp[FEATURES["FR_price_actual"]], 
            spread, 
            objective="l1"
        )

        # 3. Capture run metadata
        now = datetime.now(timezone.utc)
        self.meta = {
            "version": now.strftime("%Y%m%d_%H%M"), 
            "trained_at": now.isoformat(timespec="seconds"),
            "train_start": str(targets.index.min()), 
            "train_end": str(targets.index.max()),
            "features": FEATURES,
            "params": XGB_PARAMS
        }
        
        return self

    def _fit_one(self, X, y, objective="l2"):
        """Fit a single model on the given data."""
        ok = y.notna()
        return fct_models__booster(objective).fit(X[ok], y[ok])

    def predict(self, weather, network):
        """Predict all targets for the given weather and network data."""
        X = fct_models__base_features(weather)
        out = pd.DataFrame(index=weather.index)
        
        # 1. Predict physical targets (load, solar, wind)
        for target in ["FR_load_actual", "FR_solar_actual", "FR_wind_actual"]:
            if target in CAPACITY:
                X = X.assign(tiles=self.tile_weights[target].predict(self._tiles(weather, target)))
                
            out[target] = self.models[target].predict(X[FEATURES[target]])
            
            if target in CAPACITY:
                out[target] *= network[CAPACITY[target]]
                
        # Enforce physical constraints (no solar production at night, no negative wind)
        out["FR_solar_actual"] = out.FR_solar_actual.where(X.sun_elev > -2, 0).clip(lower=0)
        out["FR_wind_actual"] = out.FR_wind_actual.clip(lower=0)

        # 2. Predict electricity price using physical forecasts and fuel costs
        market_features = fct_features__market(
            network, 
            out.FR_load_actual, 
            out.FR_solar_actual, 
            out.FR_wind_actual
        )
        Xp = X.join(market_features)
        Xp["fuel_cost"] = self.fuel_model.predict(network[FUELS])
        
        out["FR_price_actual"] = (
            self.models["FR_price_actual"].predict(Xp[FEATURES["FR_price_actual"]]) 
            + Xp.fuel_cost
        )
        
        return out[TARGETS]

    def save(self, model_dir, metrics=None):
        """Save the forecaster to the given directory."""
        path = model_dir / self.meta["version"]
        path.mkdir(parents=True, exist_ok=True)
        
        # Serialize model and save metadata record
        joblib.dump(self, path / "forecaster.joblib")
        meta = {**self.meta, "metrics": metrics or {}}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        
        # Update the latest version pointer
        (model_dir / "LATEST").write_text(self.meta["version"])
        
        return path

    @staticmethod
    def load(model_dir, version=None):
        """Load the forecaster from the given directory."""
        version = version or (model_dir / "LATEST").read_text().strip()
        return joblib.load(model_dir / version / "forecaster.joblib")