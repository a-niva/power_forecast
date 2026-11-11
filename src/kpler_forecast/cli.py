"""
    python -m kpler_forecast.cli all                   # backtest + train + predict
    python -m kpler_forecast.cli backtest              # scores on 2023 and 2024
    python -m kpler_forecast.cli train                 # fit on 2020-2024, save models/<version>/
    python -m kpler_forecast.cli predict --split test  # forecast from weather_test + network_test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kpler_forecast import TARGETS
from kpler_forecast.evaluate import fct_evaluate__backtest
from kpler_forecast.models import Forecaster

ROOT = Path(__file__).resolve().parents[2]


def fct_cli__read_inputs(data_dir, split):
    weather = pd.read_parquet(data_dir / f"weather_{split}.parquet")
    network = pd.read_parquet(data_dir / f"network_{split}.parquet")
    assert weather.index.equals(network.index)
    return weather, network


def fct_cli__read_train(data_dir):
    """Same cleaning as section 1 of the notebook."""
    weather, network = fct_cli__read_inputs(data_dir, "train")
    targets = pd.read_parquet(data_dir / "target_train.parquet")[TARGETS].astype("float64")
    targets = targets[targets.index.year < 2025]  # the file ends with 2025-01-01 00:00
    weather, network = weather.loc[targets.index], network.loc[targets.index]
    # a plant can't produce above its capacity; wind also has a single 52.9 GW spike in Jan 2023
    targets.loc[targets.FR_solar_actual > network.FR_capacity_solar, "FR_solar_actual"] = np.nan
    targets.loc[targets.FR_wind_actual / network.FR_capacity_wind > 0.5, "FR_wind_actual"] = np.nan
    return weather, network, targets


def fct_cli__run_backtest(data_dir):
    weather, network, targets = fct_cli__read_train(data_dir)
    summary = {}
    for year in [2023, 2024]:
        res = fct_evaluate__backtest(weather, network, targets, valid_year=year)
        print(f"\n=== validation {year} (trained on years < {year}) ===")
        print(res["table"].round(2).to_string())
        summary[year] = res["table"][("model", "MAE")].round(1).to_dict()
    return summary


def fct_cli__train(data_dir, model_dir, metrics=None):
    model = Forecaster().fit(*fct_cli__read_train(data_dir))
    path = model.save(model_dir, metrics)
    print(f"saved {path}")
    return path


def fct_cli__predict(data_dir, model_dir, split, out, version=None):
    """`split` points to weather_<split>.parquet and network_<split>.parquet, e.g. 'test' or '2026'."""
    weather, network = fct_cli__read_inputs(data_dir, split)
    model = Forecaster.load(model_dir, version)
    pred = model.predict(weather, network)
    fct_cli__sanity_check(pred, weather.index)
    pred.reset_index().to_parquet(out, index=False)
    print(f"wrote {out} ({len(pred)} rows)")
    return pred


def fct_cli__sanity_check(pred, index):
    assert list(pred.columns) == TARGETS
    assert pred.index.equals(index) and str(pred.index.tz) == "UTC"
    assert pred.notna().all().all(), "NaN in forecasts"
    assert (pred[["FR_solar_actual", "FR_wind_actual", "FR_load_actual"]] >= 0).all().all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["backtest", "train", "predict", "all"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--split", default="test")
    parser.add_argument("--version", help="folder in models/, defaults to models/LATEST")
    parser.add_argument("--out", type=Path, default=ROOT / "test_results.parquet")
    args = parser.parse_args()

    metrics = None
    if args.command in ("backtest", "all"):
        metrics = fct_cli__run_backtest(args.data_dir)
    if args.command in ("train", "all"):
        fct_cli__train(args.data_dir, args.model_dir, metrics)
    if args.command in ("predict", "all"):
        fct_cli__predict(args.data_dir, args.model_dir, args.split, args.out, args.version)


if __name__ == "__main__":
    main()
