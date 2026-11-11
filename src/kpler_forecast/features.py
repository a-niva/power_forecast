"""Features: national weather aggregates, calendar, merit order."""
import holidays
import numpy as np
import pandas as pd

def fct_features__weather(weather):
    ssrd = (weather.xs("ssrd", level="weather_var", axis=1) / 3600).clip(lower=0) # ssrd is J/m2 accumulated over the hour, so /3600 gives W/m2. + Small negatives are noise, we clip them at zero.
    ws = weather.xs("100ws", level="weather_var", axis=1)

    f = pd.DataFrame(index=weather.index)

    # solar radiation:
    f["ssrd_mean"] = ssrd.mean(axis=1)
    f["ssrd_q25"] = ssrd.quantile(0.25, axis=1)
    f["ssrd_q75"] = ssrd.quantile(0.75, axis=1)

    # total cloud cover:
    f["tcc_mean"] = weather.xs("tcc", level="weather_var", axis=1).mean(axis=1) #Here we take the mean of the total cloud cover (tcc) over all tiles, as it is a national feature.
    # Quick explanation here: temperature varies almost everywhere the same way in France, so the national average doesn't disadvantage the model. Conversely, wind speeds vary significantly by location, so we don't use a national avg.
    
    #wind speed:
    f["ws_mean"] = ws.mean(axis=1)
    f["ws_q90"] = ws.quantile(0.9, axis=1)

    #temperature:
    f["temp"] = weather.xs("2t", level="weather_var", axis=1).mean(axis=1)
    #easy additional feature, buildings taking a couple days to cool down:
    f["temp_24h"] = f["temp"].ewm(span=24).mean() #take the EMA
    f["temp_72h"] = f["temp"].ewm(span=72).mean() #take the EMA
    return f.ffill().bfill()   # a few backfilled NaNs in ssrd in the very first hours




LAT, LON = 46.5, 2.5 # --> rough center of mainland France (for the sun position only)

def fct_features__sun_elevation(index):
    """Approximate solar elevation in degrees (astronomy, not external data...)"""
    doy = index.day_of_year.to_numpy()
    hour = index.hour.to_numpy() + 0.5  # middle of the hour
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (hour - 12) + LON)
    lat = np.radians(LAT)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(sin_elev))



LOCAL_TZ = "Europe/Paris"

def fct_features__calendar(index):
    # people live in Paris time, not UTC
    local = index.tz_convert(LOCAL_TZ)
    fr_holidays = holidays.France(years=range(local.year.min(), local.year.max() + 1))

    f = pd.DataFrame(index=index)
    f["hour"] = local.hour
    f["weekday"] = local.day_of_week
    f["doy_sin"] = np.sin(2 * np.pi * local.day_of_year / 365.25)  #seasonal feature, to help the model extrapolate to new years
    f["doy_cos"] = np.cos(2 * np.pi * local.day_of_year / 365.25)  #seasonal feature, to help the model extrapolate to new years
    f["is_holiday"] = pd.Index(local.date).isin(list(fr_holidays)).astype(int)
    f["is_xmas"] = ((local.month == 12) & (local.day >= 24) | (local.month == 1) & (local.day <= 2)).astype(int)
    f["is_august"] = ((local.month == 8) & (local.day.isin(range(1, 25)))).astype(int)
    f["sun_elev"] = fct_features__sun_elevation(index)
    # demand dropped with the 2022 energy crisis and hasn't come back; bet: still true in 2025
    f["post_crisis"] = (index >= pd.Timestamp("2022-10-01", tz="UTC")).astype(int)
    return f


def fct_features__market(network, load, solar, wind):
    #Demand left once solar and wind are served, and the fuel prices.
    f = network[["EEX_GAS_PEG", "EEX_CARBON", "EEX_COAL", "FR_availability_nuclear", "FR_availability_hydro", "FR_availability_gas", "FR_availability_coal"]].copy()
    f["load"] = load
    f["residual_load"] = load - solar - wind
    f["residual_after_nuclear"] = f["residual_load"] - network.FR_availability_nuclear
    f["renewable_share"] = (solar + wind) / load
    return f
