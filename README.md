# French power in 2025: load, solar, wind and price

Hourly forecasts for France over 2025 of grid demand, utility-scale solar, wind and the day-ahead price,
built from weather tiles, fuel prices, installed capacities and plant availabilities (Kpler Power & Gas case study).

Start with **[`notebooks/case_study_v4.ipynb`](notebooks/case_study_v4.ipynb)**. It is written to be read without any
knowledge of the energy sector: the data, the ideas that made the difference (one chart each), how the price model
reaches a price (SHAP), and where the model still fails and why. It runs in about a minute.

## Results

MAE on a walk-forward backtest (each year is scored by a model trained only on the years before it):

| | Load (MW) | Solar (MW) | Wind (MW) | Price (€/MWh) |
|---|---|---|---|---|
| Model, 2024 | 1 074 | 365 | 561 | 23.6 |
| Same hour 52 weeks earlier, 2024 | 3 840 | 830 | 3 910 | 53.6 |
| Model, 2023 | 1 509 | 471 | 525 | 17.8 |
| Same hour 52 weeks earlier, 2023 | 3 903 | 805 | 3 432 | 180.7 |

2024 is the year to look at: it's the closest to 2025. The 2023 price baseline uses 2022 crisis prices, so it is easy to beat.

## Run it

The data belongs to Kpler and is not in this repo. Put the five parquet files in `data/`, then:

```bash
python -m venv .venv && source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt && pip install -e .
python -m kpler_forecast.cli all      # backtest 2023 + 2024, train on 2020-2024, write the 2025 forecasts (~25 s)
```

This writes `test_results.parquet` (the deliverable: `start_date` in UTC plus the four targets, 8 760 rows).

Each training run is saved under `models/<version>/` with `meta.json` (training period, features, parameters,
backtest MAE). `models/LATEST` points to the default. A future year only needs
`weather_<split>.parquet` and `network_<split>.parquet`:

```bash
python -m kpler_forecast.cli predict --split 2026 --out results_2026.parquet [--version <models/ folder>]
```

## Main choices

- One XGBoost model per target with fixed parameters. The gain comes from the features, not from tuning.
- Solar and wind are predicted as load factors and scaled by capacity: 2025 capacity is above anything in training, and trees can't extrapolate.
- Learned tile weights (positive linear regression on the weather tiles) tell the solar and wind models where the plants are.
- A `post_crisis` flag for load: demand has been about 3 GW lower since October 2022.
- Price is learned as a spread over a fuel cost (median regression on the gas, carbon and coal prices), on top of the forecast load, solar and wind. No plant efficiency or emission figure is set by hand.
- Calendar features in Europe/Paris time; joins and outputs in UTC.

## What I couldn't fix

- The price runs about 17 €/MWh high in 2024. The market traded far below what the fuel prices suggest that year, and the data has no variable that explains it.
- Solar is too high on sunny weekends with negative prices, probably because plants are switched off when selling costs money. The solar model does not know the price.
- 2025 weather is observed, not forecast, so real-world errors would probably be larger.

## Layout

```
src/kpler_forecast/   features.py  models.py  evaluate.py  cli.py   (~380 lines; reading and cleaning live in cli.py)
notebooks/            case_study_v4.ipynb
models/               versioned fitted models + metadata
```

