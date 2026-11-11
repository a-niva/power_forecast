"""Walk-forward backtest: train on past years, score on the next one, compare with a naive baseline."""
import numpy as np
import pandas as pd

from kpler_forecast import TARGETS
from kpler_forecast.models import Forecaster


def fct_evaluate__scores(y_true, y_pred):
    err = y_pred[TARGETS] - y_true[TARGETS]  # missing actuals give NaN, which mean() skips
    return pd.DataFrame({"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean()})


def fct_evaluate__seasonal_naive(targets, index):
    """Same hour 52 weeks earlier (keeps the weekday). The bar any model must beat."""
    return targets.reindex(index - pd.Timedelta(weeks=52)).set_axis(index)


def fct_evaluate__backtest(weather, network, targets, valid_year):
    train = targets.index.year < valid_year
    valid = targets.index.year == valid_year
    model = Forecaster().fit(weather[train], network[train], targets[train])

    truth = targets[valid]
    pred = model.predict(weather[valid], network[valid])

    table = pd.concat({
        "model": fct_evaluate__scores(truth, pred),
        "baseline": fct_evaluate__scores(truth, fct_evaluate__seasonal_naive(targets, truth.index)),
    }, axis=1)
    table["skill_MAE"] = 1 - table[("model", "MAE")] / table[("baseline", "MAE")]
    return {"table": table, "pred": pred, "truth": truth, "model": model}
