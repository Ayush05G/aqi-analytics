"""Short-horizon daily AQI forecasting with SARIMA, isolated from the rest.

Approach (Delhi, daily AQI):
- Build a gap-free daily series (the Delhi AQI series has only ~10 missing days
  out of ~2,000, max gap 4 days, so short linear interpolation is safe and is
  reported via `n_interpolated`).
- Hold out the last `horizon` days as a test set.
- Fit SARIMA with weekly seasonality (m=7); the order is chosen by a small AIC
  grid search on the training set. Daily data also has an annual cycle, but
  m=365 is impractical for SARIMA; over a 2-4 week horizon the weekly term plus
  autocorrelation and differencing carry most of the signal. We validate that
  claim by benchmarking against naive baselines rather than trusting it.
- Report MAE / RMSE / MAPE on the holdout for SARIMA and for two naive baselines
  (persistence and seasonal-naive). A forecast only "works" if it beats naive.

This module has no Streamlit/plotting dependencies.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Candidate (order, seasonal_order) pairs for the AIC search. Kept small and
# weekly-seasonal so the search runs in seconds, not minutes.
_CANDIDATE_ORDERS: tuple[tuple[tuple[int, int, int], tuple[int, int, int, int]], ...] = (
    ((1, 1, 1), (0, 0, 0, 0)),   # plain ARIMA reference
    ((2, 1, 2), (0, 0, 0, 0)),
    ((1, 1, 1), (1, 0, 1, 7)),
    ((2, 1, 1), (1, 0, 1, 7)),
    ((1, 1, 2), (0, 1, 1, 7)),
    ((2, 1, 2), (1, 0, 1, 7)),
)


def build_daily_series(
    df: pd.DataFrame, city: str, col: str = "AQI", max_gap: int = 7
) -> tuple[pd.Series, int]:
    """Continuous daily series for one city + count of interpolated days.

    Resamples to a daily grid and linearly interpolates internal gaps up to
    `max_gap` days; raises if longer gaps remain.
    """
    sub = df[df["City"] == city]
    if sub.empty:
        raise ValueError(f"No rows for city {city!r}.")
    series = sub.set_index("Date")[col].sort_index().resample("D").mean()
    series = series.loc[series.first_valid_index():series.last_valid_index()]
    n_interpolated = int(series.isna().sum())
    series = series.interpolate(method="linear", limit=max_gap, limit_area="inside")
    remaining = int(series.isna().sum())
    if remaining:
        raise ValueError(
            f"{city}/{col}: {remaining} day(s) still missing after interpolating "
            f"gaps up to {max_gap} days."
        )
    series.name = col
    return series.asfreq("D"), n_interpolated


def split_train_test(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Hold out the last `horizon` observations as the test set."""
    if len(series) <= horizon * 2:
        raise ValueError(f"Series too short ({len(series)}) for horizon {horizon}.")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def forecast_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE, RMSE, and MAPE between aligned actual/predicted series."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = a - p
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / a)) * 100.0),
    }


def naive_persistence(train: pd.Series, horizon: int) -> pd.Series:
    """Baseline: tomorrow = today, repeated (last observed value held flat)."""
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(float(train.iloc[-1]), index=idx, name="persistence")


def naive_seasonal(train: pd.Series, horizon: int, m: int = 7) -> pd.Series:
    """Baseline: repeat the last `m` observed values (seasonal-naive, weekly)."""
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    last = train.iloc[-m:].to_numpy()
    values = np.resize(last, horizon)
    return pd.Series(values, index=idx, name="seasonal_naive")


def select_order(train: pd.Series):
    """Pick the (order, seasonal_order) with the lowest AIC on the training set."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for order, seasonal_order in _CANDIDATE_ORDERS:
            try:
                res = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            except Exception:
                continue
            if best is None or res.aic < best[2]:
                best = (order, seasonal_order, float(res.aic))
    if best is None:
        raise RuntimeError("No SARIMA candidate converged.")
    return best[0], best[1], best[2]


@dataclass
class ForecastResult:
    """Backtest outcome: chosen model, the forecast, and comparative metrics."""

    city: str
    col: str
    horizon: int
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float
    n_interpolated: int
    train: pd.Series
    test: pd.Series
    forecast: pd.Series
    conf_int: pd.DataFrame
    metrics: dict[str, dict[str, float]]  # model name -> {mae, rmse, mape}


def backtest(
    df: pd.DataFrame, city: str = "Delhi", col: str = "AQI", horizon: int = 30
) -> ForecastResult:
    """Fit SARIMA on all-but-last-`horizon` days, forecast, and score vs baselines."""
    series, n_interpolated = build_daily_series(df, city, col)
    train, test = split_train_test(series, horizon)

    order, seasonal_order, aic = select_order(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        fitted = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fc = fitted.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    mean.index = test.index
    conf = fc.conf_int()
    conf.index = test.index

    metrics = {
        "sarima": forecast_metrics(test, mean),
        "persistence": forecast_metrics(test, naive_persistence(train, horizon)),
        "seasonal_naive": forecast_metrics(test, naive_seasonal(train, horizon)),
    }

    return ForecastResult(
        city=city,
        col=col,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
        aic=aic,
        n_interpolated=n_interpolated,
        train=train,
        test=test,
        forecast=mean,
        conf_int=conf,
        metrics=metrics,
    )


def forecast_future(
    df: pd.DataFrame, city: str = "Delhi", col: str = "AQI", horizon: int = 30
) -> tuple[pd.Series, pd.DataFrame]:
    """Refit on the full series and forecast `horizon` days past the data end.

    For dashboard use; returns (predicted_mean, conf_int).
    """
    series, _ = build_daily_series(df, city, col)
    order, seasonal_order, _ = select_order(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        fitted = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    fc = fitted.get_forecast(steps=horizon)
    return fc.predicted_mean, fc.conf_int()


if __name__ == "__main__":
    from src.data_load import load_clean

    df = load_clean()
    result = backtest(df, "Delhi", "AQI", horizon=30)

    print(f"=== SARIMA backtest: {result.city} {result.col}, {result.horizon}-day holdout ===")
    print(f"Series: {len(result.train)} train + {len(result.test)} test days "
          f"(interpolated {result.n_interpolated})")
    print(f"Chosen order: SARIMA{result.order}x{result.seasonal_order}  AIC={result.aic:.0f}")
    print(f"Backtest window: {result.test.index.min().date()} -> {result.test.index.max().date()}")
    print("\nHoldout error (lower is better):")
    print(f"{'model':<16}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}")
    for name, m in result.metrics.items():
        print(f"{name:<16}{m['mae']:>8.1f}{m['rmse']:>8.1f}{m['mape']:>8.1f}")

    sarima_mae = result.metrics["sarima"]["mae"]
    best_naive = min(result.metrics["persistence"]["mae"], result.metrics["seasonal_naive"]["mae"])
    verdict = "beats" if sarima_mae < best_naive else "does NOT beat"
    print(
        f"\n-> SARIMA MAE {sarima_mae:.1f} {verdict} the best naive baseline "
        f"({best_naive:.1f}) over a {result.horizon}-day horizon."
    )
