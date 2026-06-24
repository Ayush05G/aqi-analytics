"""Pure analysis functions for Delhi-NCR air quality. No Streamlit, no plotting.

Phase 0-1 scope: long-run AQI trend (monthly/yearly) and a seasonal
decomposition that exposes the annual cycle (winter peaks, monsoon lows).

Missing-data policy (deliberate, per project constraints):
- Trends aggregate only observed days; the day count is returned alongside every
  aggregate so thin months/years are visible rather than hidden.
- Decomposition needs a gap-free, regularly-spaced series. We resample to monthly
  means, then linearly interpolate ONLY short internal gaps (default <= 2 months)
  and drop any leading/trailing NaNs. The number of interpolated months is
  returned so the caller can judge how much was synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _city_frame(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Rows for a single city, validated."""
    sub = df[df["City"] == city]
    if sub.empty:
        raise ValueError(f"No rows for city {city!r}. Available: {sorted(df['City'].unique())}")
    return sub


def monthly_aqi_trend(df: pd.DataFrame, city: str, col: str = "AQI") -> pd.DataFrame:
    """Monthly mean/median of `col` for one city.

    Returns columns: month (month-start datetime), mean, median, n_days.
    Months with no observations are omitted (not zero-filled).
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    grouped = sub.set_index("Date").resample("MS")[col]
    out = grouped.agg(["mean", "median", "count"]).rename(columns={"count": "n_days"})
    out = out[out["n_days"] > 0].reset_index().rename(columns={"Date": "month"})
    return out


def yearly_aqi_trend(df: pd.DataFrame, city: str, col: str = "AQI") -> pd.DataFrame:
    """Yearly mean/median of `col` for one city.

    Returns columns: year, mean, median, n_days. Use n_days to spot partial
    years (e.g. 2020 ends mid-year in this dataset).
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    out = (
        sub.assign(year=sub["Date"].dt.year)
        .groupby("year")[col]
        .agg(["mean", "median", "count"])
        .rename(columns={"count": "n_days"})
        .reset_index()
    )
    return out


def monthly_climatology(df: pd.DataFrame, city: str, col: str = "PM2.5") -> pd.DataFrame:
    """Average annual cycle: mean of `col` per calendar month (1-12), all years.

    Returns columns: month_num (1-12), month (name), mean, n_days. This is the
    quickest sanity check on seasonality — expect a winter peak and monsoon low.
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    g = sub.assign(month_num=sub["Date"].dt.month).groupby("month_num")[col]
    out = g.agg(["mean", "count"]).rename(columns={"count": "n_days"}).reset_index()
    out["month"] = out["month_num"].map(lambda m: MONTH_NAMES[m - 1])
    return out[["month_num", "month", "mean", "n_days"]]


@dataclass
class SeasonalDecomposition:
    """Result of a monthly seasonal decomposition plus provenance metadata."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    city: str
    col: str
    model: str
    period: int
    n_interpolated: int  # months filled by interpolation to close short gaps
    seasonal_cycle: pd.DataFrame  # one row per calendar month: month, effect


def _monthly_series(df: pd.DataFrame, city: str, col: str, max_gap: int) -> tuple[pd.Series, int]:
    """Gap-free monthly-mean series for decomposition + count of filled months."""
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    series = sub.set_index("Date").resample("MS")[col].mean()
    # Trim leading/trailing empty months, then fill only short internal gaps.
    series = series.loc[series.first_valid_index():series.last_valid_index()]
    n_interpolated = int(series.isna().sum())
    filled = series.interpolate(method="linear", limit=max_gap, limit_area="inside")
    remaining = int(filled.isna().sum())
    if remaining:
        raise ValueError(
            f"{city}/{col}: {remaining} month(s) remain missing after interpolating "
            f"gaps up to {max_gap} months. Series too sparse for decomposition."
        )
    filled.name = col
    return filled, n_interpolated


def seasonal_decomposition(
    df: pd.DataFrame,
    city: str,
    col: str = "PM2.5",
    period: int = 12,
    model: str = "additive",
    max_gap: int = 2,
) -> SeasonalDecomposition:
    """Decompose a city's monthly `col` series into trend/seasonal/residual.

    Requires at least two full annual cycles. The `seasonal_cycle` field gives the
    repeating monthly effect (peak month = worst season for `col`).
    """
    series, n_interpolated = _monthly_series(df, city, col, max_gap)
    if len(series) < 2 * period:
        raise ValueError(
            f"{city}/{col}: only {len(series)} months; need >= {2 * period} for period={period}."
        )

    result: DecomposeResult = seasonal_decompose(series, model=model, period=period)

    # Collapse the repeating seasonal component to one value per calendar month.
    seasonal = result.seasonal
    cycle = (
        seasonal.groupby(seasonal.index.month)
        .mean()
        .rename_axis("month_num")
        .reset_index(name="effect")
    )
    cycle["month"] = cycle["month_num"].map(lambda m: MONTH_NAMES[m - 1])

    return SeasonalDecomposition(
        observed=result.observed,
        trend=result.trend,
        seasonal=seasonal,
        resid=result.resid,
        city=city,
        col=col,
        model=model,
        period=period,
        n_interpolated=n_interpolated,
        seasonal_cycle=cycle[["month_num", "month", "effect"]],
    )


if __name__ == "__main__":
    from src.data_load import load_clean

    df = load_clean()
    city = "Delhi"

    print(f"\n=== Yearly AQI trend - {city} ===")
    print(yearly_aqi_trend(df, city).to_string(index=False))

    print(f"\n=== Monthly PM2.5 climatology (annual cycle) - {city} ===")
    clim = monthly_climatology(df, city, "PM2.5")
    print(clim.to_string(index=False))
    peak = clim.loc[clim["mean"].idxmax()]
    trough = clim.loc[clim["mean"].idxmin()]
    print(f"-> Peak: {peak['month']} ({peak['mean']:.0f})   Low: {trough['month']} ({trough['mean']:.0f})")

    print(f"\n=== Seasonal decomposition of PM2.5 - {city} ===")
    dec = seasonal_decomposition(df, city, "PM2.5")
    print(f"Months: {len(dec.observed)}  | interpolated: {dec.n_interpolated}  | model: {dec.model}")
    print("Seasonal effect by month (additive, deviation from trend):")
    print(dec.seasonal_cycle.to_string(index=False))
    sc = dec.seasonal_cycle
    print(
        f"-> Seasonal peak: {sc.loc[sc['effect'].idxmax(), 'month']}  "
        f"trough: {sc.loc[sc['effect'].idxmin(), 'month']}"
    )
