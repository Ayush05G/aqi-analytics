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

# Main Diwali (Lakshmi Puja) date per year. The festival moves with the lunar
# calendar, so these are looked up rather than computed. The dataset ends
# 2020-07-01, so Diwali 2020 (14 Nov) is out of range and intentionally omitted.
DIWALI_DATES: dict[int, str] = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
}

# India's national COVID-19 lockdown began 25 Mar 2020. April-May 2020 sit fully
# inside the strictest phase, so they form a clean "lockdown" window to compare
# against the same calendar months in pre-pandemic years.
LOCKDOWN_MONTHS: tuple[int, ...] = (4, 5)
LOCKDOWN_YEAR: int = 2020

# Oct-Nov is the post-monsoon paddy stubble-burning season in Punjab/Haryana,
# upwind of Delhi.
STUBBLE_MONTHS: tuple[int, ...] = (10, 11)


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


# ---------------------------------------------------------------------------
# Event analysis (Phase 2): quantify Diwali, the stubble season, and lockdown
# against an explicit baseline, always returning the actual numbers.
# ---------------------------------------------------------------------------


def _city_series(df: pd.DataFrame, city: str, col: str) -> pd.Series:
    """Date-indexed, gap-dropped series of `col` for one city."""
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    return sub.set_index("Date")[col].sort_index()


def diwali_effect(
    df: pd.DataFrame,
    city: str,
    col: str = "PM2.5",
    days_after: int = 2,
    baseline_days: int = 14,
) -> pd.DataFrame:
    """Per-year fireworks impact: Diwali window vs the fortnight just before it.

    The festival window is [Diwali, Diwali + days_after]; the baseline is the
    `baseline_days` immediately preceding Diwali. Using the local pre-Diwali days
    as baseline isolates the firework spike from the slow stubble-season rise that
    is already underway, rather than crediting the whole Oct-Nov background to it.

    Returns one row per year with festival/baseline means, absolute and % change,
    and the day counts behind each mean.
    """
    series = _city_series(df, city, col)
    rows: list[dict] = []
    for year, dstr in DIWALI_DATES.items():
        d = pd.Timestamp(dstr)
        festival = series.loc[d : d + pd.Timedelta(days=days_after)]
        baseline = series.loc[d - pd.Timedelta(days=baseline_days) : d - pd.Timedelta(days=1)]
        if festival.empty or baseline.empty:
            continue
        f_mean, b_mean = float(festival.mean()), float(baseline.mean())
        rows.append(
            {
                "year": year,
                "diwali": d.date(),
                "festival_mean": f_mean,
                "baseline_mean": b_mean,
                "abs_change": f_mean - b_mean,
                "pct_change": 100.0 * (f_mean - b_mean) / b_mean if b_mean else float("nan"),
                "n_festival": int(festival.size),
                "n_baseline": int(baseline.size),
            }
        )
    return pd.DataFrame(rows)


def stubble_season_effect(
    df: pd.DataFrame,
    city: str,
    col: str = "PM2.5",
    season_months: tuple[int, ...] = STUBBLE_MONTHS,
) -> pd.DataFrame:
    """Per-year Oct-Nov mean vs the rest of that same year.

    Comparing within the year controls for the long-run downward trend. Returns
    season vs rest-of-year means, the multiplier, % change, and day counts.
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    sub = sub.assign(year=sub["Date"].dt.year, month=sub["Date"].dt.month)
    rows: list[dict] = []
    for year, g in sub.groupby("year"):
        in_season = g.loc[g["month"].isin(season_months), col]
        rest = g.loc[~g["month"].isin(season_months), col]
        if in_season.empty or rest.empty:
            continue
        s_mean, r_mean = float(in_season.mean()), float(rest.mean())
        rows.append(
            {
                "year": int(year),
                "season_mean": s_mean,
                "rest_mean": r_mean,
                "multiplier": s_mean / r_mean if r_mean else float("nan"),
                "pct_change": 100.0 * (s_mean - r_mean) / r_mean if r_mean else float("nan"),
                "n_season": int(in_season.size),
                "n_rest": int(rest.size),
            }
        )
    return pd.DataFrame(rows)


def lockdown_effect(
    df: pd.DataFrame,
    city: str,
    col: str = "PM2.5",
    months: tuple[int, ...] = LOCKDOWN_MONTHS,
    lockdown_year: int = LOCKDOWN_YEAR,
) -> pd.DataFrame:
    """Per-year mean of `col` over the lockdown calendar months (Apr-May).

    Returns one row per year present, with the month-window mean, day count, and
    an is_lockdown flag for the lockdown year. The caller compares the lockdown
    year against the mean of the pre-pandemic rows (see lockdown_summary).
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    sub = sub.assign(year=sub["Date"].dt.year, month=sub["Date"].dt.month)
    window = sub[sub["month"].isin(months)]
    out = (
        window.groupby("year")[col]
        .agg(["mean", "count"])
        .rename(columns={"count": "n_days"})
        .reset_index()
    )
    out["is_lockdown"] = out["year"] == lockdown_year
    return out


def lockdown_summary(lockdown_table: pd.DataFrame) -> dict[str, float]:
    """Reduce a lockdown_effect table to lockdown vs pre-pandemic-baseline numbers."""
    base = lockdown_table.loc[~lockdown_table["is_lockdown"], "mean"]
    lock_rows = lockdown_table.loc[lockdown_table["is_lockdown"], "mean"]
    if base.empty or lock_rows.empty:
        raise ValueError("Need both lockdown-year and baseline-year rows.")
    baseline_mean = float(base.mean())
    lockdown_mean = float(lock_rows.iloc[0])
    return {
        "lockdown_mean": lockdown_mean,
        "baseline_mean": baseline_mean,
        "abs_change": lockdown_mean - baseline_mean,
        "pct_change": 100.0 * (lockdown_mean - baseline_mean) / baseline_mean,
        "n_baseline_years": int(base.size),
    }


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

    # ---- Phase 2: event analysis -----------------------------------------
    print(f"\n=== Diwali fireworks effect on PM2.5 - {city} ===")
    diw = diwali_effect(df, city, "PM2.5")
    print(diw.round(1).to_string(index=False))
    print(
        f"-> On average Diwali week PM2.5 runs {diw['pct_change'].mean():+.0f}% vs the "
        f"fortnight before (worst: {diw['year'][diw['pct_change'].idxmax()]} "
        f"{diw['pct_change'].max():+.0f}%)."
    )

    print(f"\n=== Stubble-burning season (Oct-Nov) effect on PM2.5 - {city} ===")
    stub = stubble_season_effect(df, city, "PM2.5")
    print(stub.round(2).to_string(index=False))
    print(
        f"-> Oct-Nov PM2.5 averages {stub['multiplier'].mean():.1f}x the rest of the "
        f"year ({stub['pct_change'].mean():+.0f}% on average)."
    )

    print(f"\n=== COVID lockdown (Apr-May 2020) effect on PM2.5 - {city} ===")
    lock = lockdown_effect(df, city, "PM2.5")
    print(lock.round(1).to_string(index=False))
    summ = lockdown_summary(lock)
    print(
        f"-> Apr-May 2020 PM2.5 was {summ['lockdown_mean']:.0f} vs a "
        f"{summ['n_baseline_years']}-year baseline of {summ['baseline_mean']:.0f} "
        f"({summ['pct_change']:+.0f}%)."
    )
