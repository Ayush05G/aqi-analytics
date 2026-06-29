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

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# Main Diwali (Lakshmi Puja) date per year. The festival moves with the lunar
# calendar, so these are looked up rather than computed. diwali_effect skips any
# year not covered by the loaded data, so out-of-range years are harmless.
# (Data now runs to 2023-03; Diwali 2023 is in November, hence not included.)
DIWALI_DATES: dict[int, str] = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
    2021: "2021-11-04",
    2022: "2022-10-24",
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
    """Monthly-mean series for decomposition + count of filled months.

    Short internal gaps (<= max_gap months) are interpolated. If a longer gap
    remains (e.g. the 2023-2025 concentration hole), the LONGEST contiguous run
    is returned rather than failing — so the decomposition reflects the period
    that actually has data.
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    series = sub.set_index("Date").resample("MS")[col].mean()
    series = series.loc[series.first_valid_index():series.last_valid_index()]
    filled = series.interpolate(method="linear", limit=max_gap, limit_area="inside")

    if filled.isna().any():
        # Keep the longest contiguous stretch of present months.
        valid = filled.notna()
        run_id = (valid != valid.shift()).cumsum()
        longest = valid.groupby(run_id).sum().idxmax()
        filled = filled[(run_id == longest) & valid]

    n_interpolated = int(series.reindex(filled.index).isna().sum())
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


# ---------------------------------------------------------------------------
# Source attribution (Phase 5): which pollutant drives bad-air days per season.
# We recompute each day's CPCB sub-index per pollutant from raw concentrations,
# take the pollutant with the highest sub-index as the "responsible" one (this is
# how India's National AQI is defined), validate that the max sub-index
# reproduces the dataset's AQI, then summarise by season.
# ---------------------------------------------------------------------------

# CPCB National AQI breakpoints: (conc_low, conc_high, index_low, index_high).
# Concentrations in ug/m3 except CO in mg/m3. Bands are continuous; values above
# the top band are linearly extrapolated (the dataset's AQI exceeds 500).
AQI_BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
              (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
             (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200),
            (180, 280, 200, 300), (280, 400, 300, 400), (400, 500, 400, 500)],
    "CO": [(0, 1, 0, 50), (1, 2, 50, 100), (2, 10, 100, 200),
           (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200),
            (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2000, 400, 500)],
    "O3": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200),
           (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500)],
    "NH3": [(0, 200, 0, 50), (200, 400, 50, 100), (400, 800, 100, 200),
            (800, 1200, 200, 300), (1200, 1800, 300, 400), (1800, 2400, 400, 500)],
}

ATTRIB_POLLUTANTS: tuple[str, ...] = tuple(AQI_BREAKPOINTS.keys())

# Calendar month -> meteorological season used throughout the attribution.
_SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}
SEASON_ORDER: tuple[str, ...] = ("Winter", "Summer", "Monsoon", "Post-monsoon")


def _sub_index_series(conc: pd.Series, pollutant: str) -> pd.Series:
    """CPCB sub-index for a concentration series (vectorised, NaN-safe)."""
    bands = AQI_BREAKPOINTS[pollutant]
    out = pd.Series(np.nan, index=conc.index, dtype="float64")
    assigned = pd.Series(False, index=conc.index)
    for lo, hi, ilo, ihi in bands:
        mask = conc.notna() & ~assigned & (conc <= hi)
        out[mask] = ilo + (ihi - ilo) * (conc[mask] - lo) / (hi - lo)
        assigned |= mask
    # Extrapolate values above the top band using the last band's slope.
    lo, hi, ilo, ihi = bands[-1]
    tail = conc.notna() & ~assigned
    out[tail] = ilo + (ihi - ilo) * (conc[tail] - lo) / (hi - lo)
    return out


def compute_sub_indices(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Per-day sub-indices, computed AQI (max sub-index), and responsible pollutant.

    Returns a frame indexed like the city's rows with: Date, one column per
    pollutant sub-index, computed_aqi, responsible, plus the dataset AQI for
    validation. Days with fewer than 3 available sub-indices are dropped (CPCB
    requires a minimum set before an AQI is meaningful).
    """
    sub = _city_frame(df, city).copy()
    si_cols = {}
    for p in ATTRIB_POLLUTANTS:
        if p in sub.columns:
            si_cols[f"SI_{p}"] = _sub_index_series(sub[p], p)
    si = pd.DataFrame(si_cols, index=sub.index)

    enough = si.notna().sum(axis=1) >= 3
    si = si[enough]
    result = pd.DataFrame({"Date": sub.loc[enough, "Date"], "AQI": sub.loc[enough, "AQI"]})
    result = pd.concat([result, si], axis=1)
    result["computed_aqi"] = si.max(axis=1)
    result["responsible"] = (
        si.idxmax(axis=1).str.removeprefix("SI_")
    )
    return result.reset_index(drop=True)


def validate_computed_aqi(attrib: pd.DataFrame) -> dict[str, float]:
    """Compare computed AQI (max sub-index) against the dataset's AQI column."""
    both = attrib.dropna(subset=["AQI", "computed_aqi"])
    err = both["computed_aqi"] - both["AQI"]
    return {
        "n": int(len(both)),
        "correlation": float(both["computed_aqi"].corr(both["AQI"])),
        "mae": float(err.abs().mean()),
        "within_10pct": float((err.abs() <= 0.10 * both["AQI"]).mean() * 100),
    }


def seasonal_attribution(
    df: pd.DataFrame, city: str, bad_only: bool = False, bad_threshold: float = 200.0
) -> pd.DataFrame:
    """Share (%) of days each pollutant is responsible, by season.

    With bad_only=True, restrict to bad-air days (computed AQI > bad_threshold),
    answering "what drives the *worst* days in each season".
    """
    attrib = compute_sub_indices(df, city)
    attrib = attrib.assign(season=attrib["Date"].dt.month.map(_SEASON_BY_MONTH))
    if bad_only:
        attrib = attrib[attrib["computed_aqi"] > bad_threshold]

    counts = (
        attrib.groupby(["season", "responsible"]).size().rename("n_days").reset_index()
    )
    totals = counts.groupby("season")["n_days"].transform("sum")
    counts["share_pct"] = 100.0 * counts["n_days"] / totals
    counts["season"] = pd.Categorical(counts["season"], categories=SEASON_ORDER, ordered=True)
    return counts.sort_values(["season", "share_pct"], ascending=[True, False]).reset_index(drop=True)


def dominant_pollutant_by_season(df: pd.DataFrame, city: str, bad_only: bool = False) -> pd.DataFrame:
    """The single most-responsible pollutant per season + its share."""
    att = seasonal_attribution(df, city, bad_only=bad_only)
    idx = att.groupby("season", observed=True)["share_pct"].idxmax()
    return att.loc[idx, ["season", "responsible", "share_pct", "n_days"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-city comparison (Phase 6): is Delhi uniquely bad, or seasonally bad?
# These operate on the FULL (all-city) dataset, not the NCR-filtered one.
# ---------------------------------------------------------------------------

# Default comparison set: well-covered, credible cities spanning regions.
# (Ahmedabad is excluded by default — its AQI series looks anomalously high with
# heavy gaps; it can still be passed in explicitly.)
MAJOR_CITIES: tuple[str, ...] = (
    "Delhi", "Lucknow", "Patna", "Kolkata", "Hyderabad",
    "Chennai", "Mumbai", "Bengaluru",
)


def city_summary(
    df_all: pd.DataFrame,
    cities: tuple[str, ...] = MAJOR_CITIES,
    col: str = "AQI",
    bad_threshold: float = 200.0,
) -> pd.DataFrame:
    """Per-city level summary, sorted worst-first.

    Returns mean, median, p95, % of days above `bad_threshold` (Poor or worse),
    and the observed-day count. Cities absent from the data are skipped.
    """
    rows: list[dict] = []
    present = set(df_all["City"].unique())
    for c in cities:
        if c not in present:
            continue
        d = df_all.loc[df_all["City"] == c, col].dropna()
        if d.empty:
            continue
        rows.append(
            {
                "city": c,
                "n_days": int(d.size),
                "mean": float(d.mean()),
                "median": float(d.median()),
                "p95": float(d.quantile(0.95)),
                "pct_bad": float((d > bad_threshold).mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def seasonal_city_means(
    df_all: pd.DataFrame, cities: tuple[str, ...] = MAJOR_CITIES, col: str = "PM2.5"
) -> pd.DataFrame:
    """Per-city, per-season mean of `col` (long format, season ordered)."""
    sub = df_all[df_all["City"].isin(cities)][["City", "Date", col]].dropna(subset=[col])
    sub = sub.assign(season=sub["Date"].dt.month.map(_SEASON_BY_MONTH))
    out = sub.groupby(["City", "season"])[col].mean().reset_index(name="mean")
    out["season"] = pd.Categorical(out["season"], categories=SEASON_ORDER, ordered=True)
    return out.sort_values(["City", "season"]).reset_index(drop=True)


def seasonality_index(
    df_all: pd.DataFrame, cities: tuple[str, ...] = MAJOR_CITIES, col: str = "PM2.5"
) -> pd.DataFrame:
    """How seasonal each city is: winter mean / monsoon mean of `col`.

    A high ratio means the city's pollution is driven by a winter spike (Delhi's
    signature) rather than a steady year-round level. Sorted most-seasonal first.
    """
    means = seasonal_city_means(df_all, cities, col)
    wide = means.pivot(index="City", columns="season", values="mean")
    out = pd.DataFrame(
        {
            "winter": wide.get("Winter"),
            "monsoon": wide.get("Monsoon"),
        }
    )
    out["winter_monsoon_ratio"] = out["winter"] / out["monsoon"]
    return out.dropna().sort_values("winter_monsoon_ratio", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Health-impact layer: how Delhi's PM2.5 compares to health limits, in days
# exceeded and in cigarette-equivalents.
# ---------------------------------------------------------------------------

# PM2.5 reference levels, 24-hour mean (ug/m3) unless noted.
WHO_PM25_24H = 15.0        # WHO 2021 air-quality guideline (24h)
WHO_IT1_PM25_24H = 75.0    # WHO interim target 1 (24h)
INDIA_PM25_24H = 60.0      # India NAAQS (24h)
WHO_PM25_ANNUAL = 5.0      # WHO 2021 guideline (annual mean)
INDIA_PM25_ANNUAL = 40.0   # India NAAQS (annual mean)

# Berkeley Earth rule of thumb: ~22 ug/m3 of PM2.5 over a day ~ one cigarette.
PM25_PER_CIGARETTE = 22.0


def cigarettes_per_day(pm25: float) -> float:
    """Approximate cigarette-equivalent of a day's PM2.5 exposure."""
    return pm25 / PM25_PER_CIGARETTE


def health_exceedance(df: pd.DataFrame, city: str, col: str = "PM2.5") -> pd.DataFrame:
    """Per-year health burden: mean PM2.5, days/% over each limit, cigarettes/day.

    Uses observed days only; n_days shows how many days back each year's figures.
    """
    sub = _city_frame(df, city)[["Date", col]].dropna(subset=[col])
    sub = sub.assign(year=sub["Date"].dt.year)
    rows: list[dict] = []
    for year, g in sub.groupby("year"):
        v = g[col]
        n = len(v)
        rows.append(
            {
                "year": int(year),
                "n_days": n,
                "mean_pm25": float(v.mean()),
                "days_over_who": int((v > WHO_PM25_24H).sum()),
                "days_over_india": int((v > INDIA_PM25_24H).sum()),
                "clean_days": int((v <= WHO_PM25_24H).sum()),
                "pct_over_who": float((v > WHO_PM25_24H).mean() * 100),
                "pct_over_india": float((v > INDIA_PM25_24H).mean() * 100),
                "avg_cigarettes_day": float(v.mean() / PM25_PER_CIGARETTE),
            }
        )
    return pd.DataFrame(rows)


def health_summary(df: pd.DataFrame, city: str, col: str = "PM2.5") -> dict[str, float]:
    """Headline health numbers across the whole record for one city."""
    v = _city_frame(df, city)[col].dropna()
    if v.empty:
        raise ValueError(f"No {col} data for {city}.")
    mean = float(v.mean())
    return {
        "n_days": int(v.size),
        "mean_pm25": mean,
        "pct_over_who": float((v > WHO_PM25_24H).mean() * 100),
        "pct_over_india": float((v > INDIA_PM25_24H).mean() * 100),
        "pct_clean": float((v <= WHO_PM25_24H).mean() * 100),
        "x_who_annual": mean / WHO_PM25_ANNUAL,
        "x_india_annual": mean / INDIA_PM25_ANNUAL,
        "avg_cigarettes_day": mean / PM25_PER_CIGARETTE,
        "cigarettes_year": mean / PM25_PER_CIGARETTE * 365,
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

    # ---- Phase 5: source attribution ------------------------------------
    print(f"\n=== Source attribution (CPCB sub-indices) - {city} ===")
    v = validate_computed_aqi(compute_sub_indices(df, city))
    print(
        f"Validation: computed AQI (max sub-index) vs dataset AQI on {v['n']} days "
        f"-> corr={v['correlation']:.3f}, MAE={v['mae']:.1f}, "
        f"within 10%: {v['within_10pct']:.0f}%"
    )
    print("\nDominant pollutant on BAD-air days (computed AQI > 200), by season:")
    dom = dominant_pollutant_by_season(df, city, bad_only=True)
    print(dom.round(1).to_string(index=False))
    winter = dom[dom["season"] == "Winter"]
    summer = dom[dom["season"] == "Summer"]
    if not winter.empty and not summer.empty:
        w, s = winter.iloc[0], summer.iloc[0]
        print(
            f"-> Bad winter days are driven by {w['responsible']} "
            f"({w['share_pct']:.0f}%); bad summer days by {s['responsible']} "
            f"({s['share_pct']:.0f}%)."
        )

    # ---- Phase 6: multi-city comparison ---------------------------------
    from src.data_load import load_raw

    df_all = load_raw()
    print("\n=== Multi-city comparison (mean AQI, worst-first) ===")
    summ_c = city_summary(df_all)
    print(summ_c.round(1).to_string(index=False))

    print("\n=== Seasonality (winter/monsoon PM2.5 ratio, most-seasonal first) ===")
    seas = seasonality_index(df_all)
    print(seas.round(1).to_string(index=False))

    delhi_mean = summ_c.loc[summ_c["city"] == "Delhi", "mean"].iloc[0]
    others_mean = summ_c.loc[summ_c["city"] != "Delhi", "mean"].mean()
    delhi_ratio = seas.loc[seas["City"] == "Delhi", "winter_monsoon_ratio"].iloc[0]
    most_seasonal = seas.iloc[0]
    print(
        f"-> Delhi is uniquely POLLUTED: mean AQI {delhi_mean:.0f} is "
        f"{delhi_mean / others_mean:.1f}x the other majors' average ({others_mean:.0f}). "
        f"But the winter spike is a North-India pattern, not Delhi's alone — Delhi's "
        f"winter PM2.5 is {delhi_ratio:.1f}x monsoon, similar to Patna/Lucknow/Kolkata "
        f"(most seasonal: {most_seasonal['City']} {most_seasonal['winter_monsoon_ratio']:.1f}x)."
    )

    # ---- Health-impact layer --------------------------------------------
    print(f"\n=== Health impact (PM2.5 vs limits) - {city} ===")
    he = health_exceedance(df, city)
    print(he.round(1).to_string(index=False))
    hs = health_summary(df, city)
    print(
        f"-> {city} averages PM2.5 {hs['mean_pm25']:.0f} ug/m3 "
        f"({hs['x_who_annual']:.0f}x the WHO annual guideline, "
        f"{hs['x_india_annual']:.1f}x India's). {hs['pct_over_who']:.0f}% of days exceed "
        f"the WHO 24h limit and {hs['pct_over_india']:.0f}% exceed India's; only "
        f"{hs['pct_clean']:.0f}% are 'clean'. That's ~{hs['avg_cigarettes_day']:.1f} "
        f"cigarettes/day, ~{hs['cigarettes_year']:.0f} a year, from breathing alone."
    )
