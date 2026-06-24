"""Trustworthy data layer for the Delhi-NCR air quality project.

Loads the Kaggle city_day.csv, parses dates, and filters to Delhi-NCR. Pure data
handling: it reports data quality but never silently fills gaps — missing values
stay missing here and are handled deliberately downstream.

Run as a script for a sanity report:  python -m src.data_load
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "city_day.csv"

# Cities that make up the National Capital Region. Only those actually present in
# the dataset are kept; we don't assume coverage (this dataset has Delhi +
# Gurugram only, but listing the full NCR keeps the filter honest if the source
# is ever swapped for a richer one).
NCR_CITIES: tuple[str, ...] = (
    "Delhi",
    "Gurugram",
    "Faridabad",
    "Ghaziabad",
    "Noida",
    "Greater Noida",
    "Meerut",
    "Sonipat",
    "Panipat",
    "Rohtak",
    "Bahadurgarh",
)

# Pollutant + AQI columns we care about for analysis and quality reporting.
KEY_POLLUTANTS: tuple[str, ...] = (
    "PM2.5",
    "PM10",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
)
KEY_COLUMNS: tuple[str, ...] = KEY_POLLUTANTS + ("AQI",)


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read city_day.csv and parse Date to datetime, sorted by city then date.

    Raises FileNotFoundError with a hint to run the downloader if the CSV is
    missing, and ValueError if any Date fails to parse.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad = int(df["Date"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have an unparseable Date.")

    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def filter_ncr(df: pd.DataFrame, cities: tuple[str, ...] = NCR_CITIES) -> pd.DataFrame:
    """Keep only NCR cities that are actually present in the data."""
    present = [c for c in cities if c in set(df["City"].unique())]
    if not present:
        raise ValueError(
            f"None of the NCR cities {cities} are present in the data."
        )
    return df[df["City"].isin(present)].reset_index(drop=True)


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, parse, and filter to Delhi-NCR. No imputation — gaps are preserved."""
    return filter_ncr(load_raw(path))


def null_rates(df: pd.DataFrame, columns: tuple[str, ...] = KEY_COLUMNS) -> pd.DataFrame:
    """Per-column null count and percentage, sorted worst-first."""
    n = len(df)
    out = pd.DataFrame(
        {
            "n_missing": [int(df[c].isna().sum()) for c in columns],
            "pct_missing": [
                round(100 * df[c].isna().mean(), 1) if n else 0.0 for c in columns
            ],
        },
        index=list(columns),
    )
    return out.sort_values("pct_missing", ascending=False)


def sanity_report(df: pd.DataFrame) -> None:
    """Print row count, date range, cities, and per-column null rates."""
    print("=" * 60)
    print("DELHI-NCR DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Rows:        {len(df):,}")
    print(f"Date range:  {df['Date'].min().date()} -> {df['Date'].max().date()}")

    per_city = df.groupby("City").size().sort_values(ascending=False)
    print("Cities:")
    for city, n in per_city.items():
        span = df.loc[df["City"] == city, "Date"]
        print(f"  - {city:<12} {n:>5,} rows  ({span.min().date()} -> {span.max().date()})")

    print("\nNull rates for key pollutants + AQI (worst first):")
    print(null_rates(df).to_string())
    print("=" * 60)


if __name__ == "__main__":
    sanity_report(load_clean())
