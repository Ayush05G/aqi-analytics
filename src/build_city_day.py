"""Build an EXTENDED city_day dataset (2015-2023) from real CPCB hourly station data.

The published rohanrao city_day.csv stops at 2020-07. This script reconstructs the
same daily city-level schema, extended to 2023, from the raw hourly station files
in the Kaggle dataset `abhisheksjha/time-series-air-quality-data-of-india-2010-2023`
(the genuine CPCB CAAQMS source rohanrao was derived from).

Pipeline (run locally; the cloud app just loads the committed output):
1. Map each target city to its CPCB station IDs via rohanrao's stations.csv.
2. Download those stations' hourly files (cached, resumable) into data/ (git-ignored).
3. Aggregate hourly -> daily per station, then average stations -> daily city values.
   O3 and CO additionally get a daily 8-hour maximum (the window CPCB uses for them).
4. Compute the AQI as the max CPCB sub-index (8h-max for O3/CO, 24h-mean otherwise).
5. Write data_processed/city_day_2015_2023.csv (committed, ~5 MB).

Run:  python -m src.build_city_day
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import AQI_BREAKPOINTS, _sub_index_series

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "stations_raw"          # git-ignored
STATIONS_CSV = ROOT / "data" / "stations.csv"      # from rohanrao dataset
OUT_PATH = ROOT / "data_processed" / "city_day_2015_2023.csv"  # committed
DATASET = "abhisheksjha/time-series-air-quality-data-of-india-2010-2023"
START = pd.Timestamp("2015-01-01")

TARGET_CITIES: tuple[str, ...] = (
    "Delhi", "Gurugram", "Mumbai", "Kolkata", "Bengaluru",
    "Hyderabad", "Chennai", "Lucknow", "Patna",
)

# abhisheksjha column -> our schema name.
COL_MAP = {
    "PM2.5 (ug/m3)": "PM2.5", "PM10 (ug/m3)": "PM10", "NO (ug/m3)": "NO",
    "NO2 (ug/m3)": "NO2", "NOx (ppb)": "NOx", "NH3 (ug/m3)": "NH3",
    "SO2 (ug/m3)": "SO2", "CO (mg/m3)": "CO", "Ozone (ug/m3)": "O3",
    "Benzene (ug/m3)": "Benzene",
}
POLLUTANTS = tuple(COL_MAP.values())

# Output schema (mirrors rohanrao city_day; Toluene/Xylene absent in this source).
OUT_COLUMNS = [
    "City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
    "O3", "Benzene", "AQI", "AQI_Bucket",
]

AQI_BUCKETS = [
    (0, 50, "Good"), (50, 100, "Satisfactory"), (100, 200, "Moderate"),
    (200, 300, "Poor"), (300, 400, "Very Poor"), (400, float("inf"), "Severe"),
]


def station_ids_by_city() -> dict[str, list[str]]:
    """{city: [station_id, ...]} for the target cities, from stations.csv."""
    s = pd.read_csv(STATIONS_CSV)
    s = s[s["City"].isin(TARGET_CITIES)]
    return {c: g["StationId"].tolist() for c, g in s.groupby("City")}


def download_stations(station_ids: list[str]) -> None:
    """Download (and unzip) each station's hourly file into RAW_DIR if absent."""
    import zipfile

    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for i, sid in enumerate(station_ids, 1):
        target = RAW_DIR / f"{sid}.csv"
        if target.exists():
            continue
        print(f"  [{i}/{len(station_ids)}] downloading {sid} …", flush=True)
        api.dataset_download_file(DATASET, file_name=f"{sid}.csv", path=str(RAW_DIR))
        zipped = RAW_DIR / f"{sid}.csv.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(RAW_DIR)
            zipped.unlink()


def _station_daily(path: Path) -> pd.DataFrame | None:
    """Hourly station file -> daily frame: per-pollutant means + O3/CO 8h-max."""
    raw = pd.read_csv(path)
    if "From Date" not in raw.columns:
        return None
    ts = pd.to_datetime(raw["From Date"], errors="coerce")
    raw = raw.assign(_ts=ts).dropna(subset=["_ts"]).set_index("_ts").sort_index()
    present = {src: dst for src, dst in COL_MAP.items() if src in raw.columns}
    hourly = raw[list(present)].rename(columns=present).apply(pd.to_numeric, errors="coerce")

    daily = hourly.resample("D").mean()
    # 8-hour rolling mean, then daily max — CPCB's window for O3 and CO.
    for p in ("O3", "CO"):
        if p in hourly.columns:
            roll8 = hourly[p].rolling(8, min_periods=6).mean()
            daily[f"{p}_8h"] = roll8.resample("D").max()
    return daily


def _city_daily(station_ids: list[str]) -> pd.DataFrame:
    """Average available stations -> one daily row per date for a city."""
    frames = []
    for sid in station_ids:
        path = RAW_DIR / f"{sid}.csv"
        if not path.exists():
            continue
        d = _station_daily(path)
        if d is not None and not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.groupby(level=0).mean()  # mean across stations per date


def _compute_aqi(city_daily: pd.DataFrame) -> pd.Series:
    """Max CPCB sub-index per day (8h-max for O3/CO, 24h-mean for the rest)."""
    inputs = {
        "PM2.5": city_daily.get("PM2.5"),
        "PM10": city_daily.get("PM10"),
        "NO2": city_daily.get("NO2"),
        "SO2": city_daily.get("SO2"),
        "NH3": city_daily.get("NH3"),
        "O3": city_daily.get("O3_8h"),
        "CO": city_daily.get("CO_8h"),
    }
    sis = {}
    for p, series in inputs.items():
        if series is not None and p in AQI_BREAKPOINTS:
            sis[p] = _sub_index_series(series, p)
    si_df = pd.DataFrame(sis, index=city_daily.index)
    enough = si_df.notna().sum(axis=1) >= 3  # CPCB needs a minimum set
    return si_df.max(axis=1).where(enough)


def _bucket(aqi: float) -> str | float:
    if pd.isna(aqi):
        return np.nan
    for lo, hi, name in AQI_BUCKETS:
        if aqi <= hi:
            return name
    return "Severe"


def build() -> pd.DataFrame:
    ids_by_city = station_ids_by_city()
    all_ids = [sid for ids in ids_by_city.values() for sid in ids]
    print(f"Downloading {len(all_ids)} station files for {len(ids_by_city)} cities …")
    download_stations(all_ids)

    rows = []
    for city, ids in ids_by_city.items():
        cd = _city_daily(ids)
        if cd.empty:
            print(f"  {city}: no data, skipped")
            continue
        cd["AQI"] = _compute_aqi(cd)
        cd["AQI_Bucket"] = cd["AQI"].map(_bucket)
        cd = cd.reset_index().rename(columns={"_ts": "Date", "index": "Date"})
        cd["City"] = city
        for col in OUT_COLUMNS:
            if col not in cd.columns:
                cd[col] = np.nan
        out = cd[OUT_COLUMNS]
        out = out[out["Date"] >= START]
        rows.append(out)
        print(f"  {city}: {len(out):,} days, {out['Date'].min().date()} -> {out['Date'].max().date()}")

    result = pd.concat(rows).sort_values(["City", "Date"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} — {len(result):,} rows, {result['City'].nunique()} cities, "
          f"{result['AQI'].notna().mean() * 100:.0f}% AQI non-null")
    return result


if __name__ == "__main__":
    build()
