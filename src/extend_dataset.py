"""Splice the three real sources into one extended daily dataset (2015 -> today).

Priority, per (City, Date):
- Concentrations (PM2.5, PM10, ...): abhisheksjha (<=2023-03) then OpenAQ (>=2025-02).
  The 2023-04..2025-01 stretch has no public concentration source -> stays NaN.
- AQI: abhisheksjha (<=2023-03) then the official CPCB bulletin (2023-04..2025-04)
  then OpenAQ-computed (>2025-04). So AQI is CONTINUOUS to the present.

Inputs:
- data_processed/city_day_2015_2023.csv   (from src/build_city_day)
- data/aqi_bulletin.csv                    (Kaggle saikiranudayana, AQI + prominent pollutant)
- data/openaq_recent.csv                   (from src/fetch_openaq)

Output: data_processed/city_day_2015_2026.csv (committed).

Run:  python -m src.fetch_openaq && python -m src.extend_dataset
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import AQI_BREAKPOINTS, _sub_index_series
from src.build_city_day import OUT_COLUMNS, TARGET_CITIES, _bucket

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "data_processed" / "city_day_2015_2023.csv"
BULLETIN_PATH = ROOT / "data" / "aqi_bulletin.csv"
OPENAQ_PATH = ROOT / "data" / "openaq_recent.csv"
OUT_PATH = ROOT / "data_processed" / "city_day_2015_2026.csv"

CONC_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene"]
# OpenAQ pollutants we trust: reported in ug/m3 with no unit conversion needed.
# CO/SO2 come back in inconsistent ppb units across stations, so we don't carry
# them from OpenAQ (they stay NaN for the 2025+ stretch).
TRUSTED_OAQ = ["PM2.5", "PM10", "NO2", "O3"]
BASE_END = pd.Timestamp("2023-03-31")  # abhisheksjha coverage end


def _load_base() -> pd.DataFrame:
    df = pd.read_csv(BASE_PATH, parse_dates=["Date"])
    return df[df["Date"] <= BASE_END]


def _load_bulletin() -> pd.DataFrame:
    """Official CPCB daily bulletin: AQI only, for our cities, after BASE_END."""
    b = pd.read_csv(BULLETIN_PATH)
    b["Date"] = pd.to_datetime(b["date"], format="%d-%m-%Y", errors="coerce")
    b = b[b["area"].isin(TARGET_CITIES) & b["Date"].notna()]
    b = b[b["Date"] > BASE_END]
    b = b.rename(columns={"area": "City", "aqi_value": "AQI"})
    b["AQI"] = pd.to_numeric(b["AQI"], errors="coerce")
    # One row per city-day (bulletin should already be unique).
    b = b.groupby(["City", "Date"], as_index=False)["AQI"].mean()
    return b


def _load_openaq() -> pd.DataFrame:
    if not OPENAQ_PATH.exists():
        raise FileNotFoundError(f"{OPENAQ_PATH} missing — run `python -m src.fetch_openaq` first.")
    o = pd.read_csv(OPENAQ_PATH, parse_dates=["Date"])
    return o


def build() -> pd.DataFrame:
    base = _load_base().set_index(["City", "Date"])
    bulletin = _load_bulletin().set_index(["City", "Date"])
    openaq = _load_openaq().set_index(["City", "Date"])

    # Full (City, Date) index across all sources.
    full_idx = base.index.union(bulletin.index).union(openaq.index)
    out = pd.DataFrame(index=full_idx)

    # Concentrations: base first; for the recent end fill only TRUSTED_OAQ
    # pollutants from OpenAQ (CO/SO2/NO/NOx/NH3/Benzene stay NaN there).
    for col in CONC_COLS:
        base_col = base[col] if col in base.columns else pd.Series(dtype="float64")
        out[col] = base_col.reindex(full_idx)
        if col in TRUSTED_OAQ and col in openaq.columns:
            out[col] = out[col].combine_first(openaq[col].reindex(full_idx))

    # Fallback AQI recomputed from the final (trusted) concentrations — used only
    # for the 2025+ tail beyond the bulletin, so it's PM/NO2/O3-driven.
    sis = {p: _sub_index_series(out[p], p) for p in ("PM2.5", "PM10", "NO2", "O3", "SO2", "CO")
           if p in out.columns and p in AQI_BREAKPOINTS}
    si_df = pd.DataFrame(sis, index=out.index)
    computed_aqi = si_df.max(axis=1).where(si_df.notna().sum(axis=1) >= 3)

    # AQI priority: base (stations) -> CPCB bulletin -> recomputed (OpenAQ tail).
    aqi = base["AQI"].reindex(full_idx)
    aqi = aqi.combine_first(bulletin["AQI"].reindex(full_idx))
    aqi = aqi.combine_first(computed_aqi)
    out["AQI"] = aqi
    out["AQI_Bucket"] = out["AQI"].map(_bucket)

    out = out.reset_index().sort_values(["City", "Date"]).reset_index(drop=True)
    out = out[OUT_COLUMNS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH} — {len(out):,} rows, {out['City'].nunique()} cities")
    print(f"  Date range: {out['Date'].min().date()} -> {out['Date'].max().date()}")
    print(f"  AQI non-null: {out['AQI'].notna().mean()*100:.0f}%  |  "
          f"PM2.5 non-null: {out['PM2.5'].notna().mean()*100:.0f}%")
    return out


if __name__ == "__main__":
    build()
