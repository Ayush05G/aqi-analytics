"""Fetch recent (2025+) daily city concentrations from the OpenAQ v3 API.

OpenAQ's India ingestion has a gap from ~2022 to Feb 2025, so this only covers
the recent end (2025-02 -> today). For each target city it finds nearby active
stations, pulls each sensor's daily values, converts gas units to the CPCB basis
(CO -> mg/m3, others -> ug/m3), averages stations to a daily city value, and
computes AQI as the max CPCB sub-index.

Needs an OpenAQ API key at ~/.openaq/api_key (or api_key.txt). Local-only — the
output is spliced into the committed dataset by src/extend_dataset.py.

Run:  python -m src.fetch_openaq
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.analysis import AQI_BREAKPOINTS, _sub_index_series
from src.build_city_day import AQI_BUCKETS, _bucket

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "openaq_recent.csv"  # git-ignored
BASE = "https://api.openaq.org/v3"
START = "2025-01-01"
RADIUS_M = 25000
MAX_STATIONS = 20

# City centres (lat, lon).
CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090), "Gurugram": (28.4595, 77.0266),
    "Mumbai": (19.0760, 72.8777), "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707), "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
}

# OpenAQ parameter name -> our column.
PARAM_MAP = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "no": "NO",
    "so2": "SO2", "co": "CO", "o3": "O3",
}

# ppb -> target unit conversion factors at 25 C, 1 atm (factor = MW / 24.45);
# CO additionally /1000 because CPCB uses mg/m3 for CO.
PPB_FACTOR = {"SO2": 2.62, "NO2": 1.88, "O3": 1.96, "NO": 1.23, "CO": 0.001145}


def _load_key() -> str:
    d = Path.home() / ".openaq"
    for name in ("api_key", "api_key.txt"):
        p = d / name
        if p.exists():
            return p.read_text().strip()
    raise FileNotFoundError("OpenAQ API key not found at ~/.openaq/api_key[.txt]")


def _convert(value: float, our_param: str, unit: str) -> float:
    """Convert a value to the CPCB basis (CO mg/m3, gases ug/m3, PM unchanged)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if unit and unit.lower() in ("ppb", "ppm"):
        v = value * (1000 if unit.lower() == "ppm" else 1)
        return v * PPB_FACTOR.get(our_param, 1.0)
    return value  # already ug/m3 (or mg/m3 for CO, which CPCB wants)


def _session(key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-API-Key": key})
    return s


def _get(s: requests.Session, url: str, params: dict | None = None) -> dict:
    for attempt in range(5):
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 429:  # rate limited
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _city_sensors(s: requests.Session, center: tuple[float, float]) -> list[tuple[int, str, str]]:
    """Active (2025+) sensors near a city: list of (sensor_id, our_param, unit)."""
    locs = _get(s, f"{BASE}/locations", {
        "coordinates": f"{center[0]},{center[1]}", "radius": RADIUS_M, "limit": 1000,
    })["results"]
    active = [l for l in locs if (l.get("datetimeLast") or {}).get("utc", "") >= "2025"]
    active.sort(key=lambda l: (l.get("datetimeLast") or {}).get("utc", ""), reverse=True)
    sensors: list[tuple[int, str, str]] = []
    for loc in active[:MAX_STATIONS]:
        for sen in _get(s, f"{BASE}/locations/{loc['id']}/sensors")["results"]:
            pname = sen["parameter"]["name"]
            last = (sen.get("datetimeLast") or {}).get("utc", "")
            if pname in PARAM_MAP and last >= "2025":
                sensors.append((sen["id"], PARAM_MAP[pname], sen["parameter"].get("units", "")))
    return sensors


def _sensor_daily(s: requests.Session, sid: int, our_param: str, unit: str) -> list[tuple[str, float]]:
    """Daily (date, converted value) for a sensor, from START onward."""
    rows: list[tuple[str, float]] = []
    page = 1
    while True:
        res = _get(s, f"{BASE}/sensors/{sid}/days", {"limit": 1000, "page": page}).get("results", [])
        if not res:
            break
        for x in res:
            day = x["period"]["datetimeFrom"]["utc"][:10]
            if day >= START:
                rows.append((day, _convert(x.get("value"), our_param, unit)))
        if len(res) < 1000:
            break
        page += 1
    return rows


def _compute_aqi(daily: pd.DataFrame) -> pd.Series:
    sis = {}
    for p in ("PM2.5", "PM10", "NO2", "SO2", "O3", "CO"):
        if p in daily.columns and p in AQI_BREAKPOINTS:
            sis[p] = _sub_index_series(daily[p], p)
    si = pd.DataFrame(sis, index=daily.index)
    enough = si.notna().sum(axis=1) >= 3
    return si.max(axis=1).where(enough)


def fetch() -> pd.DataFrame:
    s = _session(_load_key())
    frames = []
    for city, center in CITY_CENTERS.items():
        sensors = _city_sensors(s, center)
        print(f"{city}: {len(sensors)} active sensors", flush=True)
        long_rows = []
        for sid, param, unit in sensors:
            for day, val in _sensor_daily(s, sid, param, unit):
                long_rows.append((day, param, val))
        if not long_rows:
            print(f"  {city}: no recent data", flush=True)
            continue
        lf = pd.DataFrame(long_rows, columns=["Date", "param", "value"])
        wide = lf.groupby(["Date", "param"])["value"].mean().unstack("param")
        wide.index = pd.to_datetime(wide.index)
        wide["AQI"] = _compute_aqi(wide)
        wide["AQI_Bucket"] = wide["AQI"].map(_bucket)
        wide["City"] = city
        frames.append(wide.reset_index())
        print(f"  {city}: {len(wide)} days, {wide.index.min().date()} -> {wide.index.max().date()}", flush=True)

    result = pd.concat(frames, ignore_index=True).sort_values(["City", "Date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} — {len(result):,} rows, {result['City'].nunique()} cities", flush=True)
    return result


if __name__ == "__main__":
    fetch()
