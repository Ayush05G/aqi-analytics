"""Fetch the Kaggle 'Air Quality Data in India' dataset into data/.

Two entry points:
- download(): grab city_day.csv (used by the app + CLI). Defaults to a
  single-file download (~2.6 MB) rather than the full 73 MB archive.
- ensure_city_day(): return the local CSV path, downloading it only if missing.
  This is what the deployed app calls on a cold start.

Credentials: the Kaggle client reads ~/.kaggle/access_token (newer KGAT_ token)
or ~/.kaggle/kaggle.json, or the KAGGLE_USERNAME/KAGGLE_KEY env vars. On
Streamlit Cloud, set the token in st.secrets and call configure_credentials()
before downloading (the app does this for you).

CLI:  python -m src.download_data            # just city_day.csv
      python -m src.download_data --all      # the whole dataset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET = "rohanrao/air-quality-data-in-india"
TARGET_FILE = "city_day.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def configure_credentials(
    username: str | None = None,
    key: str | None = None,
    api_token: str | None = None,
) -> None:
    """Set Kaggle env-var credentials if provided (e.g. from st.secrets).

    Supports both the classic username/key pair and the newer single KGAT_ token
    (KAGGLE_API_TOKEN), so either kind of secret works on Streamlit Cloud.
    """
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key
    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token


def _authenticated_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download(
    dataset: str = DATASET,
    data_dir: Path = DATA_DIR,
    single_file: bool = True,
) -> Path:
    """Download city_day.csv (single_file) or the whole dataset; return its path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    api = _authenticated_api()
    print(f"Authenticated. Downloading '{dataset}' -> {data_dir} ...")
    if single_file:
        api.dataset_download_file(
            dataset, file_name=TARGET_FILE, path=str(data_dir), force=True
        )
        # Single-file downloads may arrive zipped; unzip if so.
        zipped = data_dir / f"{TARGET_FILE}.zip"
        if zipped.exists():
            import zipfile

            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(data_dir)
            zipped.unlink()
    else:
        api.dataset_download_files(dataset, path=str(data_dir), unzip=True, quiet=False)

    target = data_dir / TARGET_FILE
    if not target.exists():
        found = sorted(p.name for p in data_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"Expected {TARGET_FILE} in {data_dir} but it's missing. CSVs present: {found}"
        )
    print(f"OK: {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def ensure_city_day(data_dir: Path = DATA_DIR) -> Path:
    """Return path to city_day.csv, downloading it only if not already present."""
    target = data_dir / TARGET_FILE
    if target.exists():
        return target
    return download(data_dir=data_dir, single_file=True)


if __name__ == "__main__":
    download(single_file="--all" not in sys.argv)
