"""Download the Kaggle 'Air Quality Data in India' dataset into data/.

Requires Kaggle API credentials. With the newer KGAT_ token format, save the
token to ~/.kaggle/access_token (the client reads it automatically). With the
older format, place kaggle.json in ~/.kaggle/. See:
https://www.kaggle.com/settings/api

Run:  python -m src.download_data
"""
from __future__ import annotations

from pathlib import Path

DATASET = "rohanrao/air-quality-data-in-india"
TARGET_FILE = "city_day.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(dataset: str = DATASET, data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset into data_dir; return path to city_day.csv."""
    # Import here so the rest of the project doesn't hard-depend on the kaggle
    # client just to import this module.
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated. Downloading '{dataset}' -> {data_dir} ...")
    api.dataset_download_files(dataset, path=str(data_dir), unzip=True, quiet=False)

    target = data_dir / TARGET_FILE
    if not target.exists():
        found = sorted(p.name for p in data_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"Expected {TARGET_FILE} in {data_dir} but it's missing. "
            f"CSV files present: {found}"
        )
    size_mb = target.stat().st_size / 1e6
    print(f"OK: {target} ({size_mb:.1f} MB)")
    return target


if __name__ == "__main__":
    download()
