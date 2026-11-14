"""Download and load the Telco customer churn dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

from churn.config import DATA_URL, ID_COLUMN, RAW_CSV, TARGET


def download(url: str = DATA_URL, dest: Path = RAW_CSV) -> Path:
    """Download the raw CSV if it is not already on disk."""
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {url} -> {dest}")
    return dest


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and encode the target as 0/1."""
    df = df.drop(columns=[ID_COLUMN], errors="ignore").copy()
    # TotalCharges is blank for brand-new customers (tenure 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load(path: Path = RAW_CSV) -> pd.DataFrame:
    return clean(pd.read_csv(path))


if __name__ == "__main__":
    download()
