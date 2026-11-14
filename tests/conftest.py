import numpy as np
import pandas as pd
import pytest

from churn import config


def fake_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Small random table with the same columns as the real CSV."""
    rng = np.random.default_rng(seed)
    yes_no = ["Yes", "No"]
    df = pd.DataFrame(
        {
            config.ID_COLUMN: [f"id-{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "SeniorCitizen": rng.integers(0, 2, n),
            config.TARGET: rng.choice(yes_no, n),
        }
    )
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).astype(str)
    df.loc[0, "TotalCharges"] = " "  # blank value, like in the real data
    for col in config.CATEGORICAL_FEATURES:
        df[col] = rng.choice(yes_no, n)
    return df


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    """Train a tiny model on fake data and point the API at it."""
    import joblib

    from churn.data import clean
    from churn.train import train

    pipeline, _ = train(clean(fake_raw()))
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    return path
