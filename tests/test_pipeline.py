import numpy as np
import pandas as pd

from churn import config
from churn.data import clean
from churn.train import train


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


def test_clean_fixes_types():
    df = clean(fake_raw())
    assert config.ID_COLUMN not in df.columns
    assert df["TotalCharges"].dtype == float
    assert df.loc[0, "TotalCharges"] == 0.0
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_train_returns_metrics():
    _, metrics = train(clean(fake_raw()))
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
