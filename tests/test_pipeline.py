from churn import config
from churn.data import clean
from churn.train import train

from .conftest import fake_raw


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
