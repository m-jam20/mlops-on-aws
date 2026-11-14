"""Train a baseline churn model."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config
from churn.data import load


def build_pipeline(C: float = 1.0) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
    # class_weight="balanced": churners are the minority class, so give them more weight.
    model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def train(df: pd.DataFrame, C: float = 1.0) -> tuple[Pipeline, dict[str, float]]:
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    pipeline = build_pipeline(C=C).fit(X_train, y_train)
    return pipeline, evaluate(pipeline, X_test, y_test)


def main() -> None:
    _, metrics = train(load())
    for name, value in metrics.items():
        print(f"{name:>9}: {value:.3f}")


if __name__ == "__main__":
    main()
