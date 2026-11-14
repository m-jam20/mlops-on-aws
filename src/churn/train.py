"""Train a baseline churn model and track the run with MLflow."""

import joblib
import mlflow
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


def main(C: float = 1.0) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = load()
    with mlflow.start_run(run_name="logreg-baseline") as run:
        pipeline, metrics = train(df, C=C)
        mlflow.log_params({"model": "LogisticRegression", "C": C, "rows": len(df)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=df[config.FEATURES].head(3),
            registered_model_name="churn-model",
        )

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    for name, value in metrics.items():
        print(f"{name:>9}: {value:.3f}")
    print(f"MLflow run: {run.info.run_id}  (view with: make mlflow-ui)")
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
