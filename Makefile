.PHONY: install lint format test data train mlflow-ui

install:  ## Create .venv, install deps and git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

data:  ## Download the dataset into data/raw
	uv run python -m churn.data

train:  ## Train the model and log the run to MLflow
	uv run python -m churn.train

mlflow-ui:  ## Open the MLflow UI at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
