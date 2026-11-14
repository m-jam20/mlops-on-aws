.PHONY: install lint format test data train

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

train:  ## Train the baseline model and print metrics
	uv run python -m churn.train
