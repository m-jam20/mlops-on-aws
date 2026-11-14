.PHONY: install lint format test data train mlflow-ui aws-local-up aws-local-down tf-local serve

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

serve:  ## Run the API at http://127.0.0.1:8000 (docs at /docs)
	uv run uvicorn churn.api:app --reload --port 8000

aws-local-up:  ## Start a local fake AWS (Moto) on port 4566
	docker run -d --rm --name moto -p 4566:5000 motoserver/moto:5.2.5

aws-local-down:
	docker stop moto

tf-local:  ## Terraform init + plan + apply against local fake AWS
	terraform -chdir=infra/local init -input=false
	terraform -chdir=infra/local plan -out=plan.tfplan
	terraform -chdir=infra/local apply plan.tfplan
