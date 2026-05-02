.PHONY: install data validate cohort train api app monitor mlflow-ui lint format test check all

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
UVICORN ?= $(PYTHON) -m uvicorn
STREAMLIT ?= $(PYTHON) -m streamlit
MLFLOW ?= $(PYTHON) -m mlflow
RUFF ?= $(PYTHON) -m ruff

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.data.generate_synthetic_amr --n-rows 10000 --output data/raw/synthetic_amr.csv

validate:
	$(PYTHON) -m src.data.validate_schema --input data/raw/synthetic_amr.csv

cohort:
	$(PYTHON) -m src.data.build_cohort --input data/raw/synthetic_amr.csv --output data/processed/cohort.csv --infection-site urinary

train:
	$(PYTHON) -m src.training.train --input data/processed/cohort.csv --model-output models/treatment_response_model.joblib

api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

app:
	$(STREAMLIT) run app/streamlit_app.py

monitor:
	$(PYTHON) -m src.monitoring.drift_report --reference data/reference/reference_batch.csv --current data/processed/cohort.csv --output reports/drift_report.md

mlflow-ui:
	$(MLFLOW) ui --host 0.0.0.0 --port 5000

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	$(PYTEST)

check: lint test

all: data validate cohort train monitor test
