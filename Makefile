.PHONY: install lint api mlflow run clean

install:
	python3 -m venv venv && \
	source venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

lint:
	flake8 app

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

mlflow:
	mlflow ui --host 0.0.0.0 --port $${MLFLOW_PORT:-5001}

run:
	mlflow ui --host 0.0.0.0 --port $${MLFLOW_PORT:-5001} & uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
