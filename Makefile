.PHONY: install lint api mlflow run run-prod clean

install:
	python3 -m venv venv && \
	source venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

lint:
	flake8 app

# Development: только API
api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

# Development: только MLflow UI
mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 0.0.0.0 \
		--port $${MLFLOW_PORT:-5001} \
		--serve-artifacts

run:
	@mkdir -p mlartifacts history logs
	@echo "Starting MLflow server on port $${MLFLOW_PORT:-5001}..."
	@mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 0.0.0.0 \
		--port $${MLFLOW_PORT:-5001} \
		--serve-artifacts > logs/mlflow.log 2>&1 & \
		echo $$! > .mlflow.pid
	@sleep 3
	@echo "MLflow started. Logs: logs/mlflow.log"
	@echo "Starting API server on port $${API_PORT:-8000}..."
	@trap 'kill `cat .mlflow.pid` 2>/dev/null; rm -f .mlflow.pid' EXIT; \
	uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

# Production: с workers
run-prod:
	@mkdir -p mlartifacts history logs
	@echo "Starting MLflow server..."
	@mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 0.0.0.0 \
		--port $${MLFLOW_PORT:-5001} \
		--serve-artifacts > logs/mlflow.log 2>&1 & \
		echo $$! > .mlflow.pid
	@sleep 3
	@echo "Starting API server (production)..."
	@trap 'kill `cat .mlflow.pid` 2>/dev/null; rm -f .mlflow.pid' EXIT; \
	uvicorn app.main:app --host 0.0.0.0 --port $${API_PORT:-8000} --workers $${UVICORN_WORKERS:-2}

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f .mlflow.pid