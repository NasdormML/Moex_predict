.PHONY: install lint api mlflow run clean

install:
	pip install -r requirements.txt

lint:
	flake8 app

api:
	# Запустить только API (порт можно переопределить через API_PORT)
	uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

mlflow:
	# Запустить только MLflow UI (порт можно переопределить через MLFLOW_PORT)
	mlflow ui --host 0.0.0.0 --port $${MLFLOW_PORT:-5001}

run:
	# Установка зависимостей
	# Запустить MLflow UI в фоне и сразу же API
	mlflow ui --host 0.0.0.0 --port $${MLFLOW_PORT:-5001} & uvicorn app.main:app --reload --host 0.0.0.0 --port $${API_PORT:-8000}

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
