.PHONY: download-data train run test lint format docker-build docker-run

download-data:
	python scripts/download_data.py

train:
	python -m src.training.train

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

docker-build:
	docker build -t pathology-prediction-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name pathology-api pathology-prediction-api:latest
