.PHONY: up down install data train test all

up:            ## start postgres+pgvector, redis, mlflow
	docker compose up -d

down:
	docker compose down

install:
	pip install -r requirements.txt

data:          ## generate records, render PDF packets
	python backend/ml_training/synthetic_data.py --n 500
	python backend/ml_training/render_documents.py --limit 50

train:         ## train both models, log to MLflow
	cd backend && python ml_training/train_compliance_model.py
	cd backend && python ml_training/train_escalation_model.py

test:
	cd backend && python -m pytest tests/ -q

all: data train test
