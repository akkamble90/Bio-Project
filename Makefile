.PHONY: help setup up down init-lakehouse seed-data train etl consumer producer test-unit test-integ run-api run-ui

PYTHON := python
PIP := pip

help:
	@echo "Available commands:"
	@echo "  make setup          - Install project dependencies"
	@echo "  make up             - Launch Docker containers (Kafka, MinIO, MLflow)"
	@echo "  make down           - Stop and remove Docker containers"
	@echo "  make init-lakehouse - Run initialization scripts for MinIO buckets"
	@echo "  make seed-data      - Fetch real-world assays from ChEMBL & UniProt"
	@echo "  make train          - Train PyTorch MultimodalFusionNet with MLflow"
	@echo "  make etl            - Start Spark streaming engine"
	@echo "  make producer       - Start Kafka assay event simulator"
	@echo "  make consumer       - Start background inference streaming consumer"
	@echo "  make run-api        - Start FastAPI serving layer"
	@echo "  make run-ui         - Launch Streamlit dashboard"
	@echo "  make test-unit      - Run unit test suite"
	@echo "  make test-integ     - Run integration tests"

setup:
	$(PIP) install --upgrade pip$(PIP) install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

init-lakehouse:
	bash scripts/init_minio.sh

seed-data:
	$(PYTHON) scripts/fetch_real_data.py

train:
	$(PYTHON) src/ml/train.py

etl:
	bash scripts/submit_spark_job.sh

producer:
	$(PYTHON) src/producer/kafka_producer.py 1.5

consumer:
	$(PYTHON) src/serving/kafka_consumer.py

run-api:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run src/ui/app.py --server.port 8501

test-unit:
	pytest test/unit -v

test-integ:
	pytest test/integration -v