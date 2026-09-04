import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class AppSettings(BaseSettings):
    # Application & Environment
    app_name: str = "ProteinAggregationPlatform"
    env: str = Field(default="development", validation_alias="ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Groq LLM Settings
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", validation_alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.1, validation_alias="GROQ_TEMPERATURE")
    max_agent_cycles: int = Field(default=3, validation_alias="MAX_AGENT_CYCLES")

    # Kafka Broker & Topics
    kafka_bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_input_topic: str = Field(default="raw-aggregation-events", validation_alias="KAFKA_INPUT_TOPIC")
    kafka_inference_topic: str = Field(default="inference-requests", validation_alias="KAFKA_INFERENCE_TOPIC")
    kafka_output_topic: str = Field(default="predictions-out", validation_alias="KAFKA_OUTPUT_TOPIC")
    kafka_group_id: str = Field(default="protein-agent-consumer-group", validation_alias="KAFKA_GROUP_ID")

    # MinIO / S3 Lakehouse
    minio_endpoint: str = Field(default="http://localhost:9000", validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="feature-store", validation_alias="MINIO_BUCKET")
    minio_parquet_path: str = Field(
        default="s3a://feature-store/protein_drug_curated/",
        validation_alias="MINIO_PARQUET_PATH"
    )

    # MLflow & Model Serving
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI")
    inference_api_url: str = Field(default="http://localhost:8000/predict", validation_alias="INFERENCE_API_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = AppSettings()