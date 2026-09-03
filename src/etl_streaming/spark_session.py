import os
from pyspark.sql import SparkSession
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger("SparkSessionBuilder")

def create_spark_session(app_name: str = "ProteinAggregationETL") -> SparkSession:
    """
    Initializes a production-grade SparkSession configured for:
    1. Kafka Structured Streaming consumption.
    2. MinIO / S3A object store read and write (Parquet Lakehouse).
    3. Optimized vectorized shuffle and serialization.
    """
    logger.info(f"Building SparkSession: '{app_name}'")

    # Format MinIO endpoint for Hadoop S3A (strips protocol)
    s3_endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")

    # Required Maven packages for Kafka and Hadoop-AWS integration
    packages = [
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3",
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262"
    ]

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", ",".join(packages))
        
        # S3A / MinIO FileSystem Configuration
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{s3_endpoint}")
        .config("spark.hadoop.fs.s3a.access.key", settings.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", settings.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
        
        # Streaming & Execution Performance Optimization
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )

    # Honor SPARK_MASTER_URL if defined (e.g. running in Docker or cluster mode)
    master_url = os.getenv("SPARK_MASTER_URL")
    if master_url:
        logger.info(f"Connecting to standalone cluster master: {master_url}")
        builder = builder.master(master_url)
    else:
        logger.info("Using local multi-threaded master mode: local[*]")
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("SparkSession initialized successfully.")
    return spark