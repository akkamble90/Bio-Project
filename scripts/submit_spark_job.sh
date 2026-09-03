#!/usr/bin/env bash
set -e

SPARK_MASTER_CONTAINER="bio_spark_master"
KAFKA_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"

echo "[INFO] Submitting Streaming ETL job to Spark Master container ($SPARK_MASTER_CONTAINER)..."

docker exec -it "$SPARK_MASTER_CONTAINER" /opt/apache/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --name "ProteinAggregationStreamingETL" \
  --packages "$KAFKA_PACKAGES" \
  --properties-file /workspace/configs/spark-defaults.conf \
  --driver-memory 2G \
  --executor-memory 2G \
  /workspace/src/etl_streaming/streaming_pipeline.py

echo "[INFO] Spark streaming job terminated."