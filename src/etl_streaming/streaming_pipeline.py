import os
import sys
from pyspark.sql.functions import to_json, struct, col
from src.common.config import settings
from src.common.logger import get_logger
from src.etl_streaming.spark_session import create_spark_session
from src.etl_streaming.sql_transforms import (
    register_biochemical_udfs,
    transform_streaming_batch
)

logger = get_logger("StreamingPipeline")

def main():
    logger.info("Initializing Real-Time Protein Aggregation Streaming Pipeline...")
    
    # 1. Build optimized SparkSession
    spark = create_spark_session("ProteinAggregationStreamingETL")
    register_biochemical_udfs(spark)
    
    # 2. Ingest stream from Kafka topic: raw-aggregation-events
    logger.info(f"Subscribing to Kafka bootstrap '{settings.kafka_bootstrap_servers}', topic '{settings.kafka_input_topic}'")
    kafka_source_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_input_topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # 3. Apply schema parsing, chemical UDF featurization, and DQ barriers
    curated_df = transform_streaming_batch(spark, kafka_source_df)

    checkpoint_s3_path = f"s3a://spark-checkpoints/aggregation_etl/"
    output_lakehouse_path = settings.minio_parquet_path

    logger.info(f"Target Lakehouse sink: {output_lakehouse_path}")
    logger.info(f"Checkpoint location: {checkpoint_s3_path}")

    # 4. Sink A: Micro-batch append to MinIO Parquet Lakehouse
    lakehouse_query = (
        curated_df.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", output_lakehouse_path)
        .option("checkpointLocation", os.path.join(checkpoint_s3_path, "lakehouse"))
        .partitionBy("protein_id")
        .trigger(processingTime="5 seconds")
        .start()
    )

    # 5. Sink B: Stream transformed inference requests back to Kafka for real-time model scoring
    inference_payload_df = curated_df.select(
        col("assay_id").alias("key"),
        to_json(struct(
            col("assay_id"),
            col("event_timestamp"),
            col("protein_id"),
            col("drug_id"),
            col("drug_smiles"),
            col("drug_mw"),
            col("protein_seq_len"),
            col("protein_avg_hydropathy"),
            col("temp_c"),
            col("solution_ph"),
            col("drug_conc_um")
        )).alias("value")
    )

    kafka_inference_query = (
        inference_payload_df.writeStream
        .format("kafka")
        .outputMode("append")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("topic", settings.kafka_inference_topic)
        .option("checkpointLocation", os.path.join(checkpoint_s3_path, "kafka_inference"))
        .trigger(processingTime="5 seconds")
        .start()
    )

    logger.info("Streaming pipeline running. Listening for events...")
    
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Termination signal received. Shutting down streaming queries...")
        lakehouse_query.stop()
        kafka_inference_query.stop()
        spark.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()