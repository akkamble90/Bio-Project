from pathlib import Path
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, LongType
)
from src.etl_streaming.udfs.chemical_features import morgan_fp_udf, chemical_descriptors_udf
from src.etl_streaming.udfs.sequence_features import hydropathy_udf, protein_biophysics_udf
from src.common.logger import get_logger

logger = get_logger("SQLTransforms")

# Resolve repository root (/workspace inside Docker, /workspaces/Bio-Project in Codespaces host)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_raw_assay_schema() -> StructType:
    """Defines the Spark schema matching assay_event_schema.json."""
    return StructType([
        StructField("assay_id", StringType(), True),
        StructField("timestamp", LongType(), True),
        StructField("protein", StructType([
            StructField("uniprot_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("sequence", StringType(), True)
        ]), True),
        StructField("drug", StructType([
            StructField("chembl_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("canonical_smiles", StringType(), True),
            StructField("molecular_weight", DoubleType(), True)
        ]), True),
        StructField("conditions", StructType([
            StructField("temperature_celsius", DoubleType(), True),
            StructField("ph", DoubleType(), True),
            StructField("drug_concentration_umol", DoubleType(), True)
        ]), True),
        StructField("label", StructType([
            StructField("inhibition_percentage", DoubleType(), True),
            StructField("outcome", StringType(), True)
        ]), True)
    ])

def register_biochemical_udfs(spark: SparkSession) -> None:
    """Registers RDKit and biophysical UDFs in Spark SQL Catalog."""
    logger.info("Registering biochemical UDFs in Spark Catalog...")
    spark.udf.register("extract_fp", morgan_fp_udf)
    spark.udf.register("calc_hydropathy", hydropathy_udf)
    spark.udf.register("calc_descriptors", chemical_descriptors_udf)
    spark.udf.register("calc_biophysics", protein_biophysics_udf)

def load_sql_file(file_path: Path | str) -> str:
    """Reads raw SQL script from workspace."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing SQL file at path: {p.resolve()}")
    return p.read_text(encoding="utf-8")

def transform_streaming_batch(spark: SparkSession, kafka_df: DataFrame) -> DataFrame:
    """
    Parses raw Kafka string values into structured columns, executes
    vectorized SQL feature extraction, and applies data-quality gates.
    """
    schema = get_raw_assay_schema()
    
    # 1. Deserialize raw binary Kafka payload
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("body")
    )
    
    parsed_df.createOrReplaceTempView("raw_assay_events")
    
    # 2. Run feature extraction SQL using absolute path
    sql_dir = PROJECT_ROOT / "sql"
    feature_sql_path = sql_dir / "create_features_views.sql"
    if not feature_sql_path.exists():
        feature_sql_path = sql_dir / "create_feature_views.sql"
        
    spark.sql(load_sql_file(feature_sql_path))
    
    # 3. Run data quality filtering SQL using absolute path
    dq_sql_path = sql_dir / "data_quality_checks.sql"
    spark.sql(load_sql_file(dq_sql_path))
    
    # 4. Return clean, validated dataset view
    return spark.table("valid_assay_stream")