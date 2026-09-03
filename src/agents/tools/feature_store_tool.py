import os
import duckdb
from typing import List, Dict, Any, Optional
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger("FeatureStoreTool")

def query_feature_store(where_clause: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Executes vectorized SQL over MinIO Parquet files using DuckDB's S3 filesystem provider.
    """
    logger.info(f"Querying MinIO Feature Store (Filter: {where_clause}, Limit: {limit})")
    
    con = duckdb.connect(database=":memory:")
    
    try:
        # 1. Install & configure DuckDB HTTPFS for MinIO / S3 compatibility
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        
        endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
        con.execute(f"SET s3_endpoint='{endpoint}';")
        con.execute(f"SET s3_access_key_id='{settings.minio_access_key}';")
        con.execute(f"SET s3_secret_access_key='{settings.minio_secret_key}';")
        con.execute("SET s3_use_ssl=false;")
        con.execute("SET s3_url_style='path';")

        s3_parquet_path = f"s3://{settings.minio_bucket}/protein_drug_curated/**/*.parquet"
        
        sql = f"SELECT assay_id, protein_id, drug_id, drug_mw, protein_seq_len, protein_avg_hydropathy, temp_c, solution_ph, drug_conc_um, target_inhibition_pct, is_potent_inhibitor FROM read_parquet('{s3_parquet_path}')"
        
        if where_clause and where_clause.strip() and where_clause.lower() != "null":
            sql += f" WHERE {where_clause}"
            
        sql += f" LIMIT {limit};"
        
        df = con.execute(sql).df()
        records = df.to_dict(orient="records")
        logger.info(f"Retrieved {len(records)} records from MinIO Parquet Lakehouse.")
        return records

    except Exception as exc:
        logger.warning(f"Feature Store Parquet query failed: {exc}. Reading local fallback sample data...")
        # Fallback to local raw sample file if MinIO has not accumulated stream Parquet yet
        local_sample = os.path.join("data", "raw_samples", "sample_assays.json")
        if os.path.exists(local_sample):
            import json
            with open(local_sample, "r", encoding="utf-8") as f:
                data = json.load(f)
                out = []
                for item in data[:limit]:
                    out.append({
                        "assay_id": item.get("assay_id"),
                        "protein_id": item.get("protein", {}).get("uniprot_id"),
                        "drug_id": item.get("drug", {}).get("chembl_id"),
                        "target_inhibition_pct": item.get("label", {}).get("inhibition_percentage"),
                        "is_potent_inhibitor": 1 if item.get("label", {}).get("inhibition_percentage", 0) >= 70.0 else 0
                    })
                return out
        return []
    finally:
        con.close()