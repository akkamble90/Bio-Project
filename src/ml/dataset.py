import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.etl_streaming.udfs.chemical_features import extract_morgan_fingerprint
from src.etl_streaming.udfs.sequence_features import calculate_sequence_hydropathy
from src.common.logger import get_logger

logger = get_logger("DatasetLoader")

class AggregationDataset(Dataset):
    """PyTorch Dataset for chemical Morgan fingerprints and protein assay conditions."""
    def __init__(self, parquet_path: Optional[str] = None):
        self.records = []
        
        # 1. Attempt reading parquet lakehouse
        if parquet_path and os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(f"Loaded {len(df)} records from Parquet dataset.")
                self._load_from_dataframe(df)
                return
            except Exception as e:
                logger.warn(f"Failed to read parquet ({e}). Falling back to local sample file.")

        # 2. Local fallback if lakehouse is empty or initializing
        local_sample_file = os.path.join("data", "raw_samples", "sample_assays.json")
        if os.path.exists(local_sample_file):
            logger.info(f"Loading seed dataset from {local_sample_file}")
            with open(local_sample_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self._load_from_raw_json(raw_data)
        else:
            raise FileNotFoundError("No training data available. Run scripts/fetch_real_data.py first.")

    def _load_from_dataframe(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            fp = row.get("drug_morgan_fp")
            if fp is None or len(fp) != 1024:
                fp = extract_morgan_fingerprint(row.get("drug_smiles", ""))
            
            if fp is None:
                continue

            bio_feats = [
                float(row.get("drug_mw", 300.0)) / 1000.0,
                float(row.get("protein_seq_len", 42.0)) / 100.0,
                float(row.get("protein_avg_hydropathy", 0.0)),
                (float(row.get("temp_c", 37.0)) - 37.0) / 10.0,
                (float(row.get("solution_ph", 7.4)) - 7.0) / 2.0,
                float(row.get("drug_conc_um", 20.0)) / 100.0
            ]

            target_inhibition = float(row.get("target_inhibition_pct", 0.0))
            is_inhibitor = 1.0 if target_inhibition >= 50.0 else 0.0

            self.records.append((
                np.array(fp, dtype=np.float32),
                np.array(bio_feats, dtype=np.float32),
                np.float32(target_inhibition),
                np.float32(is_inhibitor)
            ))

    def _load_from_raw_json(self, data: list):
        for item in data:
            smiles = item.get("drug", {}).get("canonical_smiles", "")
            seq = item.get("protein", {}).get("sequence", "")
            
            fp = extract_morgan_fingerprint(smiles)
            if not fp:
                continue

            hydropathy = calculate_sequence_hydropathy(seq) or 0.0
            cond = item.get("conditions", {})
            mw = float(item.get("drug", {}).get("molecular_weight", 300.0))

            bio_feats = [
                mw / 1000.0,
                len(seq) / 100.0,
                hydropathy,
                (float(cond.get("temperature_celsius", 37.0)) - 37.0) / 10.0,
                (float(cond.get("ph", 7.4)) - 7.0) / 2.0,
                float(cond.get("drug_concentration_umol", 20.0)) / 100.0
            ]

            inhibition = float(item.get("label", {}).get("inhibition_percentage", 0.0))
            is_inhibitor = 1.0 if inhibition >= 50.0 else 0.0

            self.records.append((
                np.array(fp, dtype=np.float32),
                np.array(bio_feats, dtype=np.float32),
                np.float32(inhibition),
                np.float32(is_inhibitor)
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fp, bio, target_val, target_class = self.records[idx]
        return (
            torch.from_numpy(fp),
            torch.from_numpy(bio),
            torch.tensor(target_val, dtype=torch.float32),
            torch.tensor(target_class, dtype=torch.float32)
        )