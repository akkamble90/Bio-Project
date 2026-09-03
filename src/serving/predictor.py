import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional
from src.ml.model import AggregationMultimodalFusionNet
from src.etl_streaming.udfs.chemical_features import extract_morgan_fingerprint
from src.etl_streaming.udfs.sequence_features import calculate_sequence_hydropathy
from src.common.logger import get_logger

logger = get_logger("ModelPredictor")

class AggregationPredictor:
    """Manages model checkpoint loading and continuous multi-modal inference."""

    def __init__(self, weights_path: str = os.path.join("models", "fusion_net.pt")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = "MultimodalFusionNet-v1.0.0"
        self.model = AggregationMultimodalFusionNet(fp_dim=1024, bio_dim=6, latent_dim=128)

        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded trained model weights from: {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights ({e}). Initialized in uncalibrated baseline mode.")
        else:
            logger.info(f"Weights file not found at {weights_path}. Running with initialized architecture.")

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        smiles: str,
        sequence: str,
        molecular_weight: float = 300.0,
        temp_c: float = 37.0,
        ph: float = 7.4,
        drug_conc_um: float = 20.0
    ) -> Dict[str, Any]:
        """Runs synchronized chemical + sequence featurization and forward inference."""
        start_time = time.perf_counter()

        # 1. Featurize SMILES into 1024-bit Morgan Fingerprint
        fp = extract_morgan_fingerprint(smiles)
        if fp is None:
            # Fallback zero-vector if smiles parse fails
            fp = [0] * 1024

        # 2. Extract biophysical scale metrics
        hydropathy = calculate_sequence_hydropathy(sequence) or 0.0
        seq_len = len(sequence)

        # 3. Vectorize normalized bio/assay conditions
        bio_vector = [
            float(molecular_weight) / 1000.0,
            float(seq_len) / 100.0,
            float(hydropathy),
            (float(temp_c) - 37.0) / 10.0,
            (float(ph) - 7.0) / 2.0,
            float(drug_conc_um) / 100.0
        ]

        # 4. PyTorch zero-grad tensor inference
        fp_tensor = torch.tensor([fp], dtype=torch.float32).to(self.device)
        bio_tensor = torch.tensor([bio_vector], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_inhibition, class_logits = self.model(fp_tensor, bio_tensor)
            inhibition_val = float(pred_inhibition.squeeze().cpu().item())
            logit_val = float(class_logits.squeeze().cpu().item())
            confidence = float(1.0 / (1.0 + np.exp(-logit_val)))

        inhibition_pct = round(max(0.0, min(100.0, inhibition_val)), 2)

        if inhibition_pct >= 50.0:
            predicted_class = "INHIBITOR"
        elif inhibition_pct <= 20.0:
            predicted_class = "INACTIVE"
        else:
            predicted_class = "AGGREGATOR"

        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "predicted_inhibition_pct": inhibition_pct,
            "predicted_class": predicted_class,
            "confidence_score": round(confidence, 4),
            "model_version": self.model_version,
            "latency_ms": latency
        }