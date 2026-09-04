import os
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
import torch.nn as nn

from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger("ServingAPI")

app = FastAPI(
    title="Multimodal Drug-Target Affinity Screener",
    description="Real-time inference API evaluating SMILES topological features and protein sequence embeddings.",
    version="1.0.0",
)



# PyTorch Late-Fusion Model Definition / Loader
class MultimodalFusionNet(nn.Module):

  def __init__(
      self, chem_dim: int = 1024, prot_dim: int = 128, hidden_dim: int = 256
  ):
    super().__init__()
    self.chem_encoder = nn.Sequential(
        nn.Linear(chem_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
    )
    self.prot_encoder = nn.Sequential(
        nn.Linear(prot_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
    )
    self.fusion_head = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(
            hidden_dim, 2
        ),  # [0]: predicted_inhibition %, [1]: active_logit
    )

  def forward(self, chem_x: torch.Tensor, prot_x: torch.Tensor) -> torch.Tensor:
    h_chem = self.chem_encoder(chem_x)
    h_prot = self.prot_encoder(prot_x)
    fused = torch.cat([h_chem, h_prot], dim=-1)
    return self.fusion_head(fused)


MODEL_PATH = Path("models/fusion_net.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


@app.on_event("startup")
def load_model():
  global model
  model = MultimodalFusionNet().to(device)
  if MODEL_PATH.exists():
    try:
      state_dict = torch.load(MODEL_PATH, map_location=device)
      model.load_state_dict(state_dict)
      logger.info(f"Loaded trained PyTorch weights from {MODEL_PATH}")
    except Exception as e:
      logger.warning(
          f"Could not load state_dict directly ({e}). Using initialized weights"
          " for serving."
      )
  else:
    logger.warning(
        f"Model file not found at {MODEL_PATH}. Initialized baseline weights."
    )
  model.eval()


# Feature Extraction Helpers
def featurize_smiles(smiles: str, n_bits: int = 1024) -> np.ndarray:
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    raise ValueError(f"Invalid SMILES string: {smiles}")
  fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
  arr = np.zeros((n_bits,), dtype=np.float32)
  for bit in fp.GetOnBits():
    arr[bit] = 1.0
  return arr


def featurize_protein(sequence: str, target_dim: int = 128) -> np.ndarray:
  aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
  seq = sequence.upper().strip()
  if not seq:
    raise ValueError("Protein sequence cannot be empty.")
  counts = np.array([seq.count(aa) for aa in aa_alphabet], dtype=np.float32)
  norm = np.linalg.norm(counts)
  if norm > 0:
    counts = counts / norm
  feat = np.zeros(target_dim, dtype=np.float32)
  feat[: len(counts)] = counts
  return feat


# Request & Response Schemas
class ScreenRequest(BaseModel):
  assay_id: Optional[str] = Field(default="ASY-MANUAL-001")
  smiles: str = Field(
      ...,
      example="CC(=O)Oc1ccccc1C(=O)O",
      description="Canonical SMILES representation",
  )
  protein_sequence: str = Field(
      ..., example="MKTIIALSYIFCLVFA", description="Target amino acid sequence"
  )
  temperature_celsius: Optional[float] = 37.0
  ph: Optional[float] = 7.4
  drug_concentration_umol: Optional[float] = 10.0


class ScreenResponse(BaseModel):
  assay_id: str
  smiles: str
  molecular_weight: float
  predicted_inhibition: float
  status: str
  confidence_score: float


# API Endpoints
@app.get("/health")
def health_check() -> Dict[str, Any]:
  return {
      "status": "healthy",
      "device": str(device),
      "model_loaded": model is not None,
  }


@app.post("/predict", response_model=ScreenResponse)
def predict_affinity(payload: ScreenRequest):
  try:
    mol = Chem.MolFromSmiles(payload.smiles)
    if mol is None:
      raise HTTPException(
          status_code=400, detail="Invalid chemical SMILES provided."
      )
    mw = float(Descriptors.ExactMolWt(mol))

    chem_feat = featurize_smiles(payload.smiles)
    prot_feat = featurize_protein(payload.protein_sequence)

    t_chem = torch.tensor(
        chem_feat, dtype=torch.float32, device=device
    ).unsqueeze(0)
    t_prot = torch.tensor(
        prot_feat, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
      outputs = model(t_chem, t_prot)
      inhibition_pred = float(
          torch.clamp(outputs[0, 0], min=0.0, max=100.0).item()
      )
      prob_active = float(torch.sigmoid(outputs[0, 1]).item())

    is_active = inhibition_pred >= 60.0 or prob_active >= 0.5
    status = "ACTIVE" if is_active else "INACTIVE"

    return ScreenResponse(
        assay_id=payload.assay_id,
        smiles=payload.smiles,
        molecular_weight=round(mw, 2),
        predicted_inhibition=round(inhibition_pred, 2),
        status=status,
        confidence_score=round(prob_active, 4),
    )
  except ValueError as ve:
    raise HTTPException(status_code=400, detail=str(ve))
  except Exception as e:
    logger.error(f"Inference error: {e}")
    raise HTTPException(status_code=500, detail="Inference engine error.")