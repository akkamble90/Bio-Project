import torch
import torch.nn as nn

class MolecularFingerprintEncoder(nn.Module):
    """Encodes high-dimensional sparse Morgan bit-vectors into dense latent representations."""
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AggregationMultimodalFusionNet(nn.Module):
    """
    Multimodal deep fusion network combining:
    1. Chemical Morgan Fingerprints (1024-d)
    2. Biophysical & Assay Continuous Features (e.g., MW, Seq Len, Hydropathy, Temp, pH, Conc)
    """
    def __init__(self, fp_dim: int = 1024, bio_dim: int = 6, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.chem_encoder = MolecularFingerprintEncoder(input_dim=fp_dim, latent_dim=latent_dim, dropout=dropout)
        
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        fused_dim = latent_dim + 32

        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Dual Heads
        # 1. Regression head: % aggregation inhibition (0.0 to 100.0)
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Scaled by 100 in forward pass
        )

        # 2. Classification head: Is Potent Inhibitor (logits)
        self.classification_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, chem_fp: torch.Tensor, bio_features: torch.Tensor):
        chem_latent = self.chem_encoder(chem_fp)
        bio_latent = self.bio_encoder(bio_features)

        fused = torch.cat([chem_latent, bio_latent], dim=-1)
        trunk_repr = self.shared_trunk(fused)

        pred_inhibition_pct = self.regression_head(trunk_repr) * 100.0
        class_logits = self.classification_head(trunk_repr)

        return pred_inhibition_pct.squeeze(-1), class_logits.squeeze(-1)