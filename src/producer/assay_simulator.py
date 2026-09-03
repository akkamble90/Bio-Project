import os
import json
import time
import random
from typing import Dict, Any, List
from src.common.logger import get_logger

logger = get_logger("AssaySimulator")

FALLBACK_SEED_ASSAYS: List[Dict[str, Any]] = [
    {
        "protein": {
            "uniprot_id": "P05067",
            "name": "Amyloid-beta A4 protein (1-42)",
            "sequence": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        },
        "drug": {
            "chembl_id": "CHEMBL112",
            "name": "Resveratrol",
            "canonical_smiles": "Oc1ccc(cc1)/C=C/c2ccc(O)cc2",
            "molecular_weight": 228.24
        },
        "base_ic50_umol": 18.5,
        "base_inhibition": 82.0
    },
    {
        "protein": {
            "uniprot_id": "P37840",
            "name": "Alpha-synuclein (NAC core fragment)",
            "sequence": "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"
        },
        "drug": {
            "chembl_id": "CHEMBL148",
            "name": "Curcumin",
            "canonical_smiles": "COc1cc(ccc1O)/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2",
            "molecular_weight": 368.38
        },
        "base_ic50_umol": 12.0,
        "base_inhibition": 76.5
    },
    {
        "protein": {
            "uniprot_id": "P10636",
            "name": "Microtubule-associated protein tau (PHF6 hexapeptide)",
            "sequence": "VQIVYK"
        },
        "drug": {
            "chembl_id": "CHEMBL25",
            "name": "Aspirin",
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "molecular_weight": 180.16
        },
        "base_ic50_umol": 250.0,
        "base_inhibition": 14.0
    }
]

class AssaySimulator:
    """Generates continuous stream events with realistic experimental perturbations."""

    def __init__(self, seed_file_path: str = os.path.join("data", "raw_samples", "sample_assays.json")):
        self.seed_records = self._load_seeds(seed_file_path)
        self.counter = 1000

    def _load_seeds(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data:
                        logger.info(f"Loaded {len(data)} seeds from {path}")
                        return data
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}. Using fallback reference library.")
        return FALLBACK_SEED_ASSAYS

    def generate_event(self) -> Dict[str, Any]:
        """Produces a validated assay event adhering to assay_event_schema.json."""
        self.counter += 1
        template = random.choice(self.seed_records)

        # Perturb experimental assay conditions
        temp_c = round(random.gauss(37.0, 1.2), 2)
        ph_val = round(max(2.0, min(12.0, random.gauss(7.4, 0.3))), 2)
        conc_um = round(max(0.1, random.uniform(2.0, 50.0)), 2)

        # Continuous response model (Hill-equation dose response kinetics)
        base_ic50 = template.get("base_ic50_umol", 20.0)
        hill_coeff = 1.2
        hill_response = (conc_um ** hill_coeff) / ((base_ic50 ** hill_coeff) + (conc_um ** hill_coeff))
        inhibition_pct = round(min(100.0, max(0.0, hill_response * 100.0 + random.gauss(0, 2.5))), 2)

        outcome = "INHIBITOR" if inhibition_pct >= 50.0 else "INACTIVE"

        return {
            "assay_id": f"EXP-AGGR-2026-{self.counter:05d}",
            "timestamp": int(time.time()),
            "protein": {
                "uniprot_id": template["protein"]["uniprot_id"],
                "name": template["protein"]["name"],
                "sequence": template["protein"]["sequence"]
            },
            "drug": {
                "chembl_id": template["drug"]["chembl_id"],
                "name": template["drug"].get("name", "Unknown Compound"),
                "canonical_smiles": template["drug"]["canonical_smiles"],
                "molecular_weight": float(template["drug"].get("molecular_weight", 300.0))
            },
            "conditions": {
                "temperature_celsius": temp_c,
                "ph": ph_val,
                "drug_concentration_umol": conc_um
            },
            "label": {
                "inhibition_percentage": inhibition_pct,
                "outcome": outcome
            }
        }