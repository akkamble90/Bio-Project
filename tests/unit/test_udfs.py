import pytest
from src.etl_streaming.udfs.chemical_features import (
    extract_morgan_fingerprint,
    extract_physicochemical_descriptors
)
from src.etl_streaming.udfs.sequence_features import (
    calculate_sequence_hydropathy,
    calculate_protein_biophysics
)

def test_morgan_fingerprint_valid_smiles():
    """Validates Morgan fingerprint generation for known molecule (Resveratrol)."""
    resveratrol_smiles = "Oc1ccc(cc1)/C=C/c2ccc(O)cc2"
    fp = extract_morgan_fingerprint(resveratrol_smiles, radius=2, n_bits=1024)
    
    assert fp is not None
    assert len(fp) == 1024
    assert all(bit in (0, 1) for bit in fp)
    assert sum(fp) > 0  # Should have set bits

def test_morgan_fingerprint_invalid_smiles():
    """Ensures parser gracefully returns None for corrupted chemical strings."""
    invalid_smiles = "NOT_A_VALID_SMILES_STRING_123"
    fp = extract_morgan_fingerprint(invalid_smiles)
    assert fp is None

def test_chemical_descriptors():
    """Validates Lipinski parameter calculations."""
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    desc = extract_physicochemical_descriptors(aspirin_smiles)
    
    assert desc is not None
    assert "alogp" in desc
    assert "tpsa" in desc
    assert desc["h_donors"] == 1
    assert desc["h_acceptors"] == 4

def test_hydropathy_calculation():
    """
    Validates Kyte-Doolittle hydropathy index computation.
    Poly-isoleucine (hydrophobic) should be positive;
    Poly-arginine (hydrophilic) should be negative.
    """
    poly_ile = "IIIIII"
    poly_arg = "RRRRRR"
    
    score_ile = calculate_sequence_hydropathy(poly_ile)
    score_arg = calculate_sequence_hydropathy(poly_arg)
    
    assert score_ile is not None
    assert score_arg is not None
    assert score_ile > 0.0   # I = 4.5
    assert score_arg < 0.0   # R = -4.5
    assert round(score_ile, 2) == 4.50
    assert round(score_arg, 2) == -4.50

def test_protein_biophysics_calculation():
    """Validates mass, charge, and aliphatic index extraction."""
    seq = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"  # A-beta 42
    bio = calculate_protein_biophysics(seq)
    
    assert bio is not None
    assert bio["molecular_mass_da"] > 4000.0
    assert "net_charge" in bio
    assert "aliphatic_index" in bio