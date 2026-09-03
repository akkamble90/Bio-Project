from typing import Optional
from pyspark.sql.types import FloatType, StructType, StructField
from pyspark.sql.functions import udf

# Kyte & Doolittle hydropathy scale (J. Mol. Biol. 1982)
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Monoisotopic residue weights (g/mol) minus water
RESIDUE_WEIGHTS = {
    'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03, 'C': 103.01,
    'Q': 128.06, 'E': 129.04, 'G': 57.02, 'H': 137.06, 'I': 113.08,
    'L': 113.08, 'K': 128.09, 'M': 131.04, 'F': 147.07, 'P': 97.05,
    'S': 87.03, 'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07
}

def calculate_sequence_hydropathy(sequence: Optional[str]) -> Optional[float]:
    """
    Computes the mean hydropathy (GRAVY score) for an amino acid sequence.
    Positive values denote hydrophobic proteins (aggregation-prone);
    negative values denote hydrophilic proteins.
    """
    if not sequence or not isinstance(sequence, str):
        return None
    
    clean_seq = sequence.strip().upper()
    valid_scores = [KYTE_DOOLITTLE[aa] for aa in clean_seq if aa in KYTE_DOOLITTLE]
    
    if not valid_scores:
        return None
        
    return round(float(sum(valid_scores) / len(valid_scores)), 4)

def calculate_protein_biophysics(sequence: Optional[str]) -> Optional[dict]:
    """
    Computes structural biophysical metrics:
    - Estimated Molecular Mass (Da)
    - Net charge at physiological pH (7.4)
    - Aliphatic Index (relative volume occupied by aliphatic side chains: A, V, I, L)
    - Aromaticity (frequency of Phe, Tyr, Trp)
    """
    if not sequence or not isinstance(sequence, str):
        return None
        
    clean_seq = sequence.strip().upper()
    n = len(clean_seq)
    if n == 0:
        return None

    # Molecular Mass (sum of residue masses + terminal H2O 18.015 Da)
    mass = sum(RESIDUE_WEIGHTS.get(aa, 110.0) for aa in clean_seq) + 18.015

    # Net Charge estimation at pH 7.4: Positive (K, R, ~0.1 H), Negative (D, E)
    pos_charge = clean_seq.count('K') + clean_seq.count('R') + (0.1 * clean_seq.count('H'))
    neg_charge = clean_seq.count('D') + clean_seq.count('E')
    net_charge = round(pos_charge - neg_charge, 2)

    # Aliphatic index: X(Ala) + 2.9 * X(Val) + 3.9 * (X(Ile) + X(Leu))
    x_a = clean_seq.count('A') / n
    x_v = clean_seq.count('V') / n
    x_il = (clean_seq.count('I') + clean_seq.count('L')) / n
    aliphatic_idx = round((x_a + (2.9 * x_v) + (3.9 * x_il)) * 100, 2)

    # Aromaticity: relative frequency of Phe (F), Trp (W), Tyr (Y)
    aromatic_count = clean_seq.count('F') + clean_seq.count('W') + clean_seq.count('Y')
    aromaticity = round(aromatic_count / n, 4)

    return {
        "molecular_mass_da": round(mass, 2),
        "net_charge": float(net_charge),
        "aliphatic_index": float(aliphatic_idx),
        "aromaticity": float(aromaticity)
    }

# PySpark UDF registrations
hydropathy_udf = udf(lambda seq: calculate_sequence_hydropathy(seq), FloatType())

protein_biophysics_schema = StructType([
    StructField("molecular_mass_da", FloatType(), True),
    StructField("net_charge", FloatType(), True),
    StructField("aliphatic_index", FloatType(), True),
    StructField("aromaticity", FloatType(), True)
])

protein_biophysics_udf = udf(lambda seq: calculate_protein_biophysics(seq), protein_biophysics_schema)