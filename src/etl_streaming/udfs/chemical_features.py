from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StructType, StructField
from pyspark.sql.functions import udf

def extract_morgan_fingerprint(smiles: Optional[str], radius: int = 2, n_bits: int = 1024) -> Optional[List[int]]:
    """
    Parses a SMILES string and computes a binary Morgan Fingerprint bit-vector (ECFP4).
    Returns a List[int] of size n_bits, or None if the SMILES is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return [int(b) for b in bit_vect.ToBitString()]
    except Exception:
        return None

def extract_physicochemical_descriptors(smiles: Optional[str]) -> Optional[dict]:
    """
    Calculates key drug-likeness continuous descriptors (Lipinski parameters):
    - LogP (lipophilicity)
    - TPSA (topological polar surface area)
    - NumHDonors (hydrogen bond donors)
    - NumHAcceptors (hydrogen bond acceptors)
    - RotatableBonds
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
            
        return {
            "alogp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "h_donors": int(rdMolDescriptors.CalcNumHBD(mol)),
            "h_acceptors": int(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        }
    except Exception:
        return None

# PySpark UDF registrations
morgan_fp_udf = udf(lambda s: extract_morgan_fingerprint(s), ArrayType(IntegerType()))

descriptors_schema = StructType([
    StructField("alogp", FloatType(), True),
    StructField("tpsa", FloatType(), True),
    StructField("h_donors", IntegerType(), True),
    StructField("h_acceptors", IntegerType(), True),
    StructField("rotatable_bonds", IntegerType(), True)
])

chemical_descriptors_udf = udf(lambda s: extract_physicochemical_descriptors(s), descriptors_schema)