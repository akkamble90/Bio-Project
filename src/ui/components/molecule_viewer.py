import base64
from io import BytesIO
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors

def smiles_to_image_base64(smiles: str, img_size=(380, 280)) -> str:
    """Renders an RDKit 2D chemical structure image into base64 format for Streamlit."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if not mol:
            return ""
        img = Draw.MolToImage(mol, size=img_size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""

def render_molecule_viewer():
    """Renders interactive 2D structure visualizer and Lipinski drug-likeness profiler."""
    st.subheader(" Molecular Structure & Drug-Likeness Profiler")
    st.caption("Inspect chemical scaffolds, 2D topology, and Lipinski's Rule of Five compliance.")

    # Preset compounds for rapid demonstration
    presets = {
        "Resveratrol (CHEMBL112)": "Oc1ccc(cc1)/C=C/c2ccc(O)cc2",
        "Curcumin (CHEMBL148)": "COc1cc(ccc1O)/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2",
        "Epigallocatechin gallate - EGCG (CHEMBL299)": "Oc1cc(O)c2c(c1)OC(c3cc(O)c(O)c(O)c3)C(OC(=O)c4cc(O)c(O)c(O)c4)C2",
        "Aspirin (CHEMBL25)": "CC(=O)Oc1ccccc1C(=O)O",
        "Custom SMILES": ""
    }

    col_select, col_in = st.columns([1, 2])
    with col_select:
        selected_preset = st.selectbox("Select Preset Molecule", list(presets.keys()))
    with col_in:
        default_smiles = presets[selected_preset]
        smiles_input = st.text_input(
            "Canonical SMILES Input",
            value=default_smiles if default_smiles else "Oc1ccc(cc1)/C=C/c2ccc(O)cc2"
        )

    if not smiles_input.strip():
        st.info("Provide a valid canonical SMILES string to view chemical topology.")
        return

    mol = Chem.MolFromSmiles(smiles_input.strip())
    if mol is None:
        st.error("Invalid SMILES string. RDKit could not parse the chemical structure.")
        return

    # Calculate molecular descriptors
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    num_heavy_atoms = int(mol.GetNumHeavyAtoms())

    # Lipinski Rule of Five evaluation (MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10)
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1

    col_img, col_metrics = st.columns([1, 1])

    with col_img:
        b64_img = smiles_to_image_base64(smiles_input.strip())
        if b64_img:
            st.markdown(
                f'<div style="text-align:center; padding:10px; background:#ffffff; border-radius:8px; border: 1px solid #ddd;">'
                f'<img src="data:image/png;base64,{b64_img}" alt="Molecular Scaffold" style="max-width:100%; height:auto;" />'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Could not generate 2D depiction.")

    with col_metrics:
        st.markdown("#### Physicochemical Properties")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Molecular Weight", f"{mw} g/mol")
        m_col2.metric("LogP (Lipophilicity)", f"{logp}")
        
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Polar Surface Area (TPSA)", f"{tpsa} Å²")
        m_col4.metric("Rotatable Bonds", rot_bonds)

        m_col5, m_col6 = st.columns(2)
        m_col5.metric("H-Bond Donors", hbd)
        m_col6.metric("H-Bond Acceptors", hba)

        st.markdown("---")
        if lipinski_violations == 0:
            st.success(" **Lipinski Rule of Five:** Compliant (0 Violations - Good Oral Bioavailability)")
        else:
            st.warning(f" **Lipinski Rule of Five:** {lipinski_violations} Violation(s)")