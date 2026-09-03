-- Feature View: curated_assay_features
-- Transforms parsed streaming events into ML-ready vectorized features

CREATE OR REPLACE TEMPORARY VIEW curated_assay_features AS
SELECT
    -- Identifiers
    body.assay_id AS assay_id,
    body.timestamp AS event_timestamp,
    body.protein.uniprot_id AS protein_id,
    body.protein.name AS protein_name,
    body.drug.chembl_id AS drug_id,
    body.drug.canonical_smiles AS drug_smiles,

    -- Chemical Feature Extraction (Registered RDKit UDF: 1024-bit Morgan Fingerprint)
    extract_fp(body.drug.canonical_smiles) AS drug_morgan_fp,
    CAST(body.drug.molecular_weight AS DOUBLE) AS drug_mw,

    -- Biological Sequence Feature Extraction
    body.protein.sequence AS protein_sequence,
    LENGTH(body.protein.sequence) AS protein_seq_len,
    calc_hydropathy(body.protein.sequence) AS protein_avg_hydropathy,

    -- Experimental Assay Conditions (Continuous Features)
    ROUND(CAST(body.conditions.temperature_celsius AS DOUBLE), 2) AS temp_c,
    ROUND(CAST(body.conditions.ph AS DOUBLE), 2) AS solution_ph,
    CAST(body.conditions.drug_concentration_umol AS DOUBLE) AS drug_conc_um,

    -- Target Labels (Supervised Learning Ground Truth)
    CAST(body.label.inhibition_percentage AS DOUBLE) AS target_inhibition_pct,
    UPPER(TRIM(body.label.outcome)) AS target_class,
    CASE 
        WHEN body.label.inhibition_percentage >= 70.0 THEN 1 
        ELSE 0 
    END AS is_potent_inhibitor

FROM raw_assay_events
WHERE
    body.assay_id IS NOT NULL
    AND body.protein.sequence IS NOT NULL
    AND body.drug.canonical_smiles IS NOT NULL;