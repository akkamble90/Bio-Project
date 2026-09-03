-- Data Quality Validation View: valid_assay_stream
-- Enforces physiological and chemical boundary gates

CREATE OR REPLACE TEMPORARY VIEW valid_assay_stream AS
SELECT *
FROM curated_assay_features
WHERE
    -- 1. Physiological pH Gate (Enzyme/Protein denaturation sanity check)
    solution_ph BETWEEN 2.0 AND 12.0

    -- 2. Thermal Stability Range (Liquid assay temperatures in Celsius)
    AND temp_c BETWEEN 4.0 AND 75.0

    -- 3. Concentration Boundary (Non-negative & below saturation limits)
    AND drug_conc_um > 0.0 AND drug_conc_um <= 1000.0

    -- 4. Protein Sequence Integrity (Minimum oligomer peptide length)
    AND protein_seq_len >= 5

    -- 5. Molecular Weight Range for Drug-like Small Molecules (g/mol)
    AND drug_mw BETWEEN 50.0 AND 1500.0

    -- 6. Target Validity Gate (Inhibition percentage bounds)
    AND target_inhibition_pct BETWEEN 0.0 AND 100.0

    -- 7. Valid Fingerprint Vector Dimension Verification
    AND drug_morgan_fp IS NOT NULL
    AND SIZE(drug_morgan_fp) = 1024;