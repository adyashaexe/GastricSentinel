"""
build_clinical_features.py

Builds real per-patient clinical feature vectors from your downloaded
data/clinical/extracted/clinical.tsv (standard GDC clinical export).

Confirmed real column names from your file:
  cases.submitter_id, demographic.age_at_index,
  demographic.sex_at_birth, diagnoses.ajcc_pathologic_stage

Output: clinical_features.npz {patient_barcode (12-char): vector}
"""

import numpy as np
import pandas as pd

CLINICAL_TSV_PATH = "data/clinical/extracted/clinical.tsv"
OUTPUT_PATH = "data/clinical/clinical_features.npz"

STAGE_ORDER = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4,
}

RENAME_MAP = {
    "cases.submitter_id": "patient_barcode",
    "demographic.age_at_index": "age",
    "demographic.sex_at_birth": "gender",
    "diagnoses.ajcc_pathologic_stage": "stage",
}


def main():
    df = pd.read_csv(CLINICAL_TSV_PATH, sep="\t")

    missing = [c for c in RENAME_MAP if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found: {missing}. Re-check `clinical.tsv` headers.")
    df = df.rename(columns=RENAME_MAP)

    df = df[["patient_barcode", "age", "gender", "stage"]].drop_duplicates("patient_barcode")
    df["patient_barcode"] = df["patient_barcode"].str[:12]  # normalize to 12-char patient barcode

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())
    df["age_z"] = (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-6)

    df["gender_bin"] = df["gender"].astype(str).str.lower().map({"male": 0, "female": 1}).fillna(0.5)

    df["stage_num"] = df["stage"].map(STAGE_ORDER)
    df["stage_num"] = df["stage_num"].fillna(df["stage_num"].median())
    df["stage_z"] = (df["stage_num"] - df["stage_num"].mean()) / (df["stage_num"].std() + 1e-6)

    result = {}
    for _, row in df.iterrows():
        result[row["patient_barcode"]] = np.array(
            [row["age_z"], row["gender_bin"], row["stage_z"]], dtype=np.float32
        )

    np.savez(OUTPUT_PATH, **result)
    print(f"Saved clinical features for {len(result)} patients to {OUTPUT_PATH}")
    print(f"Stage distribution:\n{df['stage'].value_counts(dropna=False)}")


if __name__ == "__main__":
    main()
