"""
build_subtype_labels.py

Extracts the real TCGA molecular subtype label per patient from your
downloaded cBioPortal export (stad_tcga_pan_can_atlas_2018_clinical_data.tsv).

Confirmed real values: STAD_CIN, STAD_MSI, STAD_GS, STAD_EBV, STAD_POLE
(29 patients have no subtype recorded and are dropped here.)

Output: subtype_labels.json  {patient_barcode: "STAD_CIN" | "STAD_MSI" | "STAD_GS" | "STAD_EBV" | "STAD_POLE"}
"""

import json
import pandas as pd

CBIOPORTAL_TSV_PATH = "data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv"
OUTPUT_PATH = "data/clinical/subtype_labels.json"


def main():
    df = pd.read_csv(CBIOPORTAL_TSV_PATH, sep="\t")
    df = df[["Patient ID", "Subtype"]].dropna(subset=["Subtype"])
    df = df.drop_duplicates("Patient ID")

    labels = dict(zip(df["Patient ID"], df["Subtype"]))

    print(f"Kept {len(labels)} patients with a known subtype.")
    print("Class distribution:")
    print(df["Subtype"].value_counts())

    with open(OUTPUT_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
