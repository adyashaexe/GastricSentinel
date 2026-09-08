"""
build_genomic_features.py

Builds data/genomics/genomic_features.npz from the raw TCGA STAD gene
expression matrix.

Run from the repo root:
    python training/build_genomic_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GENOMIC_DIR = REPO_ROOT / "data" / "genomics"

# Your actual filename has the _PANCAN suffix — this was the wrong path
# in the old script.
RAW_MATRIX_PATH = GENOMIC_DIR / "TCGA.STAD.sampleMap_HiSeqV2_PANCAN"
OUTPUT_PATH = GENOMIC_DIR / "genomic_features.npz"

# Keep the most variable genes rather than the full ~20k gene matrix.
# Set to None to keep everything.
TOP_N_GENES = 2000


def load_expression_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the expression matrix at:\n  {path}\n"
            f"Expected 'TCGA.STAD.sampleMap_HiSeqV2_PANCAN' inside {GENOMIC_DIR}"
        )
    # UCSC Xena matrices are gene-by-sample: rows = genes (first column =
    # gene symbol), columns = sample barcodes.
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def select_top_variable_genes(df: pd.DataFrame, top_n):
    if top_n is None or top_n >= df.shape[0]:
        return df
    variances = df.var(axis=1, skipna=True)
    keep = variances.sort_values(ascending=False).head(top_n).index
    return df.loc[keep]


def main():
    print(f"Loading expression matrix from {RAW_MATRIX_PATH} ...")
    df = load_expression_matrix(RAW_MATRIX_PATH)
    print(f"Raw matrix (genes x samples): {df.shape}")

    df = select_top_variable_genes(df, TOP_N_GENES)
    print(f"After selecting top {TOP_N_GENES} variable genes: {df.shape}")

    # Transpose -> samples x genes, which is what a classifier wants.
    df = df.T

    sample_ids = df.index.astype(str).to_numpy()
    # TCGA barcodes look like TCGA-XX-XXXX-01A-...; clinical tables usually
    # key on the first 12 characters (the patient ID). Keep both.
    patient_ids = np.array([s[:12] for s in sample_ids])

    # Median-impute missing values per gene, then z-score per gene.
    df = df.apply(lambda col: col.fillna(col.median()), axis=0)
    features = df.to_numpy(dtype=np.float32)

    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0
    features = (features - means) / stds

    gene_names = df.columns.to_numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        features=features,
        sample_ids=sample_ids,
        patient_ids=patient_ids,
        gene_names=gene_names,
        feature_means=means,
        feature_stds=stds,
    )
    print(
        f"Saved {features.shape[0]} samples x {features.shape[1]} genes "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
