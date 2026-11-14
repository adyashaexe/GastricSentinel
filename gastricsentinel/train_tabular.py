"""Patient-level molecular-subtype models on TCGA-STAD (clinical-only and genomic+clinical).

Metrics come from stratified k-fold cross-validation with the preprocessor
(gene selection + scaling) re-fit inside every fold, pooled out-of-fold. This is
far more stable than a single 80/20 split when a class has ~7 patients.
Classes with fewer patients than folds are *reported and excluded*, not silently
merged into "Other".

    python -m gastricsentinel.train_tabular --clinical-tsv data/clinical/clinical.tsv \
        --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
        --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .common import save_json, set_seed
from .config import ARTIFACTS, SEED
from .data_tabular import TabularPreprocessor, build_patient_table
from .metrics import evaluate, save_report


def make_pipeline(include_genomic, top_genes, seed, n_trees):
    return Pipeline([
        ("prep", TabularPreprocessor(include_genomic=include_genomic, top_genes=top_genes)),
        ("clf", RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                                       random_state=seed, n_jobs=-1)),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical-tsv", required=True)
    ap.add_argument("--expression", required=True)
    ap.add_argument("--subtypes", required=True)
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "tabular"))
    ap.add_argument("--top-genes", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_patient_table(a.clinical_tsv, a.expression, a.subtypes)
    n_all = len(df)
    df = df.dropna(subset=["subtype"])
    counts = df["subtype"].value_counts()
    too_small = counts[counts < a.folds]
    excluded = {k: int(v) for k, v in too_small.items()}
    if excluded:
        print(f"EXCLUDED (fewer than {a.folds} patients, cannot be cross-validated): {excluded}")
        df = df[~df["subtype"].isin(too_small.index)]
    classes = sorted(df["subtype"].unique())
    print(f"patients with clinical+expression: {n_all}; with usable subtype: {len(df)}; classes: {classes}")
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["subtype"].map(c2i).to_numpy()
    X = df.drop(columns=["subtype"])

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    for name, genomic in (("clinical_only", False), ("genomic_clinical", True)):
        pipe = make_pipeline(genomic, a.top_genes, a.seed, a.trees)
        proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")
        m = evaluate(y, proba.argmax(1), classes, proba)
        save_report(m, out, name, {"protocol": f"{a.folds}-fold stratified CV, pooled out-of-fold",
                                   "n_patients": len(df), "excluded_classes": excluded or "none",
                                   "top_genes": a.top_genes if genomic else 0})
        print(f"{name}: acc {m['accuracy']:.3f}  macro-F1 {m['macro']['f1']:.3f}")
        final = make_pipeline(genomic, a.top_genes, a.seed, a.trees).fit(X, y)
        joblib.dump({"pipeline": final, "classes": classes}, out / f"{name}.joblib")

    save_json({"classes": classes, "n_patients": len(df), "excluded_classes": excluded,
               "note": "Reported metrics are cross-validated; the saved .joblib models are "
                       "re-fit on all patients for deployment."}, out / "tabular_info.json")


if __name__ == "__main__":
    main()
