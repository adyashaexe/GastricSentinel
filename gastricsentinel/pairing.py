"""How images get paired with clinical/genomic records, made explicit and auditable.

Image datasets such as HMU-GC-HE-30K and TCGA-STAD genomics come from different
cohorts: there is NO shared patient ID. Two honest modes exist:

  matched : images carry a real TCGA barcode (e.g. tiles cut from TCGA-STAD
            diagnostic slides). `group` in splits.csv must be that barcode.
            Patient-level correspondence is real.
  random  : each image is assigned a TCGA patient drawn at random from the
            patients that belong to the SAME split, independent of the image
            label. Correspondence is synthetic. The tabular modality therefore
            carries no label information by construction, so this mode is a
            null/control experiment: it checks the fusion code and shows that
            fusion cannot beat image-only without real correspondence.

A TCGA patient is only ever used in one split (patient-disjoint), and the full
assignment is written to pairing_manifest.csv so a reviewer can audit it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED


def split_patients(patients: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=SEED):
    """Return Series patient_id -> split, stratified by subtype when feasible."""
    ids = patients.index.to_numpy()
    strat = patients["subtype"] if "subtype" in patients else None
    kw = {}
    if strat is not None and strat.notna().all() and strat.value_counts().min() >= 3:
        kw = {"stratify": strat.to_numpy()}
    tv, te = train_test_split(np.arange(len(ids)), test_size=test_frac, random_state=seed, **kw)
    kw2 = {"stratify": strat.to_numpy()[tv]} if kw else {}
    try:
        tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac), random_state=seed, **kw2)
    except ValueError:
        tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac), random_state=seed)
    s = pd.Series("train", index=ids)
    s.iloc[va] = "val"
    s.iloc[te] = "test"
    return s


def build_pairing(splits: pd.DataFrame, patients: pd.DataFrame, mode: str, seed=SEED,
                  val_frac=0.15, test_frac=0.15):
    """splits: DataFrame[path,label,group,split]. Returns (pairs, patient_split, summary)."""
    rng = np.random.default_rng(seed)
    pairs = splits.copy().reset_index(drop=True)

    if mode == "random":
        psplit = split_patients(patients, val_frac, test_frac, seed)
        pid = np.empty(len(pairs), dtype=object)
        for s in ("train", "val", "test"):
            pool = psplit.index[psplit == s].to_numpy()
            idx = np.flatnonzero((pairs["split"] == s).to_numpy())
            if len(pool) == 0:
                raise ValueError(f"no TCGA patients available for split '{s}'")
            pid[idx] = rng.choice(pool, size=len(idx), replace=True)
        pairs["patient_id"] = pid
        dropped = 0
    elif mode == "matched":
        pairs["patient_id"] = pairs["group"].astype(str).str[:12]
        known = pairs["patient_id"].isin(patients.index)
        dropped = int((~known).sum())
        pairs = pairs[known].reset_index(drop=True)
        if pairs.empty:
            raise ValueError("matched pairing: no image group matches a TCGA patient barcode")
        # each patient inherits the split of its images (images were split by patient)
        ps = pairs.groupby("patient_id")["split"].nunique()
        if (ps > 1).any():
            raise ValueError("matched pairing: a patient appears in more than one image split; "
                             "re-split images with group = patient barcode")
        psplit = pairs.groupby("patient_id")["split"].first()
    else:
        raise ValueError("mode must be 'random' or 'matched'")

    pairs["pairing_mode"] = mode
    sets = {s: set(pairs[pairs["split"] == s]["patient_id"]) for s in ("train", "val", "test")}
    disjoint = not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    summary = {
        "pairing_mode": mode,
        "patient_level_correspondence": mode == "matched",
        "images_dropped_no_tabular_match": dropped,
        "per_split": {
            s: {
                "images": int((pairs["split"] == s).sum()),
                "unique_patients": int(pairs[pairs["split"] == s]["patient_id"].nunique()),
            } for s in ("train", "val", "test")
        },
        "patient_disjoint_across_splits": bool(disjoint),
        "disclosure": (
            "Image and clinical/genomic records are linked by real TCGA barcode."
            if mode == "matched" else
            "Image and clinical/genomic records come from DIFFERENT cohorts and are paired at random "
            "(class-independent, patient-disjoint across splits). No patient-level correspondence "
            "exists; fusion results are a control, not evidence of multimodal benefit."
        ),
    }
    return pairs, psplit, summary
