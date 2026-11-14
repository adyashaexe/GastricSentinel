"""Split-leakage audit.

Two checks on a splits CSV (path,label,group,split):
 1. group overlap: does any patient/slide ID appear in more than one split?
 2. near-duplicate images: perceptual (dHash) Hamming distance between each
    validation/test image and its closest training image.

Check 2 matters even when group IDs are unavailable: overlapping patches from
the same slide show up as near-duplicates across splits.

    python -m gastricsentinel.leakage_audit --splits artifacts/image/splits.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common import save_json


def dhash(path, size: int = 8) -> np.uint64:
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1).astype(np.int16)


def audit_splits(df: pd.DataFrame, threshold: int = 4, chunk: int = 256) -> dict:
    out = {"threshold_bits": threshold}

    if df["group"].notna().all():
        sets = {s: set(df[df["split"] == s]["group"]) for s in ("train", "val", "test")}
        out["group_overlap"] = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }
    else:
        out["group_overlap"] = None
        out["note"] = "no group IDs; group-overlap check skipped"

    hashes = np.array([dhash(p) for p in tqdm(df["path"], desc="dHash")], dtype=np.uint64)
    train_h = hashes[(df["split"] == "train").to_numpy()]
    out["near_duplicates"] = {}
    for s in ("val", "test"):
        ev = hashes[(df["split"] == s).to_numpy()]
        if len(ev) == 0 or len(train_h) == 0:
            continue
        mins = []
        for i in range(0, len(ev), chunk):
            d = _popcount(ev[i:i + chunk, None] ^ train_h[None, :])
            mins.append(d.min(axis=1))
        mins = np.concatenate(mins)
        out["near_duplicates"][f"{s}_vs_train"] = {
            "n_images": int(len(ev)),
            "exact_duplicates": int((mins == 0).sum()),
            "near_duplicates_le_threshold": int((mins <= threshold).sum()),
            "fraction_near_duplicate": float((mins <= threshold).mean()),
            "median_min_distance": float(np.median(mins)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_csv(a.splits)
    res = audit_splits(df, a.threshold)
    save_json(res, a.out or a.splits.replace(".csv", "_leakage_audit.json"))
    print(res)


if __name__ == "__main__":
    main()
