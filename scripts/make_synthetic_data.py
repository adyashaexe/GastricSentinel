"""Generate SYNTHETIC data in the same file layouts the real pipeline expects.

For smoke-testing the code only. The images are noise patterns and the tables are
random draws: any metric computed on them is meaningless.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_images(out: Path, rng, n_patients=24, per_class=6, size=64):
    classes = ["ADI", "LYM", "STR", "TUM"]
    base = {"ADI": (230, 225, 210), "LYM": (90, 60, 140), "STR": (200, 140, 170), "TUM": (150, 70, 130)}
    for c in classes:
        (out / "images" / c).mkdir(parents=True, exist_ok=True)
    for p in range(n_patients):
        stain = rng.normal(0, 12, 3)  # slide-level colour shift
        for c in classes:
            for i in range(per_class):
                img = np.ones((size, size, 3)) * np.array(base[c]) + stain
                if c == "STR":
                    img[:, ::6] -= 40
                if c == "TUM":
                    yy, xx = np.mgrid[:size, :size]
                    img[((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 4) ** 2] -= 50
                img += rng.normal(0, 18, img.shape)
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    out / "images" / c / f"P{p:03d}_{i:02d}.png")


def make_tables(out: Path, rng, n_genes=200):
    sub_n = {"STAD_CIN": 22, "STAD_MSI": 16, "STAD_GS": 12, "STAD_RARE": 3}  # RARE < folds -> must be excluded
    ids, subs = [], []
    k = 0
    for s, n in sub_n.items():
        for _ in range(n):
            ids.append(f"TCGA-AA-{k:04d}")
            subs.append(s)
            k += 1
    n = len(ids)
    (out / "clinical").mkdir(parents=True, exist_ok=True)
    (out / "genomics").mkdir(parents=True, exist_ok=True)

    stage = rng.choice(["Stage I", "Stage IIA", "Stage IIIB", "Stage IV", "'--"], n)
    clin = pd.DataFrame({
        "cases.submitter_id": ids,
        "demographic.age_at_index": np.where(rng.random(n) < 0.05, "'--", rng.integers(35, 85, n).astype(str)),
        "demographic.gender": rng.choice(["male", "female"], n),
        "diagnoses.ajcc_pathologic_stage": stage,
    })
    clin.to_csv(out / "clinical" / "clinical.tsv", sep="\t", index=False)

    genes = [f"GENE{j}" for j in range(n_genes)]
    shift = {s: rng.normal(0, 1.2, 20) for s in sub_n}
    expr = rng.normal(8, 1, (n, n_genes))
    for r, s in enumerate(subs):
        expr[r, :20] += shift[s]
    cols = [f"{i}-01" for i in ids]
    mat = pd.DataFrame(expr.T, index=genes, columns=cols)
    for j in range(5):  # extra non-tumour samples that must be filtered out
        mat[f"TCGA-ZZ-{j:04d}-11"] = rng.normal(8, 1, n_genes)
    mat.index.name = "sample"
    mat.to_csv(out / "genomics" / "TCGA.STAD.sampleMap_HiSeqV2_PANCAN", sep="\t")

    pd.DataFrame({"Patient ID": ids, "Subtype": subs}).to_csv(
        out / "clinical" / "subtypes.tsv", sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)
    make_images(out, rng)
    make_tables(out, rng)
    print(f"synthetic data written to {out}")
