"""Out-of-distribution checks.

Two independent layers:
  1. image_quality(): cheap checks that need no training (is this plausibly an H&E patch? blur, blank, tiny).
  2. Mahalanobis distance in the model's feature space, fitted on training embeddings. A score above the
     99th percentile of VALIDATION scores means "further from anything the model saw than 99% of held-out
     images", i.e. the prediction should not be trusted.
Neither proves an image is out of scope; they are warnings, not guarantees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def image_quality(im) -> dict:
    """Return {flags:[...], metrics:{...}} for a PIL RGB image."""
    a = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    white = (mn > 0.85).mean()                      # blank glass
    dark = (mx < 0.15).mean()
    gray = a.mean(2)
    lap = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    sharp = float(lap.var())
    tissue = (sat > 0.12) & (mn < 0.88)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # H&E is pink/purple: red and blue both exceed green on stained pixels
    he = float(((r > g + 0.02) & (b > g + 0.0) & tissue).sum() / max(tissue.sum(), 1))
    w, h = im.size
    m = {"tissue_fraction": float(tissue.mean()), "blank_fraction": float(white), "dark_fraction": float(dark),
         "sharpness": sharp, "he_like": he, "mean_saturation": float(sat.mean()), "width": w, "height": h}
    flags = []
    if min(w, h) < 64:
        flags.append("The image is very small (under 64 px); the model expects patches of roughly 224 px or more.")
    if m["tissue_fraction"] < 0.15:
        flags.append("Little or no stained tissue was found in this image (mostly blank or very pale).")
    elif m["he_like"] < 0.5:
        flags.append("The colours don't look like H&E staining (pink and purple), so this may not be the kind of image the model was trained on.")
    if m["dark_fraction"] > 0.5:
        flags.append("More than half of the image is very dark.")
    if sharp < 1e-4 and m["tissue_fraction"] >= 0.15:
        flags.append("The image looks very blurry or flat; predictions on out-of-focus images are unreliable.")
    return {"flags": flags, "metrics": m}


def fit_mahalanobis(emb: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """Class means + shared (Ledoit-Wolf shrunk) precision matrix."""
    from sklearn.covariance import LedoitWolf
    emb = emb.astype(np.float64)
    means = np.stack([emb[y == c].mean(0) if (y == c).any() else emb.mean(0) for c in range(n_classes)])
    centred = emb - means[y]
    prec = LedoitWolf().fit(centred).precision_
    return {"means": means.astype(np.float32), "precision": prec.astype(np.float32)}


def mahalanobis(stats: dict, emb: np.ndarray) -> np.ndarray:
    """Distance to the nearest class mean (squared Mahalanobis), one per row."""
    emb = np.atleast_2d(emb).astype(np.float32)
    d = []
    for mu in stats["means"]:
        z = emb - mu
        d.append(np.einsum("ij,jk,ik->i", z, stats["precision"], z))
    return np.min(np.stack(d, 1), 1)


def save_stats(path, stats: dict, threshold: float, id_scores: np.ndarray):
    np.savez(path, means=stats["means"], precision=stats["precision"], threshold=np.float32(threshold),
             id_p50=np.float32(np.percentile(id_scores, 50)), id_p95=np.float32(np.percentile(id_scores, 95)))


def load_stats(path):
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p)
    return {"means": z["means"], "precision": z["precision"], "threshold": float(z["threshold"]),
            "id_p50": float(z["id_p50"]), "id_p95": float(z["id_p95"])}
