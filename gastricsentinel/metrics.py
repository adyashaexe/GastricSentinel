"""Evaluation metrics with every averaging choice stated explicitly.

Why this exists: a published table mixed accuracy 61.2% with recall 79.5% on a
roughly balanced 8-class set. Under standard definitions that cannot come from
one set of predictions: *weighted* recall always equals accuracy, and *macro*
recall equals it when classes are balanced. This module reports each quantity
once, with its averaging method, from a single confusion matrix, and asserts
the identity so an inconsistent table cannot be produced silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from .common import save_json


def _macro_f1(y, p, labels):
    return precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)[2]


def ece(probs, y_true, n_bins: int = 15) -> float:
    """Expected calibration error (top-label confidence vs accuracy, equal-width bins)."""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, ok = probs.max(1), probs.argmax(1) == y_true
    edges, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def evaluate(y_true, y_pred, class_names, y_prob=None, n_boot: int = 1000, seed: int = 0) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    avg = {}
    for name in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=name, zero_division=0)
        avg[name] = {"precision": float(pa), "recall": float(ra), "f1": float(fa)}

    # Identity check: weighted recall == accuracy (always).
    assert abs(avg["weighted"]["recall"] - acc) < 1e-9, "metric inconsistency"

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "averaging_note": (
            "macro = unweighted mean over classes; weighted = support-weighted mean. "
            "F1 is the mean of per-class F1 (NOT the harmonic mean of averaged "
            "precision and recall). Weighted recall equals accuracy by definition."
        ),
        "macro": avg["macro"],
        "weighted": avg["weighted"],
        "per_class": [
            {"class": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])} for i in labels
        ],
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    if y_prob is not None:
        out["ece"] = ece(y_prob, y_true)
        try:
            present = [i for i in labels if (y_true == i).any()]
            y_prob = np.asarray(y_prob)
            if len(present) == 2 and len(class_names) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            elif len(present) == len(class_names):
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                    labels=labels)
            else:
                auc = None
            out["roc_auc_macro_ovr"] = None if auc is None else float(auc)
        except ValueError:
            out["roc_auc_macro_ovr"] = None

    if n_boot and len(y_true) >= 10:
        rng = np.random.default_rng(seed)
        accs, f1s = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(y_true), len(y_true))
            accs.append(accuracy_score(y_true[i], y_pred[i]))
            f1s.append(_macro_f1(y_true[i], y_pred[i], labels))
        out["ci95"] = {
            "accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "note": "image-level bootstrap; optimistic if images share patients",
        }
    return out


def save_report(metrics: dict, out_dir, prefix: str, extra: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = dict(metrics)
    if extra:
        m["context"] = extra
    save_json(m, out_dir / f"{prefix}_metrics.json")
    pd.DataFrame(m["per_class"]).to_csv(out_dir / f"{prefix}_per_class.csv", index=False)
    names = m["class_names"]
    pd.DataFrame(m["confusion_matrix"], index=names, columns=names).to_csv(
        out_dir / f"{prefix}_confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(1 + 0.8 * len(names), 1 + 0.8 * len(names)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting must never break a training run
        print(f"(confusion-matrix plot skipped: {e})")

    ctx = m.get("context", {})
    lines = [
        f"# {prefix}", "",
        f"- n = {m['n']}",
        f"- accuracy = {m['accuracy']:.4f}  (balanced accuracy {m['balanced_accuracy']:.4f})",
        f"- macro P/R/F1 = {m['macro']['precision']:.4f} / {m['macro']['recall']:.4f} / {m['macro']['f1']:.4f}",
        f"- weighted P/R/F1 = {m['weighted']['precision']:.4f} / {m['weighted']['recall']:.4f} / {m['weighted']['f1']:.4f}",
    ]
    if m.get("roc_auc_macro_ovr") is not None:
        lines.append(f"- ROC-AUC (macro OvR) = {m['roc_auc_macro_ovr']:.4f}")
    if "ci95" in m:
        lines.append(f"- 95% CI accuracy = {m['ci95']['accuracy'][0]:.4f}-{m['ci95']['accuracy'][1]:.4f}; "
                     f"macro-F1 = {m['ci95']['macro_f1'][0]:.4f}-{m['ci95']['macro_f1'][1]:.4f}")
    if ctx:
        lines += ["", "Context:"] + [f"- {k}: {v}" for k, v in ctx.items()]
    lines += ["", m["averaging_note"], "", "Per-class table:", "",
              pd.DataFrame(m["per_class"]).round(4).to_markdown(index=False)
              if _has_tabulate() else pd.DataFrame(m["per_class"]).round(4).to_string(index=False)]
    (out_dir / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
