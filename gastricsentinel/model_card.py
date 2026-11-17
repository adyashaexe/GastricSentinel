"""Collects validation facts from the training artifacts so the UI and the PDF report can show
exactly what the deployed models were evaluated on (and what that evaluation can not claim)."""
from __future__ import annotations

from pathlib import Path

from .common import load_json


def _f(x):
    return None if x is None else float(x)


def load_model_card(artifact_dir) -> dict:
    d = Path(artifact_dir)
    card: dict = {"available": False}
    tm = d / "image" / "test_metrics.json"
    if tm.exists():
        m = load_json(tm)
        ctx = m.get("context", {})
        card["available"] = True
        card["image"] = {
            "n_test": int(m["n"]), "accuracy": _f(m["accuracy"]), "balanced_accuracy": _f(m["balanced_accuracy"]),
            "macro_precision": _f(m["macro"]["precision"]), "macro_recall": _f(m["macro"]["recall"]),
            "macro_f1": _f(m["macro"]["f1"]), "ece": _f(m.get("ece")),
            "ci95_accuracy": m.get("ci95", {}).get("accuracy"), "ci95_macro_f1": m.get("ci95", {}).get("macro_f1"),
            "temperature": _f(ctx.get("temperature")), "ece_before": _f(ctx.get("ece_before")),
            "ece_after": _f(ctx.get("ece_after")), "pretrained": ctx.get("pretrained"),
            "per_class": m["per_class"], "classes": m["class_names"],
            "confusion_matrix": m.get("confusion_matrix"), "reliability": m.get("reliability"), "brier": _f(m.get("brier")),
            "roc_auc": _f(m.get("roc_auc_macro_ovr")), "arch": ctx.get("arch") or "resnet50", "tta": ctx.get("tta"),
            "epochs_run": ctx.get("epochs_run"), "aug": ctx.get("aug"),
        }
        hp = d / "image" / "history.csv"
        if hp.exists():
            import pandas as pd
            card["history"] = pd.read_csv(hp).round(4).to_dict("records")
        op = d / "image" / "ood_stats.npz"
        if op.exists():
            from .ood import load_stats
            o = load_stats(op)
            card["ood"] = {"method": "mahalanobis", "threshold": o["threshold"], "id_median": o["id_p50"], "id_p95": o["id_p95"]}
    si = d / "image" / "split_info.json"
    if si.exists():
        s = load_json(si)
        card["split"] = {"mode": s.get("mode"), "leakage_risk": bool(s.get("leakage_risk")),
                         "counts": s.get("counts"), "n_groups": s.get("n_groups")}
    la = d / "image" / "leakage_audit.json"
    if la.exists():
        a = load_json(la)
        nd = (a.get("near_duplicates") or {}).get("test_vs_train")
        card["leakage_audit"] = {"group_overlap": a.get("group_overlap"), "test_vs_train": nd}
    tab = d / "tabular" / "genomic_clinical_metrics.json"
    if tab.exists():
        m = load_json(tab)
        ti = load_json(d / "tabular" / "tabular_info.json") if (d / "tabular" / "tabular_info.json").exists() else {}
        card["tabular"] = {"protocol": m.get("context", {}).get("protocol"), "n_patients": m["n"],
                           "accuracy": _f(m["accuracy"]), "macro_f1": _f(m["macro"]["f1"]),
                           "classes": m["class_names"], "excluded_classes": ti.get("excluded_classes")}
    from .registry import list_versions
    card["versions"] = list_versions(d)
    cdir = d / "comparisons"
    card["comparisons"] = [load_json(f) for f in sorted(cdir.glob("*.json"))] if cdir.exists() else []
    fp = d / "fusion" / "pairing_summary.json"
    if fp.exists():
        f = load_json(fp)
        card["fusion"] = {"pairing_mode": f["pairing_mode"], "disclosure": f["disclosure"],
                          "patient_level_correspondence": f["patient_level_correspondence"]}
    return card


def confusion_png(artifact_dir):
    p = Path(artifact_dir) / "image" / "test_confusion_matrix.png"
    return p if p.exists() else None


def reliability_png(artifact_dir):
    p = Path(artifact_dir) / "image" / "test_reliability.png"
    return p if p.exists() else None
