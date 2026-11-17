"""Model versioning.

Every finished training run is copied into artifacts/versions/<id>/ where <id> is the first 10 hex characters of
the SHA-256 of image_model.pt (the same id the web app stamps on every analysis). The directory the app serves
(artifacts/image/) is simply the *active* version.

    python -m gastricsentinel.registry list
    python -m gastricsentinel.registry activate <id>
    python -m gastricsentinel.registry note <id> "label or comment"
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path

from .common import load_json, save_json
from .config import ARTIFACTS

FILES = ("image_model.pt", "test_metrics.json", "test_predictions.csv", "test_per_class.csv", "history.csv",
         "split_info.json", "leakage_audit.json", "ood_stats.npz", "test_confusion_matrix.png",
         "test_confusion_matrix.csv", "test_reliability.png", "test_summary.md")


def file_version(path) -> str:
    p = Path(path)
    if not p.exists():
        return "untrained"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def register(image_dir, art_dir=ARTIFACTS, args: dict | None = None, note: str = "") -> str:
    image_dir, art_dir = Path(image_dir), Path(art_dir)
    vid = file_version(image_dir / "image_model.pt")
    dest = art_dir / "versions" / vid
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (image_dir / name).exists():
            shutil.copy2(image_dir / name, dest / name)
    m = load_json(image_dir / "test_metrics.json") if (image_dir / "test_metrics.json").exists() else {}
    ctx = m.get("context", {})
    meta = {"id": vid, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": note,
            "arch": ctx.get("arch"), "accuracy": m.get("accuracy"), "macro_f1": (m.get("macro") or {}).get("f1"),
            "ece": m.get("ece"), "brier": m.get("brier"), "n_test": m.get("n"),
            "best_val_macro_f1": ctx.get("best_val_macro_f1"), "split_mode": ctx.get("split_mode"),
            "leakage_risk": ctx.get("leakage_risk"), "trainable": ctx.get("trainable"), "args": args or {}}
    if (dest / "meta.json").exists():  # re-registering the same weights keeps the user's note
        old = load_json(dest / "meta.json")
        meta["note"] = note or old.get("note", "")
    save_json(meta, dest / "meta.json")
    return vid


def list_versions(art_dir=ARTIFACTS) -> list[dict]:
    art_dir = Path(art_dir)
    active = file_version(art_dir / "image" / "image_model.pt")
    rows = []
    for d in sorted((art_dir / "versions").glob("*/meta.json")) if (art_dir / "versions").exists() else []:
        m = load_json(d)
        m["active"] = m["id"] == active
        rows.append(m)
    return sorted(rows, key=lambda r: r["created"], reverse=True)


def activate(vid: str, art_dir=ARTIFACTS):
    art_dir = Path(art_dir)
    src = art_dir / "versions" / vid
    if not (src / "image_model.pt").exists():
        raise SystemExit(f"no such version: {vid}")
    dst = art_dir / "image"
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (dst / name).exists() and not (src / name).exists():
            (dst / name).unlink()  # do not leave another version's evidence next to these weights
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"active image model is now {vid}. Restart the app to load it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "activate", "note"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    a = ap.parse_args()
    if a.cmd == "list":
        for r in list_versions(a.artifacts):
            f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"{'*' if r['active'] else ' '} {r['id']}  {r.get('arch') or '?':<18} acc {f(r['accuracy'])}  "
                  f"F1 {f(r['macro_f1'])}  ECE {f(r['ece'])}  {r['created']}  {r.get('note','')}")
    elif a.cmd == "activate":
        activate(a.id, a.artifacts)
    else:
        p = Path(a.artifacts) / "versions" / a.id / "meta.json"
        m = load_json(p)
        m["note"] = a.text
        save_json(m, p)


if __name__ == "__main__":
    main()
