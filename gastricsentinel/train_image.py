"""Train the image classifier with a leakage-aware split and full reporting.

    python -m gastricsentinel.train_image --data-dir data/images \
        --group-regex "^(P\\d+)_" --out-dir artifacts/image

Outputs (in --out-dir): image_model.pt, splits.csv, split_info.json,
leakage_audit.json, test_predictions.csv, test_metrics.json, test_per_class.csv,
test_confusion_matrix.{csv,png}, test_summary.md, history.csv
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import get_device, save_json, set_seed
from .config import ARTIFACTS, IMAGE_SIZE, SEED
from .data_images import ImageDataset, build_manifest, get_transform, make_splits
from .leakage_audit import audit_splits
from .metrics import ece, evaluate, save_report
from .models import build_image_model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    tot, ys, ps, probs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="train" if train else "eval", leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            tot += loss.item() * x.size(0)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
            probs.append(torch.softmax(out, 1).detach().cpu().numpy())
    ys, ps, probs = np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)
    return tot / len(loader.dataset), float((ys == ps).mean()), \
        float(f1_score(ys, ps, average="macro", zero_division=0)), ys, ps, probs


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).cpu() for x, _ in loader])


def fit_temperature(logits, y) -> float:
    """Single-parameter temperature scaling fitted on the validation split."""
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-2), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-2).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--manifest", default=None, help="optional CSV: path,label,group")
    ap.add_argument("--group-regex", default=None, help="regex whose 1st group is the patient/slide ID in the file stem")
    ap.add_argument("--allow-patch-level-split", action="store_true",
                    help="allow a random patch-level split (flagged leakage_risk=true)")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "image"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--trainable", choices=["fc", "layer4", "all"], default="layer4")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline/smoke tests)")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    df = build_manifest(a.data_dir, a.group_regex, a.manifest, a.max_per_class, a.seed)
    classes = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    df, info = make_splits(df, allow_patch_level=a.allow_patch_level_split, seed=a.seed)
    df["y"] = df["label"].map(c2i)
    df.drop(columns="y").to_csv(out / "splits.csv", index=False)
    info["classes"] = classes
    save_json(info, out / "split_info.json")
    print(f"classes: {classes}\nsplit: {info['mode']} {info['counts']}")

    print("auditing leakage ...")
    audit = audit_splits(df)
    save_json(audit, out / "leakage_audit.json")
    print(audit.get("near_duplicates"))

    def loader(split, train):
        d = df[df["split"] == split]
        ds = ImageDataset(d["path"], d["y"], get_transform(a.image_size, train))
        return DataLoader(ds, batch_size=a.batch_size, shuffle=train, num_workers=a.num_workers,
                          generator=torch.Generator().manual_seed(a.seed) if train else None)

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)

    model = build_image_model(len(classes), pretrained=not a.no_pretrained, trainable=a.trainable).to(device)
    weight = None
    if a.class_weights:
        cnt = df[df["split"] == "train"]["y"].value_counts().sort_index().to_numpy(float)
        weight = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trl, tra, trf, *_ = run_epoch(model, tr, criterion, opt, device, True)
        val, vaa, vaf, *_ = run_epoch(model, va, criterion, opt, device, False)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=trl, train_acc=tra, train_f1=trf,
                         val_loss=val, val_acc=vaa, val_f1=vaf))
        print(f"epoch {ep}/{a.epochs}  train acc {tra:.3f}  val acc {vaa:.3f}  val macro-F1 {vaf:.3f}")
        if vaf > best:
            best, best_state, bad = vaf, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    model.load_state_dict(best_state)
    _, _, _, y, p, probs = run_epoch(model, te, criterion, opt, device, False)
    yv = torch.tensor(df[df["split"] == "val"]["y"].to_numpy())
    val_logits = collect_logits(model, va, device)
    T = fit_temperature(val_logits, yv)
    # keep the temperature only if it actually improves VALIDATION calibration
    if ece(torch.softmax(val_logits / T, 1).numpy(), yv.numpy()) > ece(torch.softmax(val_logits, 1).numpy(), yv.numpy()):
        T = 1.0
    probs_raw = probs
    probs = torch.softmax(collect_logits(model, te, device) / T, 1).numpy()
    m = evaluate(y, p, classes, probs)
    ctx = {"temperature": round(T, 4), "ece_before": round(ece(probs_raw, y), 4),
           "ece_after": round(ece(probs, y), 4)}
    ctx.update({"split_mode": info["mode"], "leakage_risk": info["leakage_risk"],
           "best_val_macro_f1": round(best, 4), "train_seconds": round(time.time() - t0),
           "pretrained": not a.no_pretrained, "trainable": a.trainable, "seed": a.seed})
    save_report(m, out, "test", ctx)
    d = df[df["split"] == "test"]
    pred = pd.DataFrame({"path": d["path"].values, "true": [classes[i] for i in y],
                         "pred": [classes[i] for i in p]})
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, i]
    pred.to_csv(out / "test_predictions.csv", index=False)

    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "classes": classes, "image_size": int(a.image_size), "arch": "resnet50",
        "seed": int(a.seed), "split_mode": info["mode"], "leakage_risk": bool(info["leakage_risk"]),
        "best_val_macro_f1": float(best), "temperature": float(T),
    }, out / "image_model.pt")
    print(f"\nTEST  acc {m['accuracy']:.4f}  macro-F1 {m['macro']['f1']:.4f}  -> {out}")
    if info["leakage_risk"]:
        print("WARNING: patch-level split; do not report this as a leakage-free result.")


if __name__ == "__main__":
    main()
