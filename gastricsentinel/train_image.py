"""Train the image classifier with a leakage-aware split and full reporting.

    python -m gastricsentinel.train_image --data-dir data/images \
        --group-regex "^(P\\d+)_" --out-dir artifacts/image

Backbones: resnet18, resnet50, densenet121, efficientnet_b0, convnext_tiny, mobilenet_v3_large
(--backbone). Training recipe: AdamW with a higher LR on the new head, linear warm-up then cosine decay, label
smoothing, optional mixup, gradient clipping, strong stain/scale augmentation, optional flip test-time augmentation.

Outputs (in --out-dir): image_model.pt, splits.csv, split_info.json, leakage_audit.json, test_predictions.csv,
test_metrics.json (incl. reliability bins, Brier), test_per_class.csv, test_confusion_matrix.{csv,png},
test_reliability.png, ood_stats.npz, test_summary.md, history.csv. The run is also copied to
artifacts/versions/<id>/ (see gastricsentinel.registry).
"""
from __future__ import annotations

import argparse
import copy
import math
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
from .models import BACKBONES, build_image_model
from .ood import fit_mahalanobis, mahalanobis, save_stats
from .registry import register


def run_epoch(model, loader, criterion, optimizer, device, train, mixup=0.0, clip=1.0):
    model.train(train)
    tot, ys, ps, probs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="train" if train else "eval", leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if train and mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(x.size(0), device=device)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[perm])
            else:
                out = model(x)
                loss = criterion(out, y)
            if train:
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], clip)
                optimizer.step()
            tot += loss.item() * x.size(0)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
            probs.append(torch.softmax(out, 1).detach().cpu().numpy())
    ys, ps, probs = np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)
    return tot / len(loader.dataset), float((ys == ps).mean()), \
        float(f1_score(ys, ps, average="macro", zero_division=0)), ys, ps, probs


@torch.no_grad()
def collect_logits(model, loader, device, tta=False):
    """Logits for a loader. tta=True averages the original and the horizontally / vertically flipped image."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        lg = model(x)
        if tta:
            lg = (lg + model(x.flip(3)) + model(x.flip(2))) / 3
        out.append(lg.cpu())
    return torch.cat(out)


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    return torch.cat([model.embed(x.to(device)).cpu() for x, _ in loader]).numpy()


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
    ap.add_argument("--backbone", choices=BACKBONES, default="resnet50")
    ap.add_argument("--trainable", choices=["fc", "last", "layer4", "all"], default="last",
                    help="fc = head only, last = head + last stage (layer4 is the old name), all = everything")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="LR multiplier for the new classifier head")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 = off, 0.2 is a sensible start)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--aug", choices=["light", "strong"], default="strong")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation for validation/test")
    ap.add_argument("--ood-samples", type=int, default=6000, help="training images used to fit the out-of-distribution model")
    ap.add_argument("--note", default="", help="free-text label stored with this version")
    ap.add_argument("--no-register", action="store_true", help="do not copy this run into artifacts/versions/")
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

    def loader(split, train, d=None):
        d = df[df["split"] == split] if d is None else d
        ds = ImageDataset(d["path"], d["y"], get_transform(a.image_size, train, a.aug))
        return DataLoader(ds, batch_size=a.batch_size, shuffle=train, num_workers=a.num_workers,
                          generator=torch.Generator().manual_seed(a.seed) if train else None)

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)

    model = build_image_model(len(classes), pretrained=not a.no_pretrained, trainable=a.trainable,
                              arch=a.backbone, dropout=a.dropout).to(device)
    weight = None
    if a.class_weights:
        cnt = df[df["split"] == "train"]["y"].value_counts().sort_index().to_numpy(float)
        weight = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=a.label_smoothing)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.head_lr_mult), weight_decay=a.weight_decay)
    warm = max(0, min(a.warmup_epochs, a.epochs - 1))

    def lr_factor(ep):  # epoch index from 0: linear warm-up, then cosine decay to ~0
        if ep < warm:
            return (ep + 1) / (warm + 1)
        return 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, a.epochs - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trl, tra, trf, *_ = run_epoch(model, tr, criterion, opt, device, True, a.mixup, a.grad_clip)
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
    yv = torch.tensor(df[df["split"] == "val"]["y"].to_numpy())
    val_logits = collect_logits(model, va, device, a.tta)
    T = fit_temperature(val_logits, yv)
    # keep the temperature only if it actually improves VALIDATION calibration
    if ece(torch.softmax(val_logits / T, 1).numpy(), yv.numpy()) > ece(torch.softmax(val_logits, 1).numpy(), yv.numpy()):
        T = 1.0
    test_logits = collect_logits(model, te, device, a.tta)
    y = df[df["split"] == "test"]["y"].to_numpy()
    probs_raw = torch.softmax(test_logits, 1).numpy()
    probs = torch.softmax(test_logits / T, 1).numpy()
    p = probs.argmax(1)
    m = evaluate(y, p, classes, probs)
    ctx = {"temperature": round(T, 4), "ece_before": round(ece(probs_raw, y), 4),
           "ece_after": round(ece(probs, y), 4)}
    ctx.update({"split_mode": info["mode"], "leakage_risk": info["leakage_risk"],
                "best_val_macro_f1": round(best, 4), "train_seconds": round(time.time() - t0),
                "pretrained": not a.no_pretrained, "trainable": a.trainable, "seed": a.seed, "arch": a.backbone,
                "epochs_run": len(hist), "aug": a.aug, "tta": a.tta, "mixup": a.mixup,
                "label_smoothing": a.label_smoothing, "lr": a.lr})
    save_report(m, out, "test", ctx)
    d = df[df["split"] == "test"]
    pred = pd.DataFrame({"path": d["path"].values, "true": [classes[i] for i in y],
                         "pred": [classes[i] for i in p]})
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, i]
    pred.to_csv(out / "test_predictions.csv", index=False)

    # ---- out-of-distribution model: Mahalanobis distance in feature space, fitted on training images
    try:
        trd = df[df["split"] == "train"]
        trd = trd.sample(n=min(a.ood_samples, len(trd)), random_state=a.seed)
        emb = collect_embeddings(model, loader("train", False, trd), device)
        stats = fit_mahalanobis(emb, trd["y"].to_numpy(), len(classes))
        val_scores = mahalanobis(stats, collect_embeddings(model, va, device))
        save_stats(out / "ood_stats.npz", stats, float(np.percentile(val_scores, 99)), val_scores)
        print(f"OOD model saved (validation 99th-percentile distance {np.percentile(val_scores, 99):.1f})")
    except Exception as e:  # never lose a trained model over an optional extra
        print(f"(OOD model skipped: {e})")

    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "classes": classes, "image_size": int(a.image_size), "arch": a.backbone, "format": 2,
        "seed": int(a.seed), "split_mode": info["mode"], "leakage_risk": bool(info["leakage_risk"]),
        "best_val_macro_f1": float(best), "temperature": float(T),
    }, out / "image_model.pt")
    if not a.no_register:
        vid = register(out, Path(out).parent, vars(a), a.note)
        print(f"registered as version {vid}  (python -m gastricsentinel.registry list)")
    print(f"\nTEST  acc {m['accuracy']:.4f}  macro-F1 {m['macro']['f1']:.4f}  -> {out}")
    if info["leakage_risk"]:
        print("WARNING: patch-level split; do not report this as a leakage-free result.")


if __name__ == "__main__":
    main()
