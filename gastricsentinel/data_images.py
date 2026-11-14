"""Image manifest, leakage-aware splitting, transforms and dataset.

The key rule: if images from the same patient/slide can end up in different
splits, test accuracy is inflated. Splitting is therefore done on a `group`
column (patient or slide ID). If no group information exists the script
refuses to split unless `allow_patch_level=True` is passed explicitly, and the
resulting split is flagged `leakage_risk: true` in every report.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD, SEED


def build_manifest(image_root, group_regex=None, manifest_csv=None,
                   max_per_class=None, seed=SEED) -> pd.DataFrame:
    """Return DataFrame[path, label, group].

    Layout 1 (default): <image_root>/<CLASS_NAME>/**/*.png
    Layout 2: a CSV with columns path,label[,group] (paths relative to image_root
              or absolute). Use this when patient/slide IDs are in a metadata file.
    `group_regex` extracts the group from the file stem (first capture group).
    """
    root = Path(image_root)
    if manifest_csv:
        df = pd.read_csv(manifest_csv)
        if not {"path", "label"} <= set(df.columns):
            raise ValueError("manifest CSV needs columns: path,label[,group]")
        if "group" not in df.columns:
            df["group"] = None
        df["path"] = [str(p if Path(p).is_absolute() else root / p) for p in df["path"]]
    else:
        if not root.is_dir():
            raise FileNotFoundError(f"image root not found: {root}")
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not classes:
            raise FileNotFoundError(f"no class sub-folders in {root}")
        rx = re.compile(group_regex) if group_regex else None
        rows, unmatched = [], 0
        for c in classes:
            for p in sorted((root / c).rglob("*")):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                g = None
                if rx is not None:
                    m = rx.search(p.stem)
                    if m:
                        g = m.group(1) if m.groups() else m.group(0)
                    else:
                        unmatched += 1
                rows.append((str(p), c, g))
        if unmatched:
            raise ValueError(
                f"group_regex matched no group for {unmatched} files; refusing to "
                "continue because unmatched files would silently weaken the "
                "leakage guard. Fix the regex or use a manifest CSV."
            )
        df = pd.DataFrame(rows, columns=["path", "label", "group"])
    if df.empty:
        raise RuntimeError("no images found")
    if max_per_class:
        df = (df.sample(frac=1, random_state=seed)
                .groupby("label").head(max_per_class)
                .sort_values("path").reset_index(drop=True))
    return df.reset_index(drop=True)


def _group_holdout(df, labels, groups, frac, seed):
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        rest, hold = next(sgkf.split(df, labels, groups))
    except ValueError as e:
        raise ValueError(
            f"cannot make a stratified group split ({e}). Too few groups per class."
        ) from e
    return rest, hold


def make_splits(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=SEED,
                allow_patch_level=False):
    """Add a `split` column (train/val/test). Returns (df, info)."""
    df = df.copy().reset_index(drop=True)
    has_groups = df["group"].notna().all()
    info = {"seed": seed, "val_frac": val_frac, "test_frac": test_frac}

    if has_groups:
        rest, test_idx = _group_holdout(df, df["label"], df["group"], test_frac, seed)
        sub = df.iloc[rest]
        val_rel = val_frac / (1 - test_frac)
        r2, v2 = _group_holdout(sub, sub["label"], sub["group"], val_rel, seed)
        df["split"] = "train"
        df.loc[df.index[test_idx], "split"] = "test"
        df.loc[sub.index[v2], "split"] = "val"
        info.update(mode="group_level", group_level_split=True, leakage_risk=False,
                    n_groups=int(df["group"].nunique()))
    else:
        if not allow_patch_level:
            raise ValueError(
                "No patient/slide group IDs available, so a leakage-free split is "
                "impossible. Provide --group-regex or a manifest CSV with a `group` "
                "column. To proceed anyway (NOT recommended for reported results) "
                "pass --allow-patch-level-split; the report will then be flagged "
                "leakage_risk=true."
            )
        idx = np.arange(len(df))
        tv, te = train_test_split(idx, test_size=test_frac, stratify=df["label"],
                                  random_state=seed)
        tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac),
                                  stratify=df["label"].iloc[tv], random_state=seed)
        df["split"] = "train"
        df.loc[va, "split"] = "val"
        df.loc[te, "split"] = "test"
        info.update(mode="patch_level_random", group_level_split=False, leakage_risk=True)

    info["counts"] = {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}
    info["class_counts"] = {
        s: {k: int(v) for k, v in df[df["split"] == s]["label"].value_counts().sort_index().items()}
        for s in ("train", "val", "test")
    }
    if has_groups:
        sets = {s: set(df[df["split"] == s]["group"]) for s in ("train", "val", "test")}
        assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
            and not (sets["val"] & sets["test"]), "group overlap between splits"
    return df, info


def get_transform(image_size: int, train: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(self.labels[i])
