"""Train and evaluate the multimodal fusion head, with modality ablations.

Image features come from the already-trained image model's ResNet-50 backbone
(computed once, no augmentation). Clinical and genomic encoders + fusion head are
trained on top with modality dropout. Evaluated on the held-out test split under
four input conditions so the contribution of each modality is measurable:

    image_only | image+clinical | image+genomic | image+clinical+genomic

and compared with the image model's own head on exactly the same test rows.

    python -m gastricsentinel.train_fusion --image-dir artifacts/image \
        --clinical-tsv ... --expression ... [--subtypes ...] [--pairing random|matched]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .common import get_device, save_json, set_seed
from .config import ARTIFACTS, SEED
from .data_images import ImageDataset, get_transform
from .data_tabular import TabularPreprocessor, build_patient_table
from .metrics import evaluate, save_report
from .models import Backbone, FusionNet, load_image_model
from .pairing import build_pairing


@torch.no_grad()
def extract_features(backbone, paths, image_size, device, batch_size=64):
    ds = ImageDataset(paths, [0] * len(paths), get_transform(image_size, False))
    feats = []
    for x, _ in tqdm(DataLoader(ds, batch_size=batch_size), desc="image features"):
        feats.append(backbone(x.to(device)).cpu())
    return torch.cat(feats)


def predict(model, img, clin, gen, mask, device, bs=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(img), bs):
            outs.append(torch.softmax(model(img[i:i+bs].to(device), clin[i:i+bs].to(device),
                                            gen[i:i+bs].to(device), mask[i:i+bs].to(device)), 1).cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", default=str(ARTIFACTS / "image"))
    ap.add_argument("--clinical-tsv", required=True)
    ap.add_argument("--expression", required=True)
    ap.add_argument("--subtypes", default=None, help="optional; only used to stratify the patient split")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "fusion"))
    ap.add_argument("--pairing", choices=["random", "matched"], default="random")
    ap.add_argument("--top-genes", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modality-dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    set_seed(a.seed)
    device = get_device()
    img_dir, out = Path(a.image_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, ck = load_image_model(img_dir / "image_model.pt", device)
    classes = ck["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    splits = pd.read_csv(img_dir / "splits.csv")
    patients = build_patient_table(a.clinical_tsv, a.expression, a.subtypes)

    pairs, psplit, summary = build_pairing(splits, patients, a.pairing, a.seed)
    pairs.to_csv(out / "pairing_manifest.csv", index=False)
    save_json(summary, out / "pairing_summary.json")
    print(summary["disclosure"])
    print(summary["per_split"])
    if not summary["patient_disjoint_across_splits"]:
        raise RuntimeError("train/test share TCGA patients")

    # Preprocessor fit on TRAIN patients only.
    train_pids = sorted(set(pairs[pairs["split"] == "train"]["patient_id"]))
    prep = TabularPreprocessor(include_genomic=True, top_genes=a.top_genes).fit(patients.loc[train_pids])
    clin_all, gen_all = prep.transform_parts(patients.loc[pairs["patient_id"]])
    feats = extract_features(Backbone(model).to(device), pairs["path"].tolist(), ck["image_size"], device)
    y = torch.tensor(pairs["label"].map(c2i).to_numpy(), dtype=torch.long)
    clin, gen = torch.from_numpy(clin_all), torch.from_numpy(gen_all)

    def sel(s):
        i = np.flatnonzero((pairs["split"] == s).to_numpy())
        return feats[i], clin[i], gen[i], y[i]

    (ftr, ctr, gtr, ytr), (fva, cva, gva, yva), (fte, cte, gte, yte) = sel("train"), sel("val"), sel("test")
    ones = lambda n: torch.ones(n, 2)  # noqa: E731

    net = FusionNet(len(classes), clin.shape[1], gen.shape[1], img_dim=int(feats.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(ftr, ctr, gtr, ytr), batch_size=a.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(a.seed))
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        for f, c, g, t in dl:
            mask = (torch.rand(len(f), 2) > a.modality_dropout).float()
            opt.zero_grad()
            loss = ce(net(f.to(device), c.to(device), g.to(device), mask.to(device)), t.to(device))
            loss.backward()
            opt.step()
        pv = predict(net, fva, cva, gva, ones(len(fva)), device)
        vf = f1_score(yva.numpy(), pv.argmax(1), average="macro", zero_division=0)
        if vf > best:
            best, best_state, bad = vf, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    print(f"best val macro-F1 (all modalities): {best:.4f}")

    n = len(fte)
    conds = {
        "image_only": torch.zeros(n, 2),
        "image_clinical": torch.tensor([[1.0, 0.0]]).repeat(n, 1),
        "image_genomic": torch.tensor([[0.0, 1.0]]).repeat(n, 1),
        "image_clinical_genomic": ones(n),
    }
    ctx = {"pairing_mode": a.pairing, "patient_level_correspondence": summary["patient_level_correspondence"],
           "disclosure": summary["disclosure"], "best_val_macro_f1": round(best, 4)}
    rows = []
    # baseline: the image model's own softmax head on identical test rows
    with torch.no_grad():
        base = torch.softmax(model.head(fte.to(device)), 1).cpu().numpy()
    m0 = evaluate(yte.numpy(), base.argmax(1), classes, base)
    save_report(m0, out, "baseline_image_model_head", ctx)
    rows.append(("image_model_head (baseline)", m0))
    for name, mask in conds.items():
        pr = predict(net, fte, cte, gte, mask, device)
        m = evaluate(yte.numpy(), pr.argmax(1), classes, pr)
        save_report(m, out, f"fusion_{name}", ctx)
        rows.append((f"fusion {name}", m))
    abl = pd.DataFrame([{
        "condition": n, "accuracy": m["accuracy"], "macro_precision": m["macro"]["precision"],
        "macro_recall": m["macro"]["recall"], "macro_f1": m["macro"]["f1"],
        "ci95_acc_lo": m.get("ci95", {}).get("accuracy", [np.nan, np.nan])[0],
        "ci95_acc_hi": m.get("ci95", {}).get("accuracy", [np.nan, np.nan])[1],
    } for n, m in rows])
    abl.to_csv(out / "ablation_table.csv", index=False)
    print("\n" + abl.round(4).to_string(index=False))

    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "classes": classes, "clin_dim": int(clin.shape[1]), "gen_dim": int(gen.shape[1]), "img_dim": int(feats.shape[1]),
        "pairing_mode": a.pairing, "patient_level_correspondence": bool(summary["patient_level_correspondence"]),
    }, out / "fusion_model.pt")
    joblib.dump(prep, out / "fusion_preprocessor.joblib")
    if a.pairing == "random":
        print("\nNOTE: random pairing = control experiment. Expect fusion ~ image-only; "
              "do not present it as evidence of multimodal benefit.")


if __name__ == "__main__":
    main()
