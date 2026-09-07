"""
train_fusion.py

Trains the multimodal FusionModel (image + clinical + genomic -> 8-class).

IMPORTANT — READ BEFORE RUNNING:
The image dataset (NCT-CRC-HE-100K-style tissue patches) has NO real per-patient
clinical or genomic data attached — it's anonymized tissue tiles with only a
class label. TCGA-STAD (real gastric cancer genomic/clinical records) is a
SEPARATE, unrelated patient cohort with no way to join it to these image tiles.

So the clinical/genomic features used here are SYNTHETIC: generated per-image
with a documented, deliberate statistical relationship to the true tissue
label (not pure noise), so the fusion model has real signal to learn from and
its SHAP explanations are meaningful. This is a legitimate way to validate a
fusion ARCHITECTURE, but it is not real patient data, and the paper must say
so explicitly. Do not describe this as "trained on TCGA-STAD" — it wasn't.

If you later get access to real paired data (e.g. TCGA-STAD's own diagnostic
slide images, matched to their own genomic/clinical records), swap out
`make_synthetic_clinical_genomic()` for a real loader and nothing else in
this script needs to change.
"""

import os
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from model_loader import FusionModel  # reuse the exact architecture used at inference time

# --- CONFIGURATION ---
DATA_PATH = os.path.join('cancer_project', 'raw')
IMAGE_MODEL_PATH = os.path.join('models', 'gastric_resnet50.pth')  # from train.py
FUSION_SAVE_PATH = os.path.join('models', 'gastric_fusion.pth')
REPORT_SAVE_PATH = os.path.join('models', 'fusion_test_report.json')

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
SEED = 42

CLASSES = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
MALIGNANT_CLASSES = {'TUM', 'STR'}  # used only to shape synthetic feature distributions


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_synthetic_clinical_genomic(class_name: str, rng: random.Random):
    """
    Generates ONE synthetic (clinical, genomic) sample tied to a tissue class.

    Rationale for the correlation (documented here so it can be cited/defended
    in the paper's methodology section):
      - Malignant-associated classes (TUM, STR) are given higher average
        'stage', higher 'gene_score', and higher 'genomic_risk', with noise,
        so the fusion model has a learnable (if synthetic) signal instead of
        pure random noise, which would make the fusion branches contribute
        nothing and the SHAP output meaningless.
      - age/gender are class-independent (no clinical basis to tie them to
        tissue type), so they're drawn from plausible population ranges only.
    """
    age = rng.gauss(60, 12)
    age = float(np.clip(age, 20, 95))

    gender = rng.choice(["male", "female"])

    is_malignant = class_name in MALIGNANT_CLASSES
    stage = rng.choices([1, 2, 3, 4], weights=[1, 2, 3, 4] if is_malignant else [4, 3, 2, 1])[0]

    gene_score = rng.gauss(0.7, 0.15) if is_malignant else rng.gauss(0.3, 0.15)
    gene_score = float(np.clip(gene_score, 0, 1))

    genomic_risk = rng.gauss(0.65, 0.2) if is_malignant else rng.gauss(0.25, 0.2)
    genomic_risk = float(np.clip(genomic_risk, 0, 1))

    return age, gender, stage, gene_score, genomic_risk


class FusionDataset(Dataset):
    """
    Wraps an ImageFolder dataset and attaches a synthetic clinical/genomic
    tuple to every image, generated deterministically from a fixed seed so
    the same image always gets the same synthetic values across runs.
    """
    def __init__(self, image_folder: datasets.ImageFolder, indices, classes):
        self.image_folder = image_folder
        self.indices = indices
        self.classes = classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        img, label_idx = self.image_folder[real_idx]
        class_name = self.classes[label_idx]

        # Deterministic per-sample RNG: same image -> same synthetic profile every run.
        rng = random.Random(f"{real_idx}-{SEED}")
        age, gender, stage, gene_score, genomic_risk = make_synthetic_clinical_genomic(class_name, rng)

        gender_map = {"male": 0, "female": 1}
        clinical = torch.tensor([age, gender_map[gender], stage]).float()

        genes = [gene_score, genomic_risk] + [0.0] * 18  # matches utils.genomic_to_tensor padding
        genomic = torch.tensor(genes).float()
        genomic = (genomic - genomic.mean()) / (genomic.std() + 1e-6)

        return img, clinical, genomic, label_idx


def get_loaders():
    print("Preparing Data (with synthetic clinical/genomic features)...")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    base = datasets.ImageFolder(DATA_PATH, transform=eval_transform)
    all_idx = list(range(len(base)))
    targets = base.targets

    trainval_idx, test_idx = train_test_split(all_idx, test_size=TEST_SPLIT, stratify=targets, random_state=SEED)
    trainval_targets = [targets[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(trainval_idx, test_size=VAL_SPLIT, stratify=trainval_targets, random_state=SEED)

    train_ds = FusionDataset(base, train_idx, CLASSES)
    val_ds = FusionDataset(base, val_idx, CLASSES)
    test_ds = FusionDataset(base, test_idx, CLASSES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Data Loaded: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    return train_loader, val_loader, test_loader


def load_frozen_feature_extractor(device):
    """Loads your trained image model (from train.py) and freezes it entirely —
    the fusion model should learn to combine features, not re-learn vision."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained image model not found at {IMAGE_MODEL_PATH}. Run train.py first — "
            "the fusion model needs a trained image backbone to extract features from."
        )
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    modules = list(model.children())[:-1]
    extractor = nn.Sequential(*modules).to(device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


def run_epoch(fusion_model, extractor, loader, criterion, optimizer, device, train: bool):
    fusion_model.train() if train else fusion_model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for imgs, clinical, genomic, labels in loader:
            imgs, clinical, genomic, labels = (
                imgs.to(device), clinical.to(device), genomic.to(device), labels.to(device)
            )

            with torch.no_grad():
                img_features = extractor(imgs).view(imgs.size(0), -1)

            if train:
                optimizer.zero_grad()

            outputs = fusion_model(img_features, clinical, genomic)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, all_labels, all_preds


def train_fusion():
    print("=" * 70)
    print("WARNING: clinical/genomic features in this run are SYNTHETIC.")
    print("See the module docstring at the top of this file before citing results.")
    print("=" * 70)

    device = get_device()
    print(f"Compute Device: {device}")

    train_loader, val_loader, test_loader = get_loaders()
    extractor = load_frozen_feature_extractor(device)

    fusion_model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_f1, _, _ = run_epoch(fusion_model, extractor, train_loader, criterion, optimizer, device, train=True)
        print(f"Train Loss: {train_loss:.4f}  Macro-F1: {train_f1:.4f}")

        val_loss, val_f1, _, _ = run_epoch(fusion_model, extractor, val_loader, criterion, optimizer, device, train=False)
        print(f"Val   Loss: {val_loss:.4f}  Macro-F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(fusion_model.state_dict())
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(FUSION_SAVE_PATH), exist_ok=True)
            torch.save(best_state, FUSION_SAVE_PATH)
            print(f"Model improved (val macro-F1={val_f1:.4f}). Saved to {FUSION_SAVE_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs.")
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    total_time = time.time() - start_time
    print(f"\nFusion training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Macro-F1: {best_f1:.4f}")

    print("\nEvaluating best fusion checkpoint on held-out test set...")
    fusion_model.load_state_dict(best_state)
    test_loss, test_f1, test_labels, test_preds = run_epoch(
        fusion_model, extractor, test_loader, criterion, optimizer, device, train=False
    )
    report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    print(f"Test Loss: {test_loss:.4f}  Test Macro-F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASSES, zero_division=0))

    report["_NOTE"] = "Clinical/genomic features used in this run are SYNTHETIC. See train_fusion.py docstring."
    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)
    with open(REPORT_SAVE_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full test report saved to {REPORT_SAVE_PATH}")


if __name__ == "__main__":
    train_fusion()
