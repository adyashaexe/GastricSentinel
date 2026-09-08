import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import joblib
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(ROOT, "data", "images", "extracted", "GasHisSDB")
GENOMIC_PATH = os.path.join(ROOT, "data", "genomics", "genomic_features.npz")
CLINICAL_PATH = os.path.join(ROOT, "data", "clinical", "clinical_features.npz")
GENOMIC_MODEL_PATH = os.path.join(ROOT, "model", "genomic_clinical_model.joblib")
IMAGE_MODEL_PATH = os.path.join(ROOT, "model", "gastric_resnet50.pkl")

OUTPUT_MODEL = os.path.join(ROOT, "model", "fusion_model.pkl")
OUTPUT_REPORT = os.path.join(ROOT, "model", "fusion_test_report.json")

MAX_IMAGES = 2000
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model():
    checkpoint = torch.load(
        IMAGE_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint)
        )

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)

    else:
        raise ValueError(
            f"Unsupported image model format: {type(checkpoint)}"
        )

    model.to(DEVICE)
    model.eval()

    return model


def get_images():
    # Label indices must match how train.py's ImageFolder assigned them:
    # alphabetical order -> Abnormal=0, Normal=1. (Previously this was
    # hardcoded backwards, which silently inverted every prediction.)
    normal = []
    abnormal = []

    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(root, file)

            if os.path.basename(root).lower() == "normal":
                normal.append(path)

            elif os.path.basename(root).lower() == "abnormal":
                abnormal.append(path)

    normal = sorted(normal)
    abnormal = sorted(abnormal)

    n = MAX_IMAGES // 2

    normal = normal[:n]
    abnormal = abnormal[:n]

    paths = abnormal + normal
    labels = [0] * len(abnormal) + [1] * len(normal)

    print(f"Using image subset: {len(paths)}")
    print(f"Abnormal: {len(abnormal)} | Normal: {len(normal)}")

    return paths, np.asarray(labels)


def get_image_predictions(model, paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    predictions = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(paths), BATCH_SIZE),
            desc="Image inference"
        ):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = []

            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                images.append(transform(image))

            batch = torch.stack(images).to(DEVICE)
            output = model(batch)

            if output.ndim == 1:
                probs = torch.sigmoid(output)

            elif output.shape[1] == 1:
                probs = torch.sigmoid(output[:, 0])

            else:
                probs = torch.softmax(output, dim=1)[:, 1]

            predictions.extend(
                probs.cpu().numpy()
            )

    return np.asarray(predictions)


def load_genomic_clinical():
    genomic = np.load(
        GENOMIC_PATH,
        allow_pickle=True
    )

    Xg = genomic["features"]
    patient_ids = genomic["patient_ids"]

    if Xg.shape[0] != len(patient_ids):
        if Xg.shape[1] == len(patient_ids):
            Xg = Xg.T
        else:
            raise ValueError(
                f"Genomic shape {Xg.shape} does not match "
                f"{len(patient_ids)} patient IDs"
            )

    clinical = np.load(
        CLINICAL_PATH,
        allow_pickle=True
    )

    Xc = []

    for patient_id in patient_ids:
        patient_id = str(patient_id)

        if patient_id in clinical.files:
            value = np.asarray(
                clinical[patient_id],
                dtype=np.float32
            )
        else:
            value = np.zeros(3, dtype=np.float32)

        Xc.append(value)

    Xc = np.asarray(Xc, dtype=np.float32)

    if Xg.shape[0] != Xc.shape[0]:
        raise ValueError(
            f"Genomic samples {Xg.shape[0]} != "
            f"clinical samples {Xc.shape[0]}"
        )

    X = np.hstack([Xg, Xc])

    print(f"Genomic features: {Xg.shape}")
    print(f"Clinical features: {Xc.shape}")
    print(f"Combined features: {X.shape}")

    return X, patient_ids


def main():
    print("Loading models...")

    image_model = load_image_model()

    genomic_model = joblib.load(
        GENOMIC_MODEL_PATH
    )

    print("Generating image predictions...")

    image_paths, image_labels = get_images()

    image_probs = get_image_predictions(
        image_model,
        image_paths
    )

    image_preds = (
        image_probs >= 0.5
    ).astype(int)

    print(f"Images: {len(image_paths)}")

    print("Generating genomic predictions...")

    genomic_features, patient_ids = load_genomic_clinical()

    expected = genomic_model.n_features_in_

    if genomic_features.shape[1] != expected:
        raise ValueError(
            f"Genomic feature shape "
            f"{genomic_features.shape} does not match "
            f"model expectation of {expected} features"
        )

    genomic_probs = genomic_model.predict_proba(
        genomic_features
    )

    genomic_preds = genomic_model.predict(
        genomic_features
    )

    classes = [
        str(x)
        for x in genomic_model.classes_
    ]

    print(f"Genomic samples: {len(patient_ids)}")
    print(f"Genomic model features: {expected}")
    print(f"Classes: {classes}")

    accuracy = accuracy_score(
        image_labels,
        image_preds
    )

    precision = precision_score(
        image_labels,
        image_preds,
        zero_division=0
    )

    recall = recall_score(
        image_labels,
        image_preds,
        zero_division=0
    )

    f1 = f1_score(
        image_labels,
        image_preds,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            image_labels,
            image_probs
        )
    except ValueError:
        auc = 0.0

    matrix = confusion_matrix(
        image_labels,
        image_preds
    )

    report = {
        "image_samples": len(image_paths),
        "image_accuracy": float(accuracy),
        "image_precision": float(precision),
        "image_recall": float(recall),
        "image_f1": float(f1),
        "image_roc_auc": float(auc),
        "image_confusion_matrix": matrix.tolist(),
        "genomic_clinical_samples": len(patient_ids),
        "genomic_clinical_features": int(expected),
        "genomic_classes": classes
    }

    fusion_model = {
        "type": "late_fusion",
        "image_model": IMAGE_MODEL_PATH,
        "genomic_clinical_model": GENOMIC_MODEL_PATH,
        "image_classes": [
            "Abnormal",
            "Normal"
        ],
        "genomic_classes": classes,
        "image_threshold": 0.5
    }

    with open(
        OUTPUT_MODEL,
        "wb"
    ) as f:
        pickle.dump(
            fusion_model,
            f
        )

    with open(
        OUTPUT_REPORT,
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("Image model results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix:")
    print(matrix)

    print()
    print("Saved: model/fusion_model.pkl")
    print("Saved: model/fusion_test_report.json")


if __name__ == "__main__":
    main()