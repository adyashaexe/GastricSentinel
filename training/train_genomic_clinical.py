"""
train_genomic_clinical.py

Trains a genomic + clinical fusion classifier for gastric cancer molecular
subtype prediction.

Fixes vs the old train_genomic.py:
  1. Run this from the repo root, e.g.:
         python training/train_genomic_clinical.py
     (the previous crash was from running `python train_genomic.py` as if
     it were a shell script from the wrong directory — that's what produced
     the `import: command not found` spam.)
  2. STAD_POLE only has 7 patients in TCGA-STAD. After inner-joining
     genomic + clinical + label data on patient ID, some of those 7 don't
     have complete records in all three files, so the class can drop below
     3 samples — and sklearn's stratified split needs at least 2 samples
     per class in EVERY split. Instead of crashing, this script detects any
     class below MIN_SAMPLES_PER_CLASS, folds it into an "Other" bucket,
     logs exactly what it did, and then stratifies safely on what's left.

Run from the repo root:
    python training/train_genomic_clinical.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_DIR = REPO_ROOT / "data" / "clinical"
GENOMIC_DIR = REPO_ROOT / "data" / "genomics"

GENOMIC_FEATURES_PATH = GENOMIC_DIR / "genomic_features.npz"
CLINICAL_FEATURES_PATH = CLINICAL_DIR / "clinical_features.npz"
SUBTYPE_LABELS_PATH = CLINICAL_DIR / "subtype_labels.json"

MODEL_OUTPUT_PATH = REPO_ROOT / "model" / "genomic_clinical_model.joblib"

MIN_SAMPLES_PER_CLASS = 3  # below this, a class gets folded into "Other"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_npz_bundle(path: Path, name: str):
    """
    Loads an npz feature file, supporting two formats:

    1. "Bundle" format (e.g. genomic_features.npz): a single 'features'
       array of shape (n_samples, n_features) plus a parallel 'patient_ids'
       array.
    2. "Per-patient" format (your clinical_features.npz): one array PER
       KEY, where each key IS the patient ID and its value is that
       patient's feature vector (e.g. 'TCGA-BR-8368' -> shape (3,)).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} at {path}. Generate it first with the matching "
            f"build_*.py script in training/."
        )
    data = np.load(path, allow_pickle=True)

    if "features" in data.files and "patient_ids" in data.files:
        return data["patient_ids"].astype(str), data["features"].astype(np.float32)

    # Per-patient format: every key is a patient ID.
    patient_ids = np.array(data.files)
    vectors = [data[k] for k in data.files]
    lengths = {v.shape for v in vectors}
    if len(lengths) > 1:
        raise ValueError(
            f"{path}: per-patient feature vectors have inconsistent shapes "
            f"({lengths}) — can't stack into a matrix."
        )
    features = np.stack(vectors).astype(np.float32)
    # Normalize keys to 12-char patient IDs to match the genomic file.
    patient_ids = np.array([pid[:12] for pid in patient_ids])
    return patient_ids, features


def load_labels(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing subtype labels at {path}. Generate it first with "
            f"training/build_subtype_labels.py."
        )
    with open(path) as f:
        raw = json.load(f)
    # Normalize keys to 12-char patient IDs to match the feature files.
    return {str(k)[:12]: v for k, v in raw.items()}


def merge_on_patient_id(genomic_ids, genomic_X, clinical_ids, clinical_X, labels: dict):
    genomic_map = {pid: i for i, pid in enumerate(genomic_ids)}
    clinical_map = {pid: i for i, pid in enumerate(clinical_ids)}

    common_ids = sorted(set(genomic_map) & set(clinical_map) & set(labels))
    dropped = (len(genomic_ids) - len(common_ids), len(clinical_ids) - len(common_ids))
    print(
        f"Patients with complete genomic+clinical+label records: {len(common_ids)}"
    )
    print(
        f"  (dropped {dropped[0]} genomic-only / {dropped[1]} clinical-only "
        f"or unlabeled patients)"
    )

    X_genomic = np.stack([genomic_X[genomic_map[pid]] for pid in common_ids])
    X_clinical = np.stack([clinical_X[clinical_map[pid]] for pid in common_ids])
    y = np.array([labels[pid] for pid in common_ids])
    X = np.concatenate([X_genomic, X_clinical], axis=1)
    return X, y, common_ids


def collapse_rare_classes(y: np.ndarray, min_count: int):
    values, counts = np.unique(y, return_counts=True)
    rare = {v for v, c in zip(values, counts) if c < min_count}

    if not rare:
        return y, {}

    print(
        f"\nClasses below {min_count} samples after merging — folding into "
        f"'Other' so the stratified split doesn't break:"
    )
    for v, c in zip(values, counts):
        if v in rare:
            print(f"  - {v}: {c} sample(s)")

    y_collapsed = np.array(["Other" if v in rare else v for v in y])
    return y_collapsed, {v: c for v, c in zip(values, counts) if v in rare}


def main():
    genomic_ids, genomic_X = load_npz_bundle(GENOMIC_FEATURES_PATH, "genomic features")
    clinical_ids, clinical_X = load_npz_bundle(CLINICAL_FEATURES_PATH, "clinical features")
    labels = load_labels(SUBTYPE_LABELS_PATH)

    X, y, patient_ids = merge_on_patient_id(
        genomic_ids, genomic_X, clinical_ids, clinical_X, labels
    )
    print(f"Fused feature matrix: {X.shape[0]} patients x {X.shape[1]} features")

    y_safe, folded = collapse_rare_classes(y, MIN_SAMPLES_PER_CLASS)

    if len(np.unique(y_safe)) < 2:
        raise ValueError(
            "After folding rare classes there's only one class left — can't "
            "train a classifier. Lower MIN_SAMPLES_PER_CLASS or check your "
            "label file."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_safe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_safe,
    )
    print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    labels_sorted = sorted(np.unique(y_safe))
    print(labels_sorted)
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    if folded:
        print(
            "\nNote: the following original subtypes were merged into "
            "'Other' for this run due to too few samples:"
        )
        for v, c in folded.items():
            print(f"  - {v} ({c} sample(s))")
        print(
            "These patients are still in the model, just not predicted as "
            "their own class. Consider gathering more STAD_POLE cases or "
            "reporting this subtype's performance separately with a "
            "leave-one-out check instead of a train/test split."
        )

    try:
        import joblib

        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_OUTPUT_PATH)
        print(f"\nSaved trained model to {MODEL_OUTPUT_PATH}")
    except ImportError:
        print("\n(joblib not installed — skipping model save, everything else ran fine)")


if __name__ == "__main__":
    main()