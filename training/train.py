import os
import json
import copy
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


DATA_PATH = os.path.join(
    "data",
    "images",
    "extracted",
    "GasHisSDB",
    "160"
)

MODEL_DIR = "model"

MODEL_SAVE_PATH = os.path.join(
    MODEL_DIR,
    "gastric_resnet50.pkl"
)

REPORT_SAVE_PATH = os.path.join(
    MODEL_DIR,
    "gastric_test_report.json"
)


BATCH_SIZE = 64
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

IMAGE_SIZE = 224
SEED = 42
NUM_WORKERS = 0


CLASSES = [
    "Abnormal",
    "Normal"
]

CLASS_TO_IDX = {
    "Abnormal": 0,
    "Normal": 1
}


def set_seed(seed=42):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class GasHisDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def collect_samples():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n"
        )

    print("\nDataset:")
    print(DATA_PATH)

    samples = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    for root, _, files in os.walk(DATA_PATH):

        folder_name = os.path.basename(root)

        if folder_name not in CLASS_TO_IDX:
            continue

        label = CLASS_TO_IDX[folder_name]

        for filename in files:

            if filename.lower().endswith(
                valid_extensions
            ):

                image_path = os.path.join(
                    root,
                    filename
                )

                samples.append(
                    (
                        image_path,
                        label
                    )
                )

    if not samples:

        raise RuntimeError(
            "No images found in the 160 dataset."
        )

    abnormal_count = sum(
        label == CLASS_TO_IDX["Abnormal"]
        for _, label in samples
    )

    normal_count = sum(
        label == CLASS_TO_IDX["Normal"]
        for _, label in samples
    )

    print(
        f"\nTotal images: {len(samples)}"
    )

    print(
        f"Abnormal: {abnormal_count}"
    )

    print(
        f"Normal: {normal_count}"
    )

    return samples


def get_data_loaders():

    samples = collect_samples()

    labels = [
        label
        for _, label in samples
    ]

    indices = list(
        range(len(samples))
    )

    trainval_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=SEED
    )

    trainval_labels = [
        labels[i]
        for i in trainval_indices
    ]

    train_indices, val_indices = train_test_split(
        trainval_indices,
        test_size=VAL_SPLIT,
        stratify=trainval_labels,
        random_state=SEED
    )

    train_samples = [
        samples[i]
        for i in train_indices
    ]

    val_samples = [
        samples[i]
        for i in val_indices
    ]

    test_samples = [
        samples[i]
        for i in test_indices
    ]

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = GasHisDataset(
        train_samples,
        train_transform
    )

    val_dataset = GasHisDataset(
        val_samples,
        eval_transform
    )

    test_dataset = GasHisDataset(
        test_samples,
        eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"\nTrain: {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Test: {len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


def build_model():

    model = models.resnet50(
        weights="IMAGENET1K_V1"
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(CLASSES)
    )

    return model


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train=False
):

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    context = (
        torch.enable_grad()
        if train
        else torch.no_grad()
    )

    with context:

        for inputs, labels in tqdm(
            loader,
            desc="Training" if train else "Validation"
        ):

            inputs = inputs.to(device)
            labels = labels.to(device)

            if train:

                optimizer.zero_grad(
                    set_to_none=True
                )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            if train:

                loss.backward()

                optimizer.step()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1,
        all_labels,
        all_predictions
    )


def format_time(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    seconds = (
        seconds % 60
    )

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{seconds}s"
        )

    return (
        f"{minutes}m "
        f"{seconds}s"
    )


def train_model():

    set_seed(SEED)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    device = get_device()

    print(
        f"\nCompute device: {device}"
    )

    (
        train_loader,
        val_loader,
        test_loader
    ) = get_data_loaders()

    print(
        "\nLoading ResNet50..."
    )

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS
    )

    best_f1 = -1.0

    best_state = None

    epochs_without_improvement = 0

    start_time = time.time()

    epoch_times = []

    print(
        "\nStarting training..."
    )

    print(
        f"Maximum epochs: {MAX_EPOCHS}"
    )

    print(
        f"Early stopping: {PATIENCE}"
    )

    for epoch in range(MAX_EPOCHS):

        epoch_start = time.time()

        current_lr = (
            scheduler.get_last_lr()[0]
        )

        print(
            f"\nEpoch {epoch + 1}/{MAX_EPOCHS}"
        )

        print(
            f"Learning rate: {current_lr:.2e}"
        )

        (
            train_loss,
            train_accuracy,
            train_f1,
            _,
            _
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True
        )

        (
            val_loss,
            val_accuracy,
            val_f1,
            _,
            _
        ) = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False
        )

        scheduler.step()

        print(
            f"\nTrain Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_f1:.4f}"
        )

        print(
            f"\nVal Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            torch.save(
                best_state,
                MODEL_SAVE_PATH
            )

            print(
                "\nBest model saved."
            )

            print(
                f"Best validation F1: "
                f"{best_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        epoch_time = (
            time.time()
            - epoch_start
        )

        epoch_times.append(
            epoch_time
        )

        average_epoch_time = (
            sum(epoch_times)
            / len(epoch_times)
        )

        epochs_completed = epoch + 1

        epochs_remaining = (
            MAX_EPOCHS
            - epochs_completed
        )

        estimated_remaining = (
            average_epoch_time
            * epochs_remaining
        )

        finish_time = (
            datetime.now()
            + timedelta(
                seconds=estimated_remaining
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"\nEpoch time: "
            f"{format_time(epoch_time)}"
        )

        print(
            f"Elapsed: "
            f"{format_time(elapsed)}"
        )

        print(
            f"Estimated remaining: "
            f"{format_time(estimated_remaining)}"
        )

        print(
            f"Estimated finish: "
            f"{finish_time.strftime('%I:%M %p')}"
        )

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    total_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining complete in "
        f"{format_time(total_time)}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    model.load_state_dict(
        best_state
    )

    print(
        "\nEvaluating test set..."
    )

    (
        test_loss,
        test_accuracy,
        test_f1,
        test_labels,
        test_predictions
    ) = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        train=False
    )

    report = classification_report(
        test_labels,
        test_predictions,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        test_labels,
        test_predictions,
        target_names=CLASSES,
        zero_division=0
    )

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )

    report["overall"] = {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "macro_f1": test_f1
    }

    report["confusion_matrix"] = (
        cm.tolist()
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro-F1: "
        f"{test_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        report_text
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    with open(
        REPORT_SAVE_PATH,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_SAVE_PATH
    )

    print(
        f"\nTest report saved to:"
    )

    print(
        REPORT_SAVE_PATH
    )


if __name__ == "__main__":
    train_model()