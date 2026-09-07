import os
import json
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = os.path.join('cancer_project', 'raw')  # Update this to your actual data path
MODEL_SAVE_PATH = os.path.join('models', 'gastric_resnet50.pth')
REPORT_SAVE_PATH = os.path.join('models', 'test_report.json')

BATCH_SIZE = 32
MAX_EPOCHS = 100          # upper bound; early stopping will usually cut this short
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10             # early-stopping patience, measured on val macro-F1
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15          # of the *remaining* data after the test split is carved out
SEED = 42

# 8 Classes
CLASSES = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def get_data_loaders():
    print("Preparing Data...")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset three times with different transforms attached, then
    # slice each with the *same* indices — avoids custom Dataset subclassing.
    train_backing = datasets.ImageFolder(DATA_PATH, transform=train_transform)
    val_backing = datasets.ImageFolder(DATA_PATH, transform=eval_transform)
    test_backing = datasets.ImageFolder(DATA_PATH, transform=eval_transform)

    all_idx = list(range(len(train_backing)))
    targets = train_backing.targets

    # 1st split: carve out the held-out test set (stratified)
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SPLIT, stratify=targets, random_state=SEED
    )

    # 2nd split: train vs val, stratified on the remaining data
    trainval_targets = [targets[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=VAL_SPLIT, stratify=trainval_targets, random_state=SEED
    )

    train_ds = Subset(train_backing, train_idx)
    val_ds = Subset(val_backing, val_idx)
    test_ds = Subset(test_backing, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Data Loaded: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test "
          f"({len(train_ds)+len(val_ds)+len(test_ds)} total)")
    return train_loader, val_loader, test_loader


def build_model():
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze everything first...
    for param in model.parameters():
        param.requires_grad = False

    # ...then unfreeze layer4 (the last residual block) + the new head.
    # Full end-to-end fine-tuning risks overfitting on a small histology
    # dataset; unfreezing just the last block is a much better accuracy/
    # overfitting tradeoff than a fully frozen backbone.
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, desc="Training" if train else "Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, all_labels, all_preds


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders()

    print("Initializing ResNet50 (layer4 + fc unfrozen)...")
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} (lr={scheduler.get_last_lr()[0]:.2e})")
        print("-" * 10)

        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        print(f"Train Loss: {train_loss:.4f}  Macro-F1: {train_f1:.4f}")

        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Val   Loss: {val_loss:.4f}  Macro-F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(best_state, MODEL_SAVE_PATH)
            print(f"Model improved (val macro-F1={val_f1:.4f}). Saved to {MODEL_SAVE_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs.")
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Macro-F1: {best_f1:.4f}")

    # --- Final, one-time evaluation on the held-out TEST set ---
    print("\nEvaluating best checkpoint on held-out test set...")
    model.load_state_dict(best_state)
    test_loss, test_f1, test_labels, test_preds = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    report = classification_report(
        test_labels, test_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    print(f"Test Loss: {test_loss:.4f}  Test Macro-F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASSES, zero_division=0))

    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)
    with open(REPORT_SAVE_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full test report saved to {REPORT_SAVE_PATH} — these are the numbers to cite in the paper.")


if __name__ == "__main__":
    train_model()
