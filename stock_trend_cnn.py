import re
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("./data2")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

LOOKBACK = 20
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

IMG_H = IMAGE_HEIGHT[LOOKBACK]
IMG_W = IMAGE_WIDTH[LOOKBACK]

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 15
WEIGHT_DECAY = 1e-4
DROPOUT = 0.35
SEED = 42
NUM_WORKERS = 0
USE_MPS_IF_AVAILABLE = True
PRINT_EVERY = 10
EARLY_STOPPING_PATIENCE = 4
THRESHOLD = 0.48

TRAIN_YEARS = [2014]
VAL_YEARS = [2015]
TEST_YEARS = [2016]

MAX_TRAIN_SAMPLES = 10000
MAX_VAL_SAMPLES = 2000
MAX_TEST_SAMPLES = 2000


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# DEVICE
# =========================================================
def get_device():
    if USE_MPS_IF_AVAILABLE and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


# =========================================================
# FILE DISCOVERY
# =========================================================
def extract_year_from_image_file(filepath):
    match = re.search(r"ma_(\d{4})_images\.dat$", str(filepath))
    return int(match.group(1)) if match else None


def find_available_years(data_dir):
    image_files = sorted(glob.glob(str(data_dir / "*_images.dat")))
    years = []

    for f in image_files:
        y = extract_year_from_image_file(f)
        if y is not None:
            label_file = data_dir / f"20d_month_has_vb_[20]_ma_{y}_labels_w_delay.feather"
            if label_file.exists():
                years.append(y)

    return sorted(years)


# =========================================================
# DATASET
# =========================================================
class MultiYearStockDataset(Dataset):
    def __init__(self, data_dir, years, normalize=True):
        self.data_dir = Path(data_dir)
        self.years = years
        self.normalize = normalize

        self.year_info = []
        self.index_map = []

        for year in years:
            image_file = self.data_dir / f"20d_month_has_vb_[20]_ma_{year}_images.dat"
            label_file = self.data_dir / f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"

            if not image_file.exists() or not label_file.exists():
                continue

            labels = pd.read_feather(label_file).copy()

            if "Ret_5d" not in labels.columns:
                raise ValueError(f"'Ret_5d' column missing in {label_file}")

            labels["Ret_5d_binary"] = (labels["Ret_5d"] > 0).astype(np.int64)
            labels["year"] = year

            num_samples = len(labels)

            arr = np.memmap(image_file, dtype=np.uint8, mode="r")
            expected_size = num_samples * IMG_H * IMG_W
            if arr.size != expected_size:
                raise ValueError(
                    f"Size mismatch for year {year}: found {arr.size}, expected {expected_size}"
                )

            self.year_info.append(
                {
                    "year": year,
                    "image_file": image_file,
                    "labels": labels.reset_index(drop=True),
                    "num_samples": num_samples,
                    "memmap": arr,
                    "targets": labels["Ret_5d_binary"].to_numpy(dtype=np.int64),
                }
            )

        for year_idx, info in enumerate(self.year_info):
            for sample_idx in range(info["num_samples"]):
                self.index_map.append((year_idx, sample_idx))

        print(f"Built dataset for years {years} with {len(self.index_map)} samples.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        year_idx, sample_idx = self.index_map[idx]
        info = self.year_info[year_idx]

        start = sample_idx * IMG_H * IMG_W
        end = start + IMG_H * IMG_W
        flat = info["memmap"][start:end]
        img = np.asarray(flat, dtype=np.float32).reshape(IMG_H, IMG_W)

        if self.normalize:
            img = img / 255.0

        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.expand_dims(img, axis=0)
        y = float(info["targets"][sample_idx])

        return torch.tensor(img, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_labels_array(self):
        ys = []
        for info in self.year_info:
            ys.extend(info["targets"].tolist())
        return np.array(ys)

    def get_metadata_df(self):
        dfs = []
        for info in self.year_info:
            dfs.append(info["labels"])
        return pd.concat(dfs, ignore_index=True)


def maybe_make_subset(dataset, max_samples, seed=42):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    indices = np.sort(indices)
    return Subset(dataset, indices.tolist())


def get_subset_metadata_df(dataset_or_subset):
    if isinstance(dataset_or_subset, Subset):
        base = dataset_or_subset.dataset
        full_meta = base.get_metadata_df().reset_index(drop=True)
        return full_meta.iloc[dataset_or_subset.indices].reset_index(drop=True)
    return dataset_or_subset.get_metadata_df().reset_index(drop=True)


def get_subset_labels_array(dataset_or_subset):
    if isinstance(dataset_or_subset, Subset):
        base = dataset_or_subset.dataset
        full_y = base.get_labels_array()
        return full_y[dataset_or_subset.indices]
    return dataset_or_subset.get_labels_array()


# =========================================================
# MODEL
# =========================================================
class BalancedStockCNN(nn.Module):
    def __init__(self, dropout=0.35):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


# =========================================================
# EVAL
# =========================================================
def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
        "threshold": threshold,
    }


# =========================================================
# TRAIN
# =========================================================
def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, device):
    pos_weight = torch.tensor([1.1], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_model_path = RESULTS_DIR / "best_model.pt"
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % PRINT_EVERY == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Current Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, criterion, device, threshold=THRESHOLD)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | "
            f"Val Rec: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest model saved to: {best_model_path}")
    return history, best_model_path


# =========================================================
# PLOTS / SAVES
# =========================================================
def plot_loss_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], marker="o", label="Train Loss")
    plt.plot(history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=200)
    plt.close()


def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def save_metrics_report(metrics, split_name="test"):
    report_path = RESULTS_DIR / f"{split_name}_metrics.txt"
    with open(report_path, "w") as f:
        f.write(f"{split_name.upper()} METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Threshold : {metrics['threshold']:.2f}\n")
        f.write(f"Loss      : {metrics['loss']:.4f}\n")
        f.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision : {metrics['precision']:.4f}\n")
        f.write(f"Recall    : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score  : {metrics['f1']:.4f}\n")


def save_predictions(meta_test, test_metrics):
    out = meta_test.copy()
    out["y_true"] = test_metrics["targets"]
    out["y_pred"] = test_metrics["preds"]
    out["y_prob"] = test_metrics["probs"]
    out.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)


def save_experiment_summary(history, test_metrics):
    rows = []
    for i in range(len(history["train_loss"])):
        rows.append({
            "epoch": i + 1,
            "train_loss": history["train_loss"][i],
            "val_loss": history["val_loss"][i],
            "val_accuracy": history["val_accuracy"][i],
            "val_precision": history["val_precision"][i],
            "val_recall": history["val_recall"][i],
            "val_f1": history["val_f1"][i],
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "training_history.csv", index=False)

    pd.DataFrame([{
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_threshold": test_metrics["threshold"],
    }]).to_csv(RESULTS_DIR / "test_summary.csv", index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    available_years = find_available_years(DATA_DIR)
    print("Available years:", available_years)

    train_years = [y for y in TRAIN_YEARS if y in available_years]
    val_years = [y for y in VAL_YEARS if y in available_years]
    test_years = [y for y in TEST_YEARS if y in available_years]

    print("Train years:", train_years)
    print("Val years  :", val_years)
    print("Test years :", test_years)

    if len(train_years) == 0 or len(val_years) == 0 or len(test_years) == 0:
        raise ValueError("Adjust TRAIN_YEARS / VAL_YEARS / TEST_YEARS based on available files.")

    full_train_ds = MultiYearStockDataset(DATA_DIR, train_years)
    full_val_ds = MultiYearStockDataset(DATA_DIR, val_years)
    full_test_ds = MultiYearStockDataset(DATA_DIR, test_years)

    train_ds = maybe_make_subset(full_train_ds, MAX_TRAIN_SAMPLES, seed=SEED)
    val_ds = maybe_make_subset(full_val_ds, MAX_VAL_SAMPLES, seed=SEED)
    test_ds = maybe_make_subset(full_test_ds, MAX_TEST_SAMPLES, seed=SEED)

    y_train = get_subset_labels_array(train_ds)
    y_val = get_subset_labels_array(val_ds)
    y_test = get_subset_labels_array(test_ds)

    print("\nFinal dataset sizes used:")
    print("Train:", len(train_ds))
    print("Val  :", len(val_ds))
    print("Test :", len(test_ds))

    print("\nClass balance:")
    print("Train positive ratio:", y_train.mean())
    print("Val positive ratio  :", y_val.mean())
    print("Test positive ratio :", y_test.mean())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = BalancedStockCNN(dropout=DROPOUT).to(device)

    history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    plot_loss_curves(history)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    pos_weight = torch.tensor([1.1], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_metrics = evaluate_model(model, test_loader, criterion, device, threshold=THRESHOLD)

    print("\nTEST RESULTS")
    print("=" * 50)
    print(f"Threshold : {test_metrics['threshold']:.2f}")
    print(f"Test Loss : {test_metrics['loss']:.4f}")
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1 Score  : {test_metrics['f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_metrics["targets"], test_metrics["preds"], zero_division=0))

    plot_conf_matrix(test_metrics["targets"], test_metrics["preds"])
    save_metrics_report(test_metrics, split_name="test")

    meta_test = get_subset_metadata_df(test_ds)
    save_predictions(meta_test, test_metrics)
    save_experiment_summary(history, test_metrics)

    print("\nSaved outputs in ./results")


if __name__ == "__main__":
    main()