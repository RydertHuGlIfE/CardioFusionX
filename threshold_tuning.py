
import json
import csv
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from eng_dataset import ECGDataset
from model import ECGCNN


BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "experiments/residual_focal_best/model.pth"
OUTPUT_PATH = PROJECT_ROOT / "experiments/residual_focal_best/thresholds.json"
RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments/residual_focal_best/threshold_tuning_results.csv"
)
MIN_PRECISION_AT_LOW_THRESHOLD = 0.50
NUM_CLASSES = 94


print("Using device:", DEVICE)


# Validation dataset
val_dataset = ECGDataset(PROJECT_ROOT / "val_split.csv")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

label_names = val_dataset.label_columns


if len(label_names) != NUM_CLASSES:
    raise ValueError(f"Expected {NUM_CLASSES} labels, found {len(label_names)}")

train_dataset = ECGDataset(PROJECT_ROOT / "train_split.csv")
if train_dataset.label_columns != val_dataset.label_columns:
    raise ValueError(
        "Train and validation label columns differ."
    )

model_path = Path(MODEL_PATH)
if not model_path.exists():
    print(f"WARNING: residual-Focal checkpoint not found: {model_path}")
    raise SystemExit(0)


# Load model
model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    config = checkpoint.get("config", {})
    if config.get("experiment") != "residual_focal":
        raise ValueError(
            "Threshold tuning requires a residual_focal checkpoint."
        )
    if config.get("num_classes") != NUM_CLASSES:
        raise ValueError("Checkpoint class count does not match 94 labels.")
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    raise ValueError(
        "Checkpoint metadata is missing; expected a residual_focal checkpoint."
    )

model.eval()


# Collect validation predictions
all_labels = []
all_probabilities = []

with torch.no_grad():
    for signals, labels in val_loader:
        signals = signals.to(DEVICE)

        outputs = model(signals)
        probabilities = torch.sigmoid(outputs)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )


y_true = np.concatenate(all_labels)
y_prob = np.concatenate(all_probabilities)


# Find best threshold for every label
thresholds = {}
results = []

threshold_candidates = np.round(
    np.arange(0.05, 0.951, 0.05),
    2
)

for i, label in enumerate(label_names):

    true_labels = y_true[:, i]
    probabilities = y_prob[:, i]

    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    if true_labels.sum() == 0:
        print(
            f"WARNING: {label} has zero positive samples in validation."
        )
        thresholds[label] = 0.5
        results.append({
            "label": label,
            "threshold": 0.5,
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "validation_f1": 0.0,
            "validation_support": 0,
            "predicted_positive_count": int(
                (probabilities >= 0.5).sum()
            ),
        })
        continue

    for threshold in threshold_candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_precision = precision_score(
            true_labels,
            predictions,
            zero_division=0
        )
        current_recall = recall_score(
            true_labels,
            predictions,
            zero_division=0
        )

        if (
            np.isclose(threshold, 0.05)
            and current_precision < MIN_PRECISION_AT_LOW_THRESHOLD
        ):
            continue

        current_f1 = f1_score(
            true_labels,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_precision = current_precision
            best_recall = current_recall
            best_threshold = float(threshold)

    thresholds[label] = best_threshold

    results.append({
        "label": label,
        "threshold": best_threshold,
        "validation_precision": best_precision,
        "validation_recall": best_recall,
        "validation_f1": best_f1,
        "validation_support": int(true_labels.sum()),
        "predicted_positive_count": int(
            (probabilities >= best_threshold).sum()
        ),
    })

    print(
        f"{label}: "
        f"threshold={best_threshold:.2f}, "
        f"F1={best_f1:.4f}"
    )


# Save thresholds
output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with output_path.open("w") as file:
    json.dump(thresholds, file, indent=4)

results_path = Path(RESULTS_PATH)
results_path.parent.mkdir(
    parents=True,
    exist_ok=True
)
with results_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "label",
            "threshold",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_support",
            "predicted_positive_count",
        ],
    )
    writer.writeheader()
    for result in results:
        writer.writerow(result)

threshold_counts = {}
for threshold in thresholds.values():
    key = f"{threshold:.2f}"
    threshold_counts[key] = threshold_counts.get(key, 0) + 1


print("\nThreshold tuning completed!")
print("Saved:", OUTPUT_PATH)
print("Saved:", RESULTS_PATH)
print("Threshold counts:", threshold_counts)