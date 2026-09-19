
import json
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score

from eng_dataset import ECGDataset
from model import ECGCNN


BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "experiments/best/model.pth"
OUTPUT_PATH = "experiments/best/thresholds.json"
MIN_PRECISION_AT_LOW_THRESHOLD = 0.50


print("Using device:", DEVICE)


# Validation dataset
val_dataset = ECGDataset("val_split.csv")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

label_names = val_dataset.label_columns


# Load model
model = ECGCNN(num_classes=94).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

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

threshold_candidates = np.arange(
    0.05,
    0.96,
    0.05
)

for i, label in enumerate(label_names):

    true_labels = y_true[:, i]
    probabilities = y_prob[:, i]

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_precision = precision_score(
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
            best_threshold = float(threshold)

    thresholds[label] = best_threshold

    results.append({
        "label": label,
        "threshold": best_threshold,
        "validation_f1": best_f1
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


print("\nThreshold tuning completed!")
print("Saved:", OUTPUT_PATH)