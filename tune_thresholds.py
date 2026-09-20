
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eng_dataset import ECGDataset
from model import ECGCNN


# ================= CONFIG =================

NUM_CLASSES = 94
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

VAL_CSV = PROJECT_ROOT / "val_split.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "residual_focal_best"
    / "model.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "residual_focal_best"
    / "thresholds.json"
)

# ==========================================


def load_model():
    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)

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

    return model


def collect_predictions(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(DEVICE)

            logits = model(signals)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)

    return probabilities, labels


def find_best_threshold(probabilities, labels):
    best_threshold = 0.5
    best_f1 = -1.0

    # Test thresholds from 0.05 to 0.95
    thresholds = np.arange(0.05, 0.96, 0.01)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        true_positives = np.sum(
            (predictions == 1) & (labels == 1)
        )

        false_positives = np.sum(
            (predictions == 1) & (labels == 0)
        )

        false_negatives = np.sum(
            (predictions == 0) & (labels == 1)
        )

        precision = (
            true_positives /
            (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

        recall = (
            true_positives /
            (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def main():
    print("Using device:", DEVICE)
    print("Loading validation dataset...")

    dataset = ECGDataset(VAL_CSV)
    label_names = dataset.label_columns

    if len(label_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} labels, "
            f"found {len(label_names)}"
        )

    model = load_model()

    print("Collecting validation predictions...")
    probabilities, labels = collect_predictions(model, dataset)

    print("Finding optimal threshold for every label...\n")

    thresholds = {}

    for class_index, label_name in enumerate(label_names):
        best_threshold, best_f1 = find_best_threshold(
            probabilities[:, class_index],
            labels[:, class_index]
        )

        thresholds[label_name] = best_threshold

        print(
            f"{label_name}: "
            f"threshold={best_threshold:.2f}, "
            f"validation F1={best_f1:.4f}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w") as file:
        json.dump(thresholds, file, indent=2)

    print("\n==========================================")
    print("Threshold tuning completed!")
    print("Saved to:")
    print(OUTPUT_PATH)
    print("==========================================")


if __name__ == "__main__":
    main()