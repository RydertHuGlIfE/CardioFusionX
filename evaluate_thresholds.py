import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from eng_dataset import ECGDataset
from model import ECGCNN


# =========================
# CONFIG
# =========================

NUM_CLASSES = 94
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT /
    "experiments/residual_focal_best/model.pth"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT /
    "experiments/residual_focal_best/thresholds.json"
)

TEST_CSV = PROJECT_ROOT / "test_split.csv"


# =========================
# LOAD MODEL
# =========================

def load_model():
    model = ECGCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded epoch:",
        checkpoint.get("epoch", "unknown")
    )

    return model


# =========================
# LOAD THRESHOLDS
# =========================

def load_thresholds(label_names):
    with open(THRESHOLDS_PATH) as f:
        threshold_data = json.load(f)

    return np.array(
        [
            float(threshold_data[label])
            for label in label_names
        ],
        dtype=np.float32
    )


# =========================
# COLLECT PREDICTIONS
# =========================

@torch.no_grad()
def collect_predictions(model, dataset):
    all_targets = []
    all_probabilities = []

    for index in range(len(dataset)):
        signal, labels = dataset[index]

        signal = signal.unsqueeze(0).to(DEVICE)

        logits = model(signal)
        probabilities = torch.sigmoid(logits)

        all_targets.append(
            labels.numpy()
        )

        all_probabilities.append(
            probabilities.squeeze(0)
            .cpu()
            .numpy()
        )

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(dataset)}"
            )

    return (
        np.array(all_targets),
        np.array(all_probabilities)
    )


# =========================
# EVALUATE
# =========================

def evaluate(name, targets, probabilities, thresholds):
    predictions = (
        probabilities >= thresholds
    ).astype(int)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Micro Precision:",
        f"{precision_score(targets, predictions, average='micro', zero_division=0):.4f}"
    )

    print(
        "Micro Recall:",
        f"{recall_score(targets, predictions, average='micro', zero_division=0):.4f}"
    )

    print(
        "Micro F1:",
        f"{f1_score(targets, predictions, average='micro', zero_division=0):.4f}"
    )

    print(
        "Macro F1:",
        f"{f1_score(targets, predictions, average='macro', zero_division=0):.4f}"
    )

    print(
        "Samples F1:",
        f"{f1_score(targets, predictions, average='samples', zero_division=0):.4f}"
    )


# =========================
# MAIN
# =========================

def main():
    print("Using device:", DEVICE)

    dataset = ECGDataset(TEST_CSV)
    label_names = dataset.label_columns

    if len(label_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} labels, "
            f"found {len(label_names)}"
        )

    model = load_model()
    tuned_thresholds = load_thresholds(label_names)

    targets, probabilities = collect_predictions(
        model,
        dataset
    )

    default_thresholds = np.full(
        NUM_CLASSES,
        0.5,
        dtype=np.float32
    )

    evaluate(
        "DEFAULT THRESHOLD = 0.50",
        targets,
        probabilities,
        default_thresholds
    )

    evaluate(
        "TUNED THRESHOLDS",
        targets,
        probabilities,
        tuned_thresholds
    )


if __name__ == "__main__":
    main()
