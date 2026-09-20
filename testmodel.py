
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from eng_dataset import ECGDataset
from model import ECGCNN


# =========================
# CONFIG
# =========================

NUM_CLASSES = 94
NUM_LEADS = 12
SIGNAL_LENGTH = 5000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments/optimized_v1/best_macro_f1/model.pth"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "experiments/optimized_v1/best_macro_f1/conservative_thresholds.json"
)

DATASET_PATH = PROJECT_ROOT / "test_split.csv"


# =========================
# LOAD THRESHOLDS
# =========================

def load_thresholds(label_names, thresholds_path):
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Threshold file does not exist: {thresholds_path}"
        )

    with thresholds_path.open("r") as file:
        threshold_data = json.load(file)

    missing = [
        label for label in label_names
        if label not in threshold_data
    ]

    if missing:
        raise ValueError(
            f"Missing thresholds: {missing}"
        )

    return np.array(
        [
            float(threshold_data[label])
            for label in label_names
        ],
        dtype=np.float32
    )


# =========================
# LOAD MODEL
# =========================

def load_model(label_names, model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {model_path}"
        )

    model = ECGCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Invalid checkpoint format."
        )

    config = checkpoint.get("config", {})

    if config.get("experiment") != "optimized_v1":
        raise ValueError(
            "This is not an optimized_v1 checkpoint."
        )

    if config.get("num_classes") != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint class count does not match {NUM_CLASSES}."
        )

    checkpoint_labels = checkpoint.get("label_names")
    if checkpoint_labels != label_names:
        raise ValueError(
            "Checkpoint label order does not match the dataset label order."
        )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint architecture is incompatible with model.py."
        ) from error

    model.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    return model


# =========================
# EXTRACT DATASET SAMPLE
# =========================

def extract_sample(sample):
    """
    Supports common dataset formats:

    (signal, labels)
    (signal, labels, metadata)
    """

    if not isinstance(sample, (tuple, list)):
        raise ValueError(
            "Unexpected dataset sample format."
        )

    signal = sample[0]
    actual_labels = sample[1]

    return signal, actual_labels


# =========================
# CONVERT ACTUAL LABELS
# =========================

def get_actual_labels(actual_labels, label_names):
    """
    Converts multi-hot labels into label names.
    """

    if torch.is_tensor(actual_labels):
        actual_labels = actual_labels.detach().cpu().numpy()

    actual_labels = np.asarray(actual_labels)

    # Multi-hot vector: [94]
    if actual_labels.ndim == 1:
        if len(actual_labels) != len(label_names):
            raise ValueError(
                "Actual label count does not match label count."
            )

        return {
            label_names[i]
            for i in range(len(label_names))
            if actual_labels[i] > 0.5
        }

    # Already a list of names
    if actual_labels.ndim == 0:
        return {str(actual_labels.item())}

    return {
        str(label)
        for label in actual_labels.tolist()
    }


# =========================
# PREDICTION
# =========================

@torch.no_grad()
def predict(model, signal, label_names, thresholds):
    signal = torch.as_tensor(
        signal,
        dtype=torch.float32
    )

    if signal.shape != (
        NUM_LEADS,
        SIGNAL_LENGTH
    ):
        raise ValueError(
            f"Expected signal shape "
            f"{(NUM_LEADS, SIGNAL_LENGTH)}, "
            f"got {tuple(signal.shape)}"
        )

    if not torch.isfinite(signal).all():
        raise ValueError(
            "ECG contains NaN or Inf values."
        )

    signal = signal.unsqueeze(0).to(DEVICE)

    logits = model(signal)

    probabilities = torch.sigmoid(logits)
    probabilities = (
        probabilities.squeeze(0)
        .cpu()
        .numpy()
    )

    predictions = probabilities >= thresholds

    predicted_labels = {
        label_names[i]: float(probabilities[i])
        for i in range(len(label_names))
        if predictions[i]
    }

    ranked_labels = sorted(
        [
            (
                label_names[i],
                float(probabilities[i])
            )
            for i in range(len(label_names))
        ],
        key=lambda item: item[1],
        reverse=True
    )

    return predicted_labels, ranked_labels


# =========================
# COMPARISON
# =========================

def compare_predictions(
    actual_labels,
    predicted_labels
):
    predicted_set = set(predicted_labels.keys())

    correct = actual_labels & predicted_set
    missed = actual_labels - predicted_set
    extra = predicted_set - actual_labels

    return correct, missed, extra


# =========================
# DISPLAY
# =========================

def display_result(
    sample_number,
    dataset_index,
    actual_labels,
    predicted_labels,
    ranked_labels
):
    correct, missed, extra = compare_predictions(
        actual_labels,
        predicted_labels
    )

    print("\n" + "=" * 65)
    print(
        f"ECG SAMPLE {sample_number} "
        f"| DATASET INDEX: {dataset_index}"
    )
    print("=" * 65)

    print("\nACTUAL DIAGNOSIS (DATASET):")

    if actual_labels:
        for label in sorted(actual_labels):
            print(f"  - {label}")
    else:
        print("  None")

    print("\nMODEL PREDICTION:")

    if predicted_labels:
        for label, probability in sorted(
            predicted_labels.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            print(
                f"  - {label}: "
                f"{probability:.4f}"
            )
    else:
        print("  No labels crossed thresholds.")

    print("\nCORRECT LABELS:")

    if correct:
        for label in sorted(correct):
            print(f"  ✓ {label}")
    else:
        print("  None")

    print("\nMISSED LABELS:")

    if missed:
        for label in sorted(missed):
            print(f"  ✗ {label}")
    else:
        print("  None")

    print("\nEXTRA PREDICTIONS:")

    if extra:
        for label in sorted(extra):
            print(f"  + {label}")
    else:
        print("  None")

    print("\nTOP 5 MODEL OUTPUTS:")

    for rank, (label, probability) in enumerate(
        ranked_labels[:5],
        start=1
    ):
        print(
            f"  {rank}. {label}: "
            f"{probability:.4f}"
        )

    print("\n" + "-" * 65)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of random ECG samples (4-10)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(DATASET_PATH)
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.count is None:
        sample_count = random.randint(4, 10)
    else:
        sample_count = args.count

    if not 4 <= sample_count <= 10:
        raise ValueError(
            "Count must be between 4 and 10."
        )

    print("Using device:", DEVICE)
    dataset_path = Path(args.csv).resolve()
    print("Dataset:", dataset_path)
    print("Model:", MODEL_PATH)
    print("Thresholds:", THRESHOLDS_PATH)
    print("Random samples:", sample_count)

    dataset = ECGDataset(args.csv)

    label_names = dataset.label_columns

    if len(label_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} labels, "
            f"found {len(label_names)}."
        )

    thresholds = np.full(
        len(label_names),
        0.65,
        dtype=np.float32
    )
    model = load_model(label_names, MODEL_PATH)

    selected_indices = random.sample(
        range(len(dataset)),
        sample_count
    )

    total_correct = 0
    total_missed = 0
    total_extra = 0

    for sample_number, index in enumerate(
        selected_indices,
        start=1
    ):
        sample = dataset[index]

        signal, actual_labels_raw = extract_sample(
            sample
        )

        actual_labels = get_actual_labels(
            actual_labels_raw,
            label_names
        )

        predicted_labels, ranked_labels = predict(
            model=model,
            signal=signal,
            label_names=label_names,
            thresholds=thresholds
        )

        correct, missed, extra = compare_predictions(
            actual_labels,
            predicted_labels
        )

        total_correct += len(correct)
        total_missed += len(missed)
        total_extra += len(extra)

        display_result(
            sample_number=sample_number,
            dataset_index=index,
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            ranked_labels=ranked_labels
        )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("Samples analysed:", sample_count)
    print("Correct labels:", total_correct)
    print("Missed labels:", total_missed)
    print("Extra predictions:", total_extra)

    print("\nNote:")
    print(
        "This is a dataset ground-truth comparison, "
        "not a clinical diagnosis."
    )


if __name__ == "__main__":
    main()