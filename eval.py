import torch
import numpy as np
import pandas as pd
import json

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt

from eng_dataset import ECGDataset
from model import ECGCNN


BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

test_dataset = ECGDataset("test_split.csv")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

label_names = test_dataset.label_columns


def load_thresholds(thresholds_path):
    with Path(thresholds_path).open() as thresholds_file:
        threshold_data = json.load(thresholds_file)

    missing_labels = [
        label for label in label_names if label not in threshold_data
    ]
    if missing_labels:
        raise ValueError(
            f"Missing thresholds for labels: {missing_labels}"
        )

    thresholds = [
        float(threshold_data[label])
        for label in label_names
    ]

    if len(thresholds) != len(label_names):
        raise ValueError("Threshold count does not match label count.")

    if not all(0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("All thresholds must be between 0 and 1.")

    return thresholds


def load_model(model_path):
    model = ECGCNN(num_classes=94).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def save_confusion_matrix_image(matrix, label, output_path):
    tn, fp, fn, tp = matrix.ravel()

    image_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(image_matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {label}")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(image_matrix[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_combined_confusion_matrix(
    confusion_matrices,
    label_names,
    y_true,
    output_dir
):
    support = y_true.sum(axis=0)
    top_count = min(20, len(label_names))
    top_indices = np.argsort(support)[-top_count:][::-1]

    rows = int(np.ceil(top_count / 5))

    fig, axes = plt.subplots(
        rows,
        5,
        figsize=(18, 4 * rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for position, index in enumerate(top_indices):
        tn, fp, fn, tp = confusion_matrices[index].ravel()

        matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])

        axes[position].imshow(matrix)
        axes[position].set_xticks([0, 1])
        axes[position].set_yticks([0, 1])
        axes[position].set_xticklabels(["N", "P"])
        axes[position].set_yticklabels(["N", "P"])
        axes[position].set_title(label_names[index], fontsize=8)

        for row in range(2):
            for col in range(2):
                axes[position].text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=8
                )

    for axis in axes[top_count:]:
        axis.axis("off")

    fig.suptitle("Top Diagnosis Confusion Matrices", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrices_top20.png",
        dpi=200
    )
    plt.close(fig)


def evaluate_model(model_path, output_dir, thresholds_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "confusion_matrix_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if thresholds_path is None:
        thresholds = [0.5] * len(label_names)
        threshold_mode = "fixed (0.5)"
    else:
        thresholds = load_thresholds(thresholds_path)
        threshold_mode = f"tuned ({thresholds_path})"

    print(f"\nEvaluating: {model_path}")
    print(f"Threshold mode: {threshold_mode}")

    model = load_model(model_path)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(DEVICE)

            outputs = model(signals)
            probabilities = torch.sigmoid(outputs)

            threshold_tensor = torch.tensor(
                thresholds,
                dtype=probabilities.dtype,
                device=probabilities.device
            )

            predictions = (
                probabilities >= threshold_tensor
            ).int().cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels.numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)

    metrics = {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "label_accuracy": np.mean(y_true == y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        )
    }

    pd.DataFrame([metrics]).to_json(
        output_dir / "test_metrics.json",
        orient="records",
        indent=4
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    confusion_matrices = multilabel_confusion_matrix(
        y_true,
        y_pred
    )

    np.save(
        output_dir / "confusion_matrices.npy",
        confusion_matrices
    )

    rows = []

    for i, label in enumerate(label_names):
        tn, fp, fn, tp = confusion_matrices[i].ravel()

        rows.append({
            "label": label,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "confusion_matrices.csv",
        index=False
    )

    support = y_true.sum(axis=0)
    top_indices = np.argsort(support)[-20:][::-1]
    individual_indices = top_indices[:10]

    for i in individual_indices:
        safe_label = (
            label_names[i]
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        save_confusion_matrix_image(
            confusion_matrices[i],
            label_names[i],
            image_dir / f"{i:03d}_{safe_label}.png"
        )

    save_combined_confusion_matrix(
        confusion_matrices,
        label_names,
        y_true,
        output_dir
    )

    pd.DataFrame(
        y_prob,
        columns=label_names
    ).to_csv(
        output_dir / "prediction_probabilities.csv",
        index=False
    )

    print("\nMetrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Saved:", output_dir)
    print("Individual confusion matrices:", image_dir)


evaluate_model(
    "experiments/latest/model.pth",
    "experiments/evaluation/latest_fixed",
    thresholds_path=None
)

evaluate_model(
    "experiments/best/model.pth",
    "experiments/evaluation/best_tuned",
    thresholds_path="experiments/best/thresholds.json"
)

evaluate_model(
    "experiments/best/model.pth",
    "experiments/evaluation/best_fixed",
    thresholds_path=None
)

print("\nBoth models evaluated successfully!")
