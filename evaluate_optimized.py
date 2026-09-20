import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from eng_dataset import ECGDataset
from model import ECGCNN


BATCH_SIZE = 8
NUM_CLASSES = 94
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "optimized_v1"


def verify_label_schema(datasets):
    name, reference = datasets[0]
    for other_name, dataset in datasets[1:]:
        if dataset.label_columns != reference.label_columns:
            raise ValueError(f"Label columns differ between {name} and {other_name}.")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid optimized_v1 checkpoint: {path}")
    config = checkpoint.get("config", {})
    if config.get("experiment") != "optimized_v1":
        raise ValueError(f"Checkpoint is not from optimized_v1: {path}")
    if config.get("num_classes") != NUM_CLASSES:
        raise ValueError(f"Checkpoint class count is not {NUM_CLASSES}: {path}")
    return checkpoint


def load_thresholds(path, label_names):
    data = json.loads(Path(path).read_text())
    missing = [label for label in label_names if label not in data]
    if missing:
        raise ValueError(f"Missing thresholds for labels: {missing}")
    thresholds = np.array([float(data[label]) for label in label_names], dtype=np.float32)
    if thresholds.shape != (len(label_names),) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError(f"Invalid thresholds in {path}")
    return thresholds


def safe_auc(y_true, y_prob):
    valid = [i for i in range(y_true.shape[1]) if np.unique(y_true[:, i]).size == 2]
    metrics = {"auroc_macro": None, "auroc_macro_valid_labels": len(valid), "auroc_micro": None, "auprc_macro": None, "auprc_macro_valid_labels": len(valid), "auprc_micro": None}
    if valid:
        metrics["auroc_macro"] = float(np.mean([roc_auc_score(y_true[:, i], y_prob[:, i]) for i in valid]))
        metrics["auprc_macro"] = float(np.mean([average_precision_score(y_true[:, i], y_prob[:, i]) for i in valid]))
    if np.unique(y_true).size == 2:
        metrics["auroc_micro"] = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
        metrics["auprc_micro"] = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    return metrics


def evaluate(checkpoint_path, output_dir, threshold_mode, thresholds):
    checkpoint = load_checkpoint(checkpoint_path)
    label_names = checkpoint["label_names"]
    test_dataset = ECGDataset(PROJECT_ROOT / "test_split.csv")
    if test_dataset.label_columns != label_names:
        raise ValueError("Test label order does not match checkpoint label order.")
    if len(label_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} labels, found {len(label_names)}")

    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels, probabilities = [], []
    with torch.no_grad():
        for signals, batch_labels in loader:
            if not torch.isfinite(signals).all() or not torch.isfinite(batch_labels).all():
                raise ValueError("NaN or Inf found in test batch.")
            outputs = model(signals.to(DEVICE))
            if not torch.isfinite(outputs).all():
                raise ValueError("NaN or Inf found in test outputs.")
            labels.append(batch_labels.numpy())
            probabilities.append(torch.sigmoid(outputs).cpu().numpy())

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    y_pred = (y_prob >= thresholds).astype(int)
    per_label = pd.DataFrame(classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)).transpose()
    per_label = per_label.loc[label_names]
    support = y_true.sum(axis=0)
    label_f1 = per_label["f1-score"].to_numpy()
    metrics = {
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_ratio": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "false_positive_count": int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negative_count": int(((y_pred == 0) & (y_true == 1)).sum()),
        "labels_predicted": int((y_pred.sum(axis=0) > 0).sum()),
        "labels_never_predicted": int((y_pred.sum(axis=0) == 0).sum()),
        "macro_f1_support_ge_10": float(label_f1[support >= 10].mean()) if np.any(support >= 10) else 0.0,
        "macro_f1_support_ge_20": float(label_f1[support >= 20].mean()) if np.any(support >= 20) else 0.0,
        "mean_brier_score": float(np.mean((y_prob - y_true) ** 2)),
    }
    metrics.update(safe_auc(y_true, y_prob))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_label.insert(0, "label", label_names)
    per_label.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_json(output_dir / "aggregate_metrics.json", orient="records", indent=2)
    pd.DataFrame(y_prob, columns=label_names).to_csv(output_dir / "prediction_probabilities.csv", index=False)
    pd.DataFrame(y_pred, columns=label_names).to_csv(output_dir / "predictions.csv", index=False)
    np.save(output_dir / "confusion_matrices.npy", np.array([
        [[int(((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()), int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())],
         [int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()), int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())]]
        for i in range(len(label_names))
    ]))
    metadata = {
        "model_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold_mode": threshold_mode,
        "number_of_test_samples": len(test_dataset),
        "number_of_labels": len(label_names),
        "test_split_used_only_for_final_evaluation": True,
    }
    (output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n{threshold_mode}: {checkpoint_path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved: {output_dir}")


def main():
    os.chdir(PROJECT_ROOT)
    print(f"Using device: {DEVICE}")
    test_dataset = ECGDataset(PROJECT_ROOT / "test_split.csv")
    train_dataset = ECGDataset(PROJECT_ROOT / "train_split.csv")
    val_dataset = ECGDataset(PROJECT_ROOT / "val_split.csv")
    verify_label_schema([("train", train_dataset), ("validation", val_dataset), ("test", test_dataset)])
    best_dir = EXPERIMENT_ROOT / "best_macro_f1"
    checkpoint = best_dir / "model.pth"
    threshold_path = best_dir / "thresholds.json"
    conservative_path = best_dir / "conservative_thresholds.json"
    if not checkpoint.exists():
        print(f"WARNING: missing optimized_v1 checkpoint: {checkpoint}")
        return
    evaluate(checkpoint, EXPERIMENT_ROOT / "evaluation" / "best_macro_f1_fixed", "fixed (0.5)", np.full(NUM_CLASSES, 0.5, dtype=np.float32))
    if threshold_path.exists():
        evaluate(checkpoint, EXPERIMENT_ROOT / "evaluation" / "best_macro_f1_tuned", "validation-tuned", load_thresholds(threshold_path, test_dataset.label_columns))
    else:
        print(f"WARNING: missing validation thresholds: {threshold_path}")
    if conservative_path.exists():
        evaluate(checkpoint, EXPERIMENT_ROOT / "evaluation" / "best_macro_f1_conservative", "conservative validation thresholds", load_thresholds(conservative_path, test_dataset.label_columns))
    else:
        print(f"WARNING: missing conservative thresholds: {conservative_path}")


if __name__ == "__main__":
    main()
