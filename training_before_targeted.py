import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from eng_dataset import ECGDataset
from model import ECGCNN


SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8
GRADIENT_CLIP_MAX_NORM = 1.0
NUM_CLASSES = 94
SIGNAL_LENGTH = 5000
MIN_PRECISION_AT_LOW_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = np.round(np.arange(0.05, 0.951, 0.05), 2)
AMP_ENABLED = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "optimized_v1"


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, inputs, targets):
        weighted_bce = self.bce(inputs, targets)
        probabilities = torch.sigmoid(inputs)
        true_class_probability = probabilities * targets + (1 - probabilities) * (1 - targets)
        focal_factor = (1 - true_class_probability) ** self.gamma
        return (focal_factor * weighted_bce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def verify_label_schema(datasets):
    name, reference = datasets[0]
    for other_name, dataset in datasets[1:]:
        if dataset.label_columns != reference.label_columns:
            raise ValueError(f"Label columns differ between {name} and {other_name}.")


def inspect_split_integrity(split_paths, label_columns):
    frames = {name: pd.read_csv(path) for name, path in split_paths.items()}
    report = {"patient_id_check": "not available"}
    for name, frame in frames.items():
        if frame[label_columns].isna().any().any() or frame["mat_path"].isna().any():
            raise ValueError(f"Missing values found in {name} split.")
    paths_by_split = {name: set(frame["mat_path"].astype(str)) for name, frame in frames.items()}
    names = list(paths_by_split)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            overlap = paths_by_split[first_name] & paths_by_split[second_name]
            if overlap:
                raise ValueError(f"Duplicate ECG records across splits: {sorted(overlap)}")
    patient_columns = ["patient_id", "patient", "subject_id", "subject"]
    patient_column = next((column for column in patient_columns if all(column in frame.columns for frame in frames.values())), None)
    if patient_column:
        patients = {name: set(frame[patient_column].astype(str)) for name, frame in frames.items()}
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                overlap = patients[first_name] & patients[second_name]
                if overlap:
                    raise ValueError(f"Patient leakage across splits: {sorted(overlap)}")
        report["patient_id_check"] = f"passed: {patient_column}"
    report["duplicate_record_check"] = "passed"
    report["missing_value_check"] = "passed"
    report["split_sizes"] = {name: len(frame) for name, frame in frames.items()}
    return report


def check_batch_finite(signals, labels, split_name):
    if signals.ndim != 3 or signals.shape[1] != 12:
        raise ValueError(f"Invalid {split_name} signal shape: {signals.shape}")
    if not torch.isfinite(signals).all() or not torch.isfinite(labels).all():
        raise ValueError(f"NaN or Inf found in {split_name} batch.")


def gradient_diagnostics(model):
    nonfinite_parameters = 0
    maximum_finite_gradient = None

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        finite_values = parameter.grad.detach()[
            torch.isfinite(parameter.grad.detach())
        ]
        if finite_values.numel() != parameter.grad.numel():
            nonfinite_parameters += 1
        if finite_values.numel() > 0:
            finite_maximum = finite_values.abs().max().item()
            if (
                maximum_finite_gradient is None
                or finite_maximum > maximum_finite_gradient
            ):
                maximum_finite_gradient = finite_maximum

    return nonfinite_parameters, maximum_finite_gradient


def tune_thresholds(y_true, y_prob, label_names):
    thresholds = {}
    rows = []
    for index, label in enumerate(label_names):
        true_labels = y_true[:, index]
        probabilities = y_prob[:, index]
        support = int(true_labels.sum())
        if support == 0:
            threshold = 0.5
            predictions = (probabilities >= threshold).astype(int)
            thresholds[label] = threshold
            rows.append({"label": label, "selected_threshold": threshold, "validation_precision": 0.0, "validation_recall": 0.0, "validation_f1": 0.0, "validation_support": 0, "predicted_positive_count": int(predictions.sum())})
            continue
        best = None
        for threshold in THRESHOLD_CANDIDATES:
            predictions = (probabilities >= threshold).astype(int)
            precision = precision_score(true_labels, predictions, zero_division=0)
            if np.isclose(threshold, 0.05) and precision < MIN_PRECISION_AT_LOW_THRESHOLD:
                continue
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            candidate = (f1, precision, recall, float(threshold), int(predictions.sum()))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            best = (0.0, 0.0, 0.0, 0.5, 0)
        f1, precision, recall, threshold, predicted_count = best
        thresholds[label] = threshold
        rows.append({"label": label, "selected_threshold": threshold, "validation_precision": precision, "validation_recall": recall, "validation_f1": f1, "validation_support": support, "predicted_positive_count": predicted_count})
    return thresholds, rows


def safe_auc_metrics(y_true, y_prob):
    valid = [index for index in range(y_true.shape[1]) if np.unique(y_true[:, index]).size == 2]
    result = {"auroc_macro": None, "auroc_macro_valid_labels": len(valid), "auroc_micro": None, "auprc_macro": None, "auprc_macro_valid_labels": len(valid), "auprc_micro": None}
    if valid:
        result["auroc_macro"] = float(np.mean([roc_auc_score(y_true[:, i], y_prob[:, i]) for i in valid]))
        result["auprc_macro"] = float(np.mean([average_precision_score(y_true[:, i], y_prob[:, i]) for i in valid]))
    if np.unique(y_true).size == 2:
        result["auroc_micro"] = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
        result["auprc_micro"] = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
    return result


def classification_metrics(y_true, y_prob, y_pred, label_names):
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0)
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
        "macro_f1_support_ge_10": float(per_label_f1[support >= 10].mean()) if np.any(support >= 10) else 0.0,
        "macro_f1_support_ge_20": float(per_label_f1[support >= 20].mean()) if np.any(support >= 20) else 0.0,
    }
    metrics.update(safe_auc_metrics(y_true, y_prob))
    report = pd.DataFrame(classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)).transpose()
    return metrics, report


def collect_predictions(model, loader, criterion, split_name, scaler):
    model.eval()
    losses, labels, probabilities = [], [], []
    with torch.no_grad():
        for signals, batch_labels in tqdm(
            loader,
            desc=f"{split_name.title()} evaluation",
            dynamic_ncols=True,
            leave=True,
        ):
            check_batch_finite(signals, batch_labels, split_name)
            signals, batch_labels = signals.to(DEVICE), batch_labels.to(DEVICE)
            with autocast(
                device_type=DEVICE.type,
                enabled=scaler.is_enabled(),
            ):
                outputs = model(signals)
                loss = criterion(outputs, batch_labels)
            if not torch.isfinite(outputs).all() or not torch.isfinite(loss):
                raise ValueError(f"NaN or Inf found during {split_name} evaluation.")
            losses.append(loss.item() * signals.size(0))
            labels.append(batch_labels.cpu().numpy())
            probabilities.append(torch.sigmoid(outputs).cpu().numpy())
    return sum(losses) / len(loader.dataset), np.concatenate(labels), np.concatenate(probabilities)


def save_checkpoint(path, checkpoint, thresholds, conservative, rows, report):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path / "model.pth")
    (path / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (path / "conservative_thresholds.json").write_text(json.dumps(conservative, indent=2))
    pd.DataFrame(rows).to_csv(path / "validation_threshold_results.csv", index=False)
    report.to_csv(path / "validation_per_class_metrics.csv")


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    train_csv, val_csv = PROJECT_ROOT / "train_split.csv", PROJECT_ROOT / "val_split.csv"
    train_dataset, val_dataset = ECGDataset(train_csv), ECGDataset(val_csv)
    verify_label_schema([("train", train_dataset), ("validation", val_dataset)])
    label_names = train_dataset.label_columns
    if len(label_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} labels, found {len(label_names)}")
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "data_integrity.json").write_text(json.dumps(inspect_split_integrity({"train": train_csv, "validation": val_csv}, label_names), indent=2))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    positive_counts = torch.tensor(train_dataset.df[label_names].sum(axis=0).to_numpy(), dtype=torch.float32)
    negative_counts = len(train_dataset) - positive_counts
    pos_weight = torch.where(positive_counts > 0, negative_counts / positive_counts, torch.ones_like(positive_counts)).clamp(max=20)
    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)
    test_output = model(torch.randn(2, 12, SIGNAL_LENGTH, device=DEVICE))
    if tuple(test_output.shape) != (2, NUM_CLASSES) or not torch.isfinite(test_output).all():
        raise ValueError("Model architecture check failed.")
    architecture = str(model)
    (EXPERIMENT_ROOT / "model_architecture.txt").write_text(architecture)
    criterion = FocalLoss(FOCAL_GAMMA, pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)
    scaler = GradScaler(
        "cuda",
        enabled=AMP_ENABLED and DEVICE.type == "cuda",
    )
    best_values = {"macro_f1": -np.inf, "micro_f1": -np.inf, "macro_recall": -np.inf, "val_loss": np.inf}
    patience_count = 0
    config = {"seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "focal_gamma": FOCAL_GAMMA, "pos_weight_max": 20, "scheduler": "ReduceLROnPlateau(mode='max', factor=0.5)", "checkpoint_selection": "validation tuned macro_f1", "threshold_data": "validation only", "augmentation": "disabled to preserve ECG morphology", "experiment": "optimized_v1", "num_classes": NUM_CLASSES, "label_names": label_names, "focal_loss_formulation": "weighted_bce=BCEWithLogitsLoss(pos_weight,reduction='none'); pt=sigmoid(logit)*y+(1-sigmoid(logit))*(1-y); loss=mean((1-pt)^gamma*weighted_bce)"}
    (EXPERIMENT_ROOT / "config.json").write_text(json.dumps(config, indent=2))
    history_path = EXPERIMENT_ROOT / "training_history.csv"
    with history_path.open("w", newline="") as file:
        csv.writer(file).writerow(["epoch", "train_loss", "val_loss", "macro_f1", "micro_f1", "macro_recall", "lr", "mean_grad_norm", "clipped_batches"])
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, grad_total, clipped = 0.0, 0.0, 0
        nonfinite_gradient_batches = 0
        for batch_idx, (signals, labels) in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} Train",
            dynamic_ncols=True,
            leave=True,
        )):
            if not torch.isfinite(signals).all() or not torch.isfinite(labels).all():
                print(f"Skipping non-finite input at batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=DEVICE.type,
                enabled=AMP_ENABLED and scaler.is_enabled(),
            ):
                outputs, loss = model(signals), None
                loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print(f"Skipping non-finite loss at batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue
            if AMP_ENABLED:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            nonfinite_parameters, maximum_finite_gradient = gradient_diagnostics(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_MAX_NORM,
            )
            grad_value = float(grad_norm.item())
            if nonfinite_parameters > 0 or not np.isfinite(grad_value):
                nonfinite_gradient_batches += 1
                input_min = signals.detach().min().item()
                input_max = signals.detach().max().item()
                output_min = outputs.detach().min().item()
                output_max = outputs.detach().max().item()
                print(
                    f"Non-finite gradients | epoch={epoch} "
                    f"batch={batch_idx} | input_min={input_min} "
                    f"input_max={input_max} | output_min={output_min} "
                    f"output_max={output_max} | loss={loss.item()} | "
                    f"nonfinite_gradient_parameters={nonfinite_parameters} | "
                    f"maximum_finite_gradient={maximum_finite_gradient}"
                )
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_gradient_batches > 5:
                    raise RuntimeError(
                        f"Non-finite gradients occurred more than 5 times "
                        f"in epoch {epoch}."
                    )
                continue
            grad_total += grad_value
            clipped += int(grad_value > GRADIENT_CLIP_MAX_NORM)
            if AMP_ENABLED:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            total_loss += loss.item() * signals.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, y_val, p_val = collect_predictions(model, val_loader, criterion, "validation", scaler)
        thresholds, threshold_rows = tune_thresholds(y_val, p_val, label_names)
        threshold_array = np.array([thresholds[label] for label in label_names])
        conservative = {label: max(value, 0.70) for label, value in thresholds.items()}
        metrics, report = classification_metrics(y_val, p_val, (p_val >= threshold_array).astype(int), label_names)
        scheduler.step(metrics["macro_f1"])
        lr = optimizer.param_groups[0]["lr"]
        checkpoint = {"epoch": epoch, "completed_epochs": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "train_loss": train_loss, "val_loss": val_loss, "validation_metrics": metrics, "config": config, "model_architecture": architecture, "label_names": label_names, "learning_rate": lr, "mean_gradient_norm": grad_total / len(train_loader), "clipped_batches": clipped}
        save_checkpoint(EXPERIMENT_ROOT / "latest", checkpoint, thresholds, conservative, threshold_rows, report)
        primary_improved = metrics["macro_f1"] > best_values["macro_f1"]
        for metric_name, directory in [("macro_f1", "best_macro_f1"), ("micro_f1", "best_micro_f1"), ("macro_recall", "best_macro_recall"), ("val_loss", "best_val_loss")]:
            value = val_loss if metric_name == "val_loss" else metrics[metric_name]
            improved = value < best_values[metric_name] if metric_name == "val_loss" else value > best_values[metric_name]
            if improved:
                best_values[metric_name] = value
                selected = dict(checkpoint)
                selected["selection_metric"], selected["selection_value"] = metric_name, value
                save_checkpoint(EXPERIMENT_ROOT / directory, selected, thresholds, conservative, threshold_rows, report)
        with history_path.open("a", newline="") as file:
            csv.writer(file).writerow([epoch, train_loss, val_loss, metrics["macro_f1"], metrics["micro_f1"], metrics["macro_recall"], lr, grad_total / len(train_loader), clipped])
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} macro_recall={metrics['macro_recall']:.4f}")
        patience_count = 0 if primary_improved else patience_count + 1
        if patience_count >= EARLY_STOPPING_PATIENCE:
            print("Early stopping based on validation Macro-F1.")
            break
    print(f"Training complete. Artifacts saved under {EXPERIMENT_ROOT}")


if __name__ == "__main__":
    main()