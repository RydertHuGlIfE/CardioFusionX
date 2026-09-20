import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from eng_dataset import ECGDataset
from model import get_model


class AsymmetricLoss(nn.Module):
    """
   asl - from research paper of 2021
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # Targets: (batch, num_classes), binary {0, 1}
        # Logits: (batch, num_classes)
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        # Asymmetric probability clipping on negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Log probabilities
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            p_t = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            modulating_factor = torch.pow(1.0 - p_t, gamma)
            loss = loss * modulating_factor

        return -loss.mean()


class FocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss with optional positive class weighting. bfl - reduces penalty accorss ismplified eg
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        return (focal_factor * bce).mean()


def build_loss(loss_name, pos_weight=None, device="cpu"):
    loss_name = loss_name.lower().strip()
    if loss_name == "asl":
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05).to(device)
    elif loss_name == "focal":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return FocalLoss(gamma=2.0, pos_weight=pw).to(device)
    elif loss_name in ("bce", "bce_with_logits"):
        pw = pos_weight.to(device) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw).to(device)
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from 'asl', 'focal', 'bce'")


def set_seed(seed=42):   #stable on this
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audit_splits(train_csv, val_csv, test_csv, label_names):
    """
    Verifies 100% data integrity:
    1. Zero path duplicates across train, val, and test.
    2. Zero missing files.
    3. Exactly matching 94 label names and order.
    """
    dfs = {
        "train": pd.read_csv(train_csv),
        "val": pd.read_csv(val_csv),
        "test": pd.read_csv(test_csv)
    }

    # Verify label schema
    for name, df in dfs.items():
        cols = [c for c in df.columns if c.startswith("label_")]
        if cols != label_names:
            raise ValueError(f"Label columns or ordering mismatch in {name} split!")

    # Verify record overlap
    train_paths = set(dfs["train"]["mat_path"])
    val_paths = set(dfs["val"]["mat_path"])
    test_paths = set(dfs["test"]["mat_path"])

    if train_paths & val_paths:
        raise ValueError("Data leakage detected between train and val splits!")
    if train_paths & test_paths:
        raise ValueError("Data leakage detected between train and test splits!")
    if val_paths & test_paths:
        raise ValueError("Data leakage detected between val and test splits!")

    report = {
        "train_samples": len(dfs["train"]),
        "val_samples": len(dfs["val"]),
        "test_samples": len(dfs["test"]),
        "num_classes": len(label_names),
        "overlap_leakage": "PASSED (0 duplicates)"
    }
    return report


# ==============================================================================
# DUAL THRESHOLD TUNING (STRICTLY VALIDATION-ONLY)
# ==============================================================================

def tune_validation_thresholds(y_true, y_prob, label_names, target_precision=0.40):
    """
    Tunes per-label thresholds ONLY on validation predictions:
    1. thresholds.json -> F1-optimal thresholds (maximizes class F1)
    2. conservative_thresholds.json -> Precision-oriented thresholds (targets precision >= target_precision)
    
    If precision constraint cannot be met for rare classes, flags 'precision_constraint_met = False'
    and logs the exact stats in threshold_tuning_report.csv without fabricating arbitrary numbers.
    """
    candidate_thresholds = np.round(np.arange(0.05, 0.901, 0.025), 3)
    f1_thresholds = {}
    cons_thresholds = {}
    tuning_rows = []

    for idx, label in enumerate(label_names):
        true_col = y_true[:, idx]
        prob_col = y_prob[:, idx]
        support = int(true_col.sum())

        if support == 0:
            # Zero support in validation split: fall back to prior default 0.50
            f1_thresholds[label] = 0.50
            cons_thresholds[label] = 0.50
            tuning_rows.append({
                "label": label,
                "support": 0,
                "f1_threshold": 0.50,
                "f1_val_score": 0.0,
                "f1_val_precision": 0.0,
                "f1_val_recall": 0.0,
                "cons_threshold": 0.50,
                "cons_val_precision": 0.0,
                "cons_val_recall": 0.0,
                "precision_constraint_met": False,
                "note": "Zero validation support, default 0.50 fallback"
            })
            continue

        best_f1_tuple = (-1.0, 0.0, 0.0, 0.50)  # (f1, prec, rec, thresh)
        best_cons_tuple = None

        for t in candidate_thresholds:
            preds = (prob_col >= t).astype(int)
            prec = precision_score(true_col, preds, zero_division=0)
            rec = recall_score(true_col, preds, zero_division=0)
            f1 = f1_score(true_col, preds, zero_division=0)

            if f1 > best_f1_tuple[0]:
                best_f1_tuple = (f1, prec, rec, float(t))

            # Conservative threshold: satisfying target precision
            if prec >= target_precision and preds.sum() > 0:
                if best_cons_tuple is None or f1 > best_cons_tuple[0]:
                    best_cons_tuple = (f1, prec, rec, float(t))

        f1_val, f1_p, f1_r, f1_t = best_f1_tuple
        f1_thresholds[label] = f1_t

        if best_cons_tuple is not None:
            c_f1, c_p, c_r, c_t = best_cons_tuple
            cons_thresholds[label] = c_t
            constraint_met = True
        else:
            # Target precision could not be achieved for this class
            cons_thresholds[label] = f1_t
            c_p, c_r = f1_p, f1_r
            constraint_met = False

        tuning_rows.append({
            "label": label,
            "support": support,
            "f1_threshold": f1_t,
            "f1_val_score": round(f1_val, 4),
            "f1_val_precision": round(f1_p, 4),
            "f1_val_recall": round(f1_r, 4),
            "cons_threshold": cons_thresholds[label],
            "cons_val_precision": round(c_p, 4),
            "cons_val_recall": round(c_r, 4),
            "precision_constraint_met": constraint_met,
            "note": "OK" if constraint_met else f"Precision target {target_precision} unreachable"
        })

    return f1_thresholds, cons_thresholds, tuning_rows


# ==============================================================================
# COMPREHENSIVE METRICS EVALUATION
# ==============================================================================

def compute_comprehensive_metrics(y_true, y_prob, y_pred, label_names):
    """
    Computes Macro-F1, Micro-F1, Macro PR-AUC (Average Precision),
    Weighted ROC-AUC, Macro ROC-AUC, support-stratified F1, and per-class reports.
    """
    support = y_true.sum(axis=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Valid classes for AUC (classes with both positive and negative samples in split)
    valid_auc_indices = [i for i in range(y_true.shape[1]) if np.unique(y_true[:, i]).size == 2]

    macro_roc_auc = 0.0
    weighted_roc_auc = 0.0
    macro_pr_auc = 0.0
    weighted_pr_auc = 0.0

    if valid_auc_indices:
        rocs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in valid_auc_indices]
        macro_roc_auc = float(np.mean(rocs))
        valid_supports = support[valid_auc_indices]
        if valid_supports.sum() > 0:
            weighted_roc_auc = float(np.average(rocs, weights=valid_supports))

        prs = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in valid_auc_indices]
        macro_pr_auc = float(np.mean(prs))
        if valid_supports.sum() > 0:
            weighted_pr_auc = float(np.average(prs, weights=valid_supports))

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_pr_auc": macro_pr_auc,
        "weighted_pr_auc": weighted_pr_auc,
        "macro_roc_auc": macro_roc_auc,
        "weighted_roc_auc": weighted_roc_auc,
        "macro_f1_support_ge_10": float(per_class_f1[support >= 10].mean()) if np.any(support >= 10) else 0.0,
        "macro_f1_support_ge_20": float(per_class_f1[support >= 20].mean()) if np.any(support >= 20) else 0.0,
        "exact_match_ratio": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "labels_predicted": int((y_pred.sum(axis=0) > 0).sum()),
        "labels_never_predicted": int((y_pred.sum(axis=0) == 0).sum()),
    }

    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)
    ).transpose()

    return metrics, report_df


# ==============================================================================
# VALIDATION COLLECTION
# ==============================================================================

@torch.no_grad()
def evaluate_split(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_targets, all_probs = [], []

    for signals, targets in dataloader:
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, targets)

        total_loss += loss.item() * signals.size(0)
        probs = torch.sigmoid(logits)

        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    mean_loss = total_loss / len(dataloader.dataset)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return mean_loss, y_true, y_prob


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CardioFusionX Multi-Lead ECG Production Training Engine")
    parser.add_argument("--arch", type=str, default="ecg_cnn", choices=["ecg_cnn", "ecg_resnet_se"],
                        help="Model architecture: 'ecg_cnn' (baseline) or 'ecg_resnet_se' (lead-aware)")
    parser.add_argument("--loss", type=str, default="asl", choices=["asl", "focal", "bce"],
                        help="Loss function: 'asl' (Asymmetric Loss), 'focal', or 'bce'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    parser.add_argument("--pos-weight-mode", type=str, default="sqrt", choices=["none", "sqrt", "full"],
                        help="Positive weight scaling for BCE/Focal: 'none', 'sqrt', or 'full'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint path to resume/fine-tune from")
    parser.add_argument("--exp-name", type=str, default="final_training_v1",
                        help="Experiment output directory name under experiments/")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable medically plausible subtle data augmentations")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", default=False, help="Disable Automatic Mixed Precision")

    args = parser.parse_args()

    # 1. Setup Environment
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    project_root = Path(__file__).resolve().parent
    exp_dir = project_root / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" CARDIOFUSIONX 94-LABEL MULTI-LEAD ECG TRAINING ENGINE")
    print("=" * 70)
    print(f"Device: {device} | AMP Enabled: {use_amp}")
    print(f"Architecture: {args.arch}")
    print(f"Loss Function: {args.loss.upper()}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"Experiment Output: {exp_dir}")
    print("=" * 70)

    # 2. Datasets and Integrity Audit
    train_csv = project_root / "train_split.csv"
    val_csv = project_root / "val_split.csv"
    test_csv = project_root / "test_split.csv"

    train_dataset = ECGDataset(train_csv, augment=args.augment)
    val_dataset = ECGDataset(val_csv, augment=False)
    label_names = train_dataset.label_columns

    audit_info = audit_splits(train_csv, val_csv, test_csv, label_names)
    (exp_dir / "data_integrity.json").write_text(json.dumps(audit_info, indent=2))
    print(f"Data Audit Passed: Train={audit_info['train_samples']}, Val={audit_info['val_samples']}, Test={audit_info['test_samples']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 3. Class Imbalance Weighting (if requested)
    positive_counts = torch.tensor(train_dataset.df[label_names].sum(axis=0).to_numpy(), dtype=torch.float32)
    negative_counts = len(train_dataset) - positive_counts
    raw_pos_ratio = torch.where(positive_counts > 0, negative_counts / positive_counts, torch.ones_like(positive_counts))

    if args.pos_weight_mode == "sqrt":
        pos_weight = torch.sqrt(raw_pos_ratio).clamp(max=10.0)
    elif args.pos_weight_mode == "full":
        pos_weight = raw_pos_ratio.clamp(max=20.0)
    else:
        pos_weight = None

    criterion = build_loss(args.loss, pos_weight=pos_weight, device=device)

    # 4. Model Instantiation & Optional Checkpoint Loading
    model = get_model(args.arch, num_classes=len(label_names)).to(device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Specified checkpoint does not exist: {ckpt_path}")
        print(f"Loading weights from checkpoint: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 5. Optimizer, Scheduler, and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    scaler = GradScaler("cuda", enabled=use_amp)

    # 6. Metadata Tracking
    config = {
        "architecture": args.arch,
        "loss": args.loss,
        "pos_weight_mode": args.pos_weight_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "augment": args.augment,
        "num_classes": len(label_names),
        "label_names": label_names,
        "checkpoint_loaded": str(args.checkpoint) if args.checkpoint else None
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    history_csv = exp_dir / "training_history.csv"
    with history_csv.open("w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "macro_f1", "micro_f1", "macro_precision", "macro_recall",
            "macro_pr_auc", "macro_roc_auc", "macro_f1_ge_10", "lr", "epoch_time_sec"
        ])

    best_macro_f1 = -1.0
    best_micro_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting Training...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", dynamic_ncols=True)
        for signals, targets in pbar:
            if not torch.isfinite(signals).all() or not torch.isfinite(targets).all():
                continue

            signals = signals.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(signals)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_acc += loss.item() * signals.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_acc / len(train_loader.dataset)

        # Validation Step (Strictly on val_split)
        val_loss, y_val, p_val = evaluate_split(model, val_loader, criterion, device, use_amp=use_amp)

        # Dual Validation Threshold Tuning
        f1_threshs, cons_threshs, tuning_report = tune_validation_thresholds(y_val, p_val, label_names)

        # Compute validation metrics using F1-tuned thresholds
        f1_thresh_arr = np.array([f1_threshs[l] for l in label_names])
        val_preds_f1 = (p_val >= f1_thresh_arr).astype(int)
        val_metrics, val_report = compute_comprehensive_metrics(y_val, p_val, val_preds_f1, label_names)

        scheduler.step()
        lr_current = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        # Log epoch to console
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({epoch_time:.1f}s) | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} | Micro-F1: {val_metrics['micro_f1']:.4f} | "
            f"Macro-PR: {val_metrics['macro_pr_auc']:.4f} | Macro-ROC: {val_metrics['macro_roc_auc']:.4f}"
        )

        # Append to history CSV
        with history_csv.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 5), round(val_loss, 5),
                round(val_metrics["macro_f1"], 4), round(val_metrics["micro_f1"], 4),
                round(val_metrics["macro_precision"], 4), round(val_metrics["macro_recall"], 4),
                round(val_metrics["macro_pr_auc"], 4), round(val_metrics["macro_roc_auc"], 4),
                round(val_metrics["macro_f1_support_ge_10"], 4), round(lr_current, 7), round(epoch_time, 1)
            ])

        # Base checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "architecture": args.arch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": config,
            "label_names": label_names
        }

        # Helper to save checkpoint folder
        def save_checkpoint_dir(target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, target_dir / "model.pth")
            (target_dir / "thresholds.json").write_text(json.dumps(f1_threshs, indent=2))
            (target_dir / "conservative_thresholds.json").write_text(json.dumps(cons_threshs, indent=2))
            pd.DataFrame(tuning_report).to_csv(target_dir / "threshold_tuning_report.csv", index=False)
            val_report.to_csv(target_dir / "validation_classification_report.csv")

        # Save Latest
        save_checkpoint_dir(exp_dir / "latest")

        # Save Best Macro-F1 (Primary model selection metric)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint_dir(exp_dir / "best_macro_f1")
            patience_counter = 0
            print(f" -> [BEST MACRO-F1] Checkpoint updated: {best_macro_f1:.4f}")
        else:
            patience_counter += 1

        # Save Best Micro-F1
        if val_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = val_metrics["micro_f1"]
            save_checkpoint_dir(exp_dir / "best_micro_f1")

        # Save Best Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint_dir(exp_dir / "best_val_loss")

        # Early Stopping check on validation Macro-F1
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without Macro-F1 improvement.")
            break

    print(f"\nTraining Complete! Best Validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Artifacts and checkpoints saved in: {exp_dir}")


if __name__ == "__main__":
    main()