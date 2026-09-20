import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from eng_dataset import ECGDataset
from model import get_model, ECGCNN


def load_diagnosis_mapping(csv_path):
    """Loads mapping from diagnosis code to human-readable condition name."""
    mapping = {}
    p = Path(csv_path)
    if p.is_file():
        try:
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                code = str(row.get("diagnosis_code", "")).strip()
                name = str(row.get("diagnosis_name", "")).strip()
                if code and name:
                    mapping[code] = name
                    mapping[f"label_{code}"] = name
        except Exception:
            pass
    return mapping


def load_checkpoint_and_model(model_path, num_classes, device):
    """Loads model with architecture detection from checkpoint metadata."""
    ckpt_path = Path(model_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("architecture", ckpt.get("config", {}).get("architecture", "ecg_cnn"))

    try:
        model = get_model(arch, num_classes=num_classes)
        model.load_state_dict(state_dict)
    except Exception:
        # Fallback to ECGCNN
        model = ECGCNN(num_classes=num_classes)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, ckpt


def load_threshold_array(thresholds_path, label_names, fallback=0.50):
    p = Path(thresholds_path)
    if not p.is_file():
        return np.full(len(label_names), fallback, dtype=np.float32)
    with p.open() as f:
        data = json.load(f)
    return np.array([float(data.get(l, fallback)) for l in label_names], dtype=np.float32)


@torch.no_grad()
def collect_probabilities(model, dataloader, device, limit=None):
    model.eval()
    all_targets, all_probs = [], []
    count = 0

    for signals, targets in tqdm(dataloader, desc="Collecting Test Probabilities", dynamic_ncols=True):
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        probs = torch.sigmoid(logits)

        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())
        count += signals.size(0)
        if limit and count >= limit:
            break

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    if limit:
        y_true = y_true[:limit]
        y_prob = y_prob[:limit]
    return y_true, y_prob


def calculate_metrics(y_true, y_prob, thresholds, label_names):
    y_pred = (y_prob >= thresholds).astype(int)
    support = y_true.sum(axis=0)

    # Valid indices for AUC
    valid_auc = [i for i in range(y_true.shape[1]) if np.unique(y_true[:, i]).size == 2]

    macro_roc = 0.0
    weighted_roc = 0.0
    macro_pr = 0.0
    weighted_pr = 0.0

    if valid_auc:
        rocs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in valid_auc]
        macro_roc = float(np.mean(rocs))
        valid_sup = support[valid_auc]
        if valid_sup.sum() > 0:
            weighted_roc = float(np.average(rocs, weights=valid_sup))

        prs = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in valid_auc]
        macro_pr = float(np.mean(prs))
        if valid_sup.sum() > 0:
            weighted_pr = float(np.average(prs, weights=valid_sup))

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)

    res = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_pr_auc": macro_pr,
        "weighted_pr_auc": weighted_pr,
        "macro_roc_auc": macro_roc,
        "weighted_roc_auc": weighted_roc,
        "macro_f1_support_ge_10": float(per_f1[support >= 10].mean()) if np.any(support >= 10) else 0.0,
        "macro_f1_support_ge_20": float(per_f1[support >= 20].mean()) if np.any(support >= 20) else 0.0,
        "labels_predicted": int((y_pred.sum(axis=0) > 0).sum()),
        "labels_never_predicted": int((y_pred.sum(axis=0) == 0).sum()),
        "per_class_f1": per_f1,
        "per_class_recall": per_rec,
        "per_class_precision": per_prec,
        "support": support
    }
    return res


def print_comparison_table(metrics_orig, metrics_new, title="EVALUATION COMPARISON"):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Original':<12} | {'New Model':<12} | {'Difference':<12}")
    print("-" * 80)

    display_keys = [
        ("Macro-F1", "macro_f1"),
        ("Micro-F1", "micro_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Micro Precision", "micro_precision"),
        ("Micro Recall", "micro_recall"),
        ("Macro PR-AUC", "macro_pr_auc"),
        ("Weighted PR-AUC", "weighted_pr_auc"),
        ("Macro ROC-AUC", "macro_roc_auc"),
        ("Weighted ROC-AUC", "weighted_roc_auc"),
        ("Macro-F1 (Supp >= 10)", "macro_f1_support_ge_10"),
        ("Macro-F1 (Supp >= 20)", "macro_f1_support_ge_20"),
        ("Labels Predicted", "labels_predicted"),
        ("Labels Unpredicted", "labels_never_predicted"),
    ]

    for label, key in display_keys:
        v_orig = metrics_orig[key]
        v_new = metrics_new[key]
        diff = v_new - v_orig
        if isinstance(v_orig, int):
            print(f"{label:<25} | {v_orig:<12} | {v_new:<12} | {diff:+d}")
        else:
            print(f"{label:<25} | {v_orig:<12.4f} | {v_new:<12.4f} | {diff:+.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Full Test-Set Evaluation and Comparative Benchmark")
    parser.add_argument("--new-model", type=str,
                        default="experiments/targeted_finetune_v1/best_macro_f1/model.pth",
                        help="Path to new model checkpoint")
    parser.add_argument("--new-thresh", type=str,
                        default="experiments/targeted_finetune_v1/best_macro_f1/thresholds.json",
                        help="Path to new model tuned thresholds")
    parser.add_argument("--new-cons-thresh", type=str,
                        default="experiments/targeted_finetune_v1/best_macro_f1/conservative_thresholds.json",
                        help="Path to new model conservative thresholds")
    parser.add_argument("--orig-model", type=str,
                        default="experiments/optimized_v1/best_macro_f1/model.pth",
                        help="Path to original baseline checkpoint")
    parser.add_argument("--orig-thresh", type=str,
                        default="experiments/optimized_v1/best_macro_f1/thresholds.json",
                        help="Path to original baseline tuned thresholds")
    parser.add_argument("--orig-cons-thresh", type=str,
                        default="experiments/optimized_v1/best_macro_f1/conservative_thresholds.json",
                        help="Path to original baseline conservative thresholds")
    parser.add_argument("--test-csv", type=str, default="test_split.csv", help="Test split CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--output-csv", type=str, default="per_class_comparison.csv",
                        help="Output path for per-class comparison CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on samples (e.g. 500 for fast verification)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    dataset = ECGDataset(args.test_csv, augment=False)
    label_names = dataset.label_columns
    num_classes = len(label_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    diag_map = load_diagnosis_mapping("diagnosis_code_mapping.csv")

    # 1. Load Original Model & Compute Probabilities
    print(f"\n[1/2] Loading Original Baseline: {args.orig_model}")
    orig_model, _ = load_checkpoint_and_model(args.orig_model, num_classes, device)
    y_true, orig_probs = collect_probabilities(orig_model, loader, device, limit=args.limit)

    # 2. Load New Model & Compute Probabilities
    print(f"\n[2/2] Loading New Model: {args.new_model}")
    new_model, _ = load_checkpoint_and_model(args.new_model, num_classes, device)
    _, new_probs = collect_probabilities(new_model, loader, device, limit=args.limit)

    # 3. Evaluate Under F1-Oriented Tuned Thresholds
    orig_f1_t = load_threshold_array(args.orig_thresh, label_names)
    new_f1_t = load_threshold_array(args.new_thresh, label_names)

    orig_metrics_f1 = calculate_metrics(y_true, orig_probs, orig_f1_t, label_names)
    new_metrics_f1 = calculate_metrics(y_true, new_probs, new_f1_t, label_names)

    print_comparison_table(
        orig_metrics_f1, new_metrics_f1,
        title="BENCHMARK 1: F1-ORIENTED VALIDATION-TUNED THRESHOLDS"
    )

    # 4. Evaluate Under Precision-Oriented Conservative Thresholds
    orig_cons_t = load_threshold_array(args.orig_cons_thresh, label_names)
    new_cons_t = load_threshold_array(args.new_cons_thresh, label_names)

    orig_metrics_cons = calculate_metrics(y_true, orig_probs, orig_cons_t, label_names)
    new_metrics_cons = calculate_metrics(y_true, new_probs, new_cons_t, label_names)

    print_comparison_table(
        orig_metrics_cons, new_metrics_cons,
        title="BENCHMARK 2: PRECISION-ORIENTED CONSERVATIVE THRESHOLDS"
    )

    # 5. Export Per-Class Comparison CSV (Based on F1-tuned thresholds)
    class_rows = []
    for idx, label in enumerate(label_names):
        code = label.replace("label_", "")
        name = diag_map.get(label, diag_map.get(code, "Unknown Condition"))
        sup = int(orig_metrics_f1["support"][idx])

        o_f1 = orig_metrics_f1["per_class_f1"][idx]
        n_f1 = new_metrics_f1["per_class_f1"][idx]
        d_f1 = n_f1 - o_f1

        o_r = orig_metrics_f1["per_class_recall"][idx]
        n_r = new_metrics_f1["per_class_recall"][idx]
        d_r = n_r - o_r

        o_p = orig_metrics_f1["per_class_precision"][idx]
        n_p = new_metrics_f1["per_class_precision"][idx]
        d_p = n_p - o_p

        if d_f1 > 0.02:
            status = "IMPROVED"
        elif d_f1 < -0.02:
            status = "DEGRADED"
        else:
            status = "MAINTAINED"

        class_rows.append({
            "label": label,
            "diagnosis_code": code,
            "diagnosis_name": name,
            "support": sup,
            "original_f1": round(o_f1, 4),
            "new_f1": round(n_f1, 4),
            "delta_f1": round(d_f1, 4),
            "original_recall": round(o_r, 4),
            "new_recall": round(n_r, 4),
            "delta_recall": round(d_r, 4),
            "original_precision": round(o_p, 4),
            "new_precision": round(n_p, 4),
            "delta_precision": round(d_p, 4),
            "status": status
        })

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(args.output_csv, index=False)
    print(f"\nPer-Class Comparison Table successfully exported to: {args.output_csv}")

    improved_count = (class_df["status"] == "IMPROVED").sum()
    degraded_count = (class_df["status"] == "DEGRADED").sum()
    maintained_count = (class_df["status"] == "MAINTAINED").sum()

    print(f"Class Summary -> Improved: {improved_count} | Maintained: {maintained_count} | Degraded: {degraded_count}")


if __name__ == "__main__":
    main()
