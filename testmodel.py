import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from eng_dataset import ECGDataset
from model import get_model, ECGCNN


NUM_CLASSES = 94
NUM_LEADS = 12
SIGNAL_LENGTH = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent


def load_diagnosis_mapping():
    """Maps SNOMED diagnosis codes and label keys to human-readable names."""
    mapping = {}
    csv_path = PROJECT_ROOT / "diagnosis_code_mapping.csv"
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                code = str(row.get("diagnosis_code", "")).strip()
                name = str(row.get("diagnosis_name", "")).strip()
                if code and name:
                    mapping[code] = name
                    mapping[f"label_{code}"] = name
        except Exception:
            pass
    return mapping


def get_label_display_name(label_key, mapping):
    if label_key in {None, ""}:
        return "Unknown - Verify"
    clean_code = str(label_key).replace("label_", "")
    if label_key in mapping:
        return mapping[label_key]
    if clean_code in mapping:
        return mapping[clean_code]
    return "Unknown - Verify"


def resolve_default_model():
    candidates = [
        PROJECT_ROOT / "/run/media/Ryder/Coding/Coding/CardioFusionX/models/ECG_model/model1.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[1]


def resolve_threshold_file(model_path, mode="tuned", custom_thresh=None):
    if custom_thresh and Path(custom_thresh).is_file():
        return Path(custom_thresh)

    model_dir = Path(model_path).parent
    filename = "thresholds.json" if mode == "tuned" else "conservative_thresholds.json"
    candidate = model_dir / filename
    if candidate.is_file():
        return candidate

    # Fallback to model_dir's thresholds.json or global fallback
    if (model_dir / "thresholds.json").is_file():
        return model_dir / "thresholds.json"
    return candidate


def load_thresholds(label_names, thresholds_path, default_val=0.50):
    p = Path(thresholds_path)
    if not p.is_file():
        print(f"Warning: Threshold file not found ({p}). Using default {default_val} across all labels.")
        return np.full(len(label_names), default_val, dtype=np.float32)

    with p.open() as f:
        data = json.load(f)

    return np.array([float(data.get(l, default_val)) for l in label_names], dtype=np.float32)


def load_model(model_path, label_names):
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {p}")

    ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("architecture", ckpt.get("config", {}).get("architecture", "ecg_cnn"))

    try:
        model = get_model(arch, num_classes=len(label_names))
        model.load_state_dict(state_dict)
    except Exception:
        model = ECGCNN(num_classes=len(label_names))
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"Loaded checkpoint: {p.name} (Epoch: {epoch}, Architecture: {arch})")
    return model


@torch.no_grad()
def predict_sample(model, signal, label_names, thresholds):
    if not isinstance(signal, torch.Tensor):
        signal = torch.as_tensor(signal, dtype=torch.float32)

    if signal.ndim == 2:
        signal = signal.unsqueeze(0)

    signal = signal.to(DEVICE)
    logits = model(signal)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Keep label, probability, and threshold aligned for every model output.
    class_count = min(len(label_names), len(probs), len(thresholds))
    label_names = label_names[:class_count]
    thresholds = thresholds[:class_count]

    # Detection is threshold-based; ranking is independent of detection status.
    preds = probs[:class_count] >= thresholds

    active_predictions = [
        (label_names[i], float(probs[i]), float(thresholds[i]))
        for i in range(class_count)
        if preds[i]
    ]

    ranked_predictions = sorted(
        [
            (label_names[i], float(probs[i]), float(thresholds[i]))
            for i in range(class_count)
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return active_predictions, ranked_predictions


def display_sample(sample_num, dataset_idx, actual_labels, active_preds, ranked_preds, mapping):
    top10_preds = ranked_preds[:10]
    top10_labels = {item[0] for item in top10_preds}
    pred_labels = {item[0] for item in active_preds}
    correct = actual_labels & pred_labels
    missed = actual_labels - pred_labels
    extra = pred_labels - actual_labels

    print("\n" + "=" * 75)
    print(f" ECG SAMPLE {sample_num} | DATASET RECORD INDEX: {dataset_idx}")
    print("=" * 75)

    print("\n[GROUND TRUTH CLINICAL DIAGNOSES]")
    if actual_labels:
        for l in sorted(actual_labels):
            name = get_label_display_name(l, mapping)
            top10_marker = "YES" if l in top10_labels else "NO"
            print(f"  * {l} -> {name} (Top 10: {top10_marker})")
    else:
        print("  * None (Normal / Unlabeled)")

    print("\n[ACTIVE MODEL PREDICTIONS (Threshold Crossed)]")
    if active_preds:
        for l, prob, thresh in sorted(active_preds, key=lambda x: x[1], reverse=True):
            name = get_label_display_name(l, mapping)
            mark = "✓" if l in actual_labels else "+"
            print(f"  {mark} {l} -> {name:<32} (Prob: {prob:.4f} | Thresh: {thresh:.2f})")
    else:
        print("  No labels crossed the selected thresholds.")

    print("\n[TOP 10 DISEASE PREDICTIONS]")
    if top10_preds:
        rows = [
            (rank, label, get_label_display_name(label, mapping), prob, thresh,
             "✓ DETECTED" if prob >= thresh else "✗ NOT DETECTED")
            for rank, (label, prob, thresh) in enumerate(top10_preds, start=1)
        ]
        label_width = max(len("Label"), *(len(row[1]) for row in rows))
        name_width = max(len("Disease Name"), *(len(row[2]) for row in rows))
        print(
            f"  {'Rank':>4} | {'Label':<{label_width}} | "
            f"{'Disease Name':<{name_width}} | {'Probability':>11} | "
            f"{'Threshold':>9} | Status"
        )
        print("  " + "-" * (4 + label_width + name_width + 11 + 9 + 22))
        for rank, label, name, prob, thresh, status in rows:
            print(
                f"  {rank:>4} | {label:<{label_width}} | {name:<{name_width}} | "
                f"{prob * 100:>10.2f}% | {thresh:>9.2f} | {status}"
            )
    else:
        print("  No model outputs were available.")

    detected_top10 = sum(prob >= thresh for _, _, _, prob, thresh, _ in rows) if top10_preds else 0
    print("\n[TOP-10 SUMMARY]")
    print(f"Top-10 diseases displayed: {len(top10_preds)}")
    print(f"Detected among Top-10: {detected_top10}")
    print(f"Not detected among Top-10: {len(top10_preds) - detected_top10}")

    print("\n[COMPARISON SUMMARY]")
    print(f"  Matched:  {len(correct)} {[get_label_display_name(x, mapping) for x in correct]}")
    print(f"  Missed:   {len(missed)} {[get_label_display_name(x, mapping) for x in missed]}")
    print(f"  Extra:    {len(extra)} {[get_label_display_name(x, mapping) for x in extra]}")

    print("\n[ALL DETECTED DISEASES]")
    if active_preds:
        for l, prob, thresh in sorted(active_preds, key=lambda x: x[1], reverse=True):
            name = get_label_display_name(l, mapping)
            print(f"  ✓ {l} -> {name} (Prob: {prob:.4f} | Thresh: {thresh:.2f})")
    else:
        print("  No labels crossed the selected thresholds.")

    print("-" * 75)

    return len(correct), len(missed), len(extra)


def main():
    parser = argparse.ArgumentParser(description="CardioFusionX Interactive Sample Inspector")
    parser.add_argument("--count", type=int, default=5, help="Number of random ECG samples to inspect (1 to 20)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--csv", type=str, default="test_split.csv", help="Dataset CSV path")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--thresholds", type=str, default=None, help="Custom path to thresholds JSON")
    parser.add_argument("--threshold-mode", type=str, default="tuned", choices=["tuned", "conservative"],
                        help="Threshold strategy: 'tuned' (F1-oriented) or 'conservative' (precision-oriented)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    model_path = Path(args.model) if args.model else resolve_default_model()
    thresh_path = resolve_threshold_file(model_path, mode=args.threshold_mode, custom_thresh=args.thresholds)
    mapping = load_diagnosis_mapping()

    print("=" * 75)
    print(" CARDIOFUSIONX CLINICAL SAMPLE INSPECTOR")
    print("=" * 75)
    print(f"Model Checkpoint:    {model_path}")
    print(f"Threshold Strategy:  {args.threshold_mode.upper()} ({thresh_path.name})")
    print(f"Evaluation Dataset:  {args.csv}")
    print(f"Samples to inspect:  {args.count}")
    print("=" * 75)

    dataset = ECGDataset(args.csv, augment=False)
    label_names = dataset.label_columns
    thresholds = load_thresholds(label_names, thresh_path)
    model = load_model(model_path, label_names)

    selected_indices = random.sample(range(len(dataset)), min(args.count, len(dataset)))

    total_c, total_m, total_e = 0, 0, 0
    for idx, sample_idx in enumerate(selected_indices, start=1):
        signal, raw_labels = dataset[sample_idx]
        actual_labels = {
            label_names[i]
            for i in range(len(label_names))
            if raw_labels[i] > 0.5
        }

        active_preds, ranked_preds = predict_sample(model, signal, label_names, thresholds)
        c, m, e = display_sample(idx, sample_idx, actual_labels, active_preds, ranked_preds, mapping)
        total_c += c
        total_m += m
        total_e += e

    print("\n" + "=" * 75)
    print(" INSPECTION SUMMARY")
    print("=" * 75)
    print(f"Total Samples Inspected:  {args.count}")
    print(f"Total Correct Diagnoses:  {total_c}")
    print(f"Total Missed Diagnoses:   {total_m}")
    print(f"Total Extra Predictions:  {total_e}")
    print("=" * 75)


if __name__ == "__main__":
    main()