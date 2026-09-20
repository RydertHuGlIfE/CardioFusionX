import io
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from config import MAX_UPLOAD_BYTES, NEAR_THRESHOLD_MARGIN, NUM_LEADS, SAMPLE_RATE, SIGNAL_LENGTH, label_display_name
from model_loader import load_thresholds


class InputValidationError(ValueError):
    pass


def validate_signal(signal):
    if not isinstance(signal, np.ndarray) or not np.issubdtype(signal.dtype, np.number):
        raise InputValidationError("Corrupted signal: val must be numeric.")
    if signal.shape != (NUM_LEADS, SIGNAL_LENGTH):
        raise InputValidationError(f"Incorrect ECG shape: expected ({NUM_LEADS}, {SIGNAL_LENGTH}), got {tuple(signal.shape)}.")
    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise InputValidationError("Corrupted signal: NaN and infinite values are not supported.")
    if signal.size == 0:
        raise InputValidationError("Unsupported input: the ECG signal is empty.")
    if np.max(np.abs(signal)) > 1e6:
        raise InputValidationError("Unsupported input: signal amplitude exceeds the safety limit.")
    return signal


def read_mat_upload(file_storage):
    filename = Path(file_storage.filename or "").name
    if not filename.lower().endswith(".mat"):
        raise InputValidationError("Invalid file format: only .mat files are supported.")
    payload = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InputValidationError(f"File is too large: maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not payload:
        raise InputValidationError("Corrupted file: the uploaded file is empty.")
    try:
        mat = loadmat(io.BytesIO(payload))
    except Exception as exc:
        raise InputValidationError(f"Corrupted file: could not read MATLAB data ({exc}).") from exc
    if "val" not in mat:
        raise InputValidationError("Missing val key: the MAT file must contain a variable named 'val'.")
    return validate_signal(mat["val"]), filename


def normalize_signal(signal):
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    return ((signal - mean) / std).astype(np.float32)


def _prediction_rows(probabilities, thresholds, labels, mapping):
    rows = []
    for label, probability, threshold in zip(labels, probabilities, thresholds):
        rows.append({
            "code": label,
            "name": label_display_name(label, mapping),
            "probability": round(float(probability), 6),
            "threshold": round(float(threshold), 6),
            "detected": bool(probability >= threshold),
            "near_threshold": bool(abs(float(probability) - float(threshold)) <= NEAR_THRESHOLD_MARGIN),
        })
    return rows


def prediction_payload(probabilities, thresholds, labels, mapping, model_info, threshold_strategy):
    rows = _prediction_rows(probabilities, thresholds, labels, mapping)
    ranked = sorted(rows, key=lambda row: row["probability"], reverse=True)
    detected = [row for row in ranked if row["detected"]]
    possible = [row for row in ranked if row["near_threshold"] and not row["detected"]]
    max_probability = max((row["probability"] for row in rows), default=0.0)
    entropy = float(-np.sum(np.clip(probabilities, 1e-7, 1 - 1e-7) * np.log2(np.clip(probabilities, 1e-7, 1 - 1e-7)) + (1 - np.clip(probabilities, 1e-7, 1 - 1e-7)) * np.log2(1 - np.clip(probabilities, 1e-7, 1 - 1e-7))))
    uncertainty = "low" if len(possible) <= 2 else "moderate" if len(possible) <= 8 else "high"
    return {
        "all_predictions": ranked,
        "top_predictions": ranked[:10],
        "detected": detected,
        "possible": possible[:10],
        "detected_count": len(detected),
        "max_probability": round(max_probability, 6),
        "entropy": round(entropy, 4),
        "uncertainty": uncertainty,
        "model": model_info,
        "threshold_strategy": threshold_strategy,
        "threshold_source": "fixed 0.50" if threshold_strategy == "fixed" else "validation-derived per-class file or 0.50 fallback",
        "probability_disclaimer": "Sigmoid outputs are not calibrated clinical confidence scores.",
    }


def predict_signal(model, normalized_signal, labels, thresholds, mapping, model_info, threshold_strategy):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install ecg_demo/requirements.txt to enable inference.") from exc
    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalized_signal).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor)).squeeze(0).detach().cpu().numpy()
    if probabilities.shape != (len(labels),):
        raise RuntimeError(f"Model returned {probabilities.shape} outputs; expected ({len(labels)},).")
    return prediction_payload(probabilities, thresholds, labels, mapping, model_info, threshold_strategy)


def prepare_prediction(file_storage, model_key, threshold_strategy, mapping, labels):
    raw_signal, filename = read_mat_upload(file_storage)
    from model_loader import load_model

    model, model_info = load_model(model_key, labels)
    thresholds, threshold_file = load_thresholds(labels, model_info["path"], threshold_strategy)
    result = predict_signal(model, normalize_signal(raw_signal), labels, thresholds, mapping, model_info, threshold_strategy)
    result.update({
        "filename": filename,
        "shape": list(raw_signal.shape),
        "sampling_frequency": SAMPLE_RATE,
        "threshold_file": str(threshold_file) if threshold_file else None,
    })
    return raw_signal, result, model
