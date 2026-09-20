import json
import importlib.util
from pathlib import Path

import numpy as np

from config import DEFAULT_THRESHOLD, MODEL_CATALOG, NUM_CLASSES, canonical_labels


def _get_model_factory():
    model_path = Path(__file__).resolve().parents[1] / "model.py"
    spec = importlib.util.spec_from_file_location("cardiofusionx_root_model", model_path)
    if spec is None or spec.loader is None:
        raise ModelConfigurationError(f"Root model definition not found: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model


class ModelConfigurationError(RuntimeError):
    pass


def threshold_path(model_path, strategy):
    model_dir = Path(model_path).parent
    if strategy in {"tuned", "f1"}:
        candidates = [model_dir / "thresholds.json"]
    elif strategy == "conservative":
        candidates = [model_dir / "conservative_thresholds.json", model_dir / "thresholds.json"]
    else:
        candidates = []
    return next((path for path in candidates if path.is_file()), None)


def load_thresholds(label_names, model_path, strategy):
    if strategy == "fixed":
        return np.full(len(label_names), DEFAULT_THRESHOLD, dtype=np.float32), None
    path = threshold_path(model_path, strategy)
    if path is None:
        return np.full(len(label_names), DEFAULT_THRESHOLD, dtype=np.float32), None
    try:
        data = json.loads(path.read_text())
        values = np.array([float(data.get(label, DEFAULT_THRESHOLD)) for label in label_names], dtype=np.float32)
        if values.shape != (len(label_names),) or not np.isfinite(values).all() or not ((0 <= values).all() and (values <= 1).all()):
            raise ValueError("threshold values must be finite numbers between 0 and 1")
        return values, path
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Could not load thresholds from {path}: {exc}") from exc


def _checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}, checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) or {}
    metadata = {
        "architecture": checkpoint.get("architecture", config.get("architecture")),
        "num_classes": checkpoint.get("num_classes", config.get("num_classes")),
        "label_names": checkpoint.get("label_names", config.get("label_names")),
        "epoch": checkpoint.get("epoch", "unknown"),
    }
    return metadata, state_dict


def _validate_state_dict(model, state_dict, architecture, label_names):
    if not isinstance(state_dict, dict):
        raise ModelConfigurationError("Checkpoint does not contain a usable model_state_dict.")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    shape_errors = [
        key for key in expected.keys() & state_dict.keys()
        if tuple(expected[key].shape) != tuple(state_dict[key].shape)
    ]
    if missing or unexpected or shape_errors:
        details = []
        if missing:
            details.append(f"missing keys: {len(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {len(unexpected)}")
        if shape_errors:
            details.append(f"shape mismatches: {', '.join(shape_errors[:3])}")
        raise ModelConfigurationError(
            f"Checkpoint does not match architecture '{architecture}' ({'; '.join(details)})."
        )
    output_shape = tuple(expected["classifier.1.weight"].shape)
    if output_shape[0] != len(label_names):
        raise ModelConfigurationError(
            f"Checkpoint output has {output_shape[0]} classes but the canonical label list has {len(label_names)}."
        )


def load_model(model_key, label_names):
    """Load and validate a selected checkpoint; PyTorch is intentionally lazy."""
    if model_key not in MODEL_CATALOG:
        raise ModelConfigurationError(f"Unknown model selection: {model_key}")
    spec = MODEL_CATALOG[model_key]
    model_path = Path(spec["path"])
    if not model_path.is_file():
        raise ModelConfigurationError(f"Checkpoint is missing: {model_path}")
    if len(label_names) != NUM_CLASSES:
        raise ModelConfigurationError(f"Expected {NUM_CLASSES} canonical labels, found {len(label_names)}.")
    try:
        import torch
        get_model = _get_model_factory()
    except ImportError as exc:
        raise ModelConfigurationError("PyTorch is not installed. Install ecg_demo/requirements.txt to enable inference.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        metadata, state_dict = _checkpoint_metadata(checkpoint)
        checkpoint_arch = metadata.get("architecture") or spec["architecture"]
        if checkpoint_arch != spec["architecture"]:
            raise ModelConfigurationError(
                f"Selected architecture '{spec['architecture']}' does not match checkpoint architecture '{checkpoint_arch}'."
            )
        if metadata.get("num_classes") not in (None, len(label_names)):
            raise ModelConfigurationError("Checkpoint class count does not match the canonical 94-class label list.")
        checkpoint_labels = metadata.get("label_names")
        if checkpoint_labels is not None and list(checkpoint_labels) != list(label_names):
            raise ModelConfigurationError("Checkpoint label order does not match the canonical training label order.")
        model = get_model(checkpoint_arch, num_classes=len(label_names)).to(device)
        _validate_state_dict(model, state_dict, checkpoint_arch, label_names)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except ModelConfigurationError:
        raise
    except Exception as exc:
        raise ModelConfigurationError(f"Could not load or validate checkpoint {model_path}: {exc}") from exc
    return model, {
        "key": model_key,
        "name": spec["name"],
        "architecture": checkpoint_arch,
        "path": str(model_path),
        "epoch": metadata.get("epoch", "unknown"),
        "device": str(device),
    }
