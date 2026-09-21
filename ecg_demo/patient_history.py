"""Separate UCI Heart Disease patient-history model integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "patient_history"
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_NAME = "num"
TARGET_DEFINITION = "Original UCI num: 0=no angiographic disease, 1-4=presence; model target is num > 0."
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = [feature for feature in FEATURE_NAMES if feature not in CATEGORICAL_FEATURES]
DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "Peak exercise ST slope",
    "ca": "Major vessels",
    "thal": "Thalassemia test",
}


class PatientHistoryError(ValueError):
    pass


def artifact_paths(artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Path]:
    return {
        "model": artifact_dir / "patient_history_model.joblib",
        "legacy_model": artifact_dir / "model1.pth",
        "metadata": artifact_dir / "feature_metadata.json",
        "metrics": artifact_dir / "metrics.json",
        "config": artifact_dir / "training_config.json",
    }


def resolve_history_model_path(artifact_dir: Path = ARTIFACT_DIR) -> Path:
    paths = artifact_paths(artifact_dir)
    for candidate in (paths["model"], paths["legacy_model"]):
        if candidate.is_file():
            return candidate
    return paths["model"]


def load_history_artifact(artifact_dir: Path = ARTIFACT_DIR):
    paths = artifact_paths(artifact_dir)
    model_path = resolve_history_model_path(artifact_dir)
    if not model_path.is_file():
        return None, {"available": False, "message": f"Patient-history model not trained. Expected {model_path}"}
    try:
        model = joblib.load(model_path)
        metadata = json.loads(paths["metadata"].read_text()) if paths["metadata"].is_file() else {}
        return model, {"available": True, **metadata, "artifact_dir": str(artifact_dir), "artifact_path": str(model_path)}
    except Exception as exc:
        return None, {"available": False, "message": f"Could not load patient-history model: {exc}"}


def _coerce_input(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise PatientHistoryError("Patient history must be a JSON object or form fields.")
    aliases = {"blood_pressure": "trestbps", "cholesterol": "chol", "max_heart_rate": "thalach", "st_depression": "oldpeak", "exercise_angina": "exang"}
    normalized = {aliases.get(str(key), str(key)): value for key, value in payload.items()}
    unknown = sorted(set(normalized) - set(FEATURE_NAMES))
    if unknown:
        raise PatientHistoryError(f"Unsupported patient-history fields: {', '.join(unknown)}")
    missing = [feature for feature in FEATURE_NAMES if feature not in normalized or normalized[feature] in (None, "")]
    if missing:
        raise PatientHistoryError(f"Missing required fields: {', '.join(missing)}")
    row = {}
    for feature in FEATURE_NAMES:
        try:
            value = float(normalized[feature])
        except (TypeError, ValueError) as exc:
            raise PatientHistoryError(f"Invalid numeric value for {feature}.") from exc
        if not np.isfinite(value):
            raise PatientHistoryError(f"Invalid non-finite value for {feature}.")
        row[feature] = value
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def validate_history_input(payload: dict[str, Any]) -> pd.DataFrame:
    frame = _coerce_input(payload)
    limits = {
        "age": (1, 120), "sex": (0, 1), "cp": (1, 4), "trestbps": (50, 300),
        "chol": (0, 1000), "fbs": (0, 1), "restecg": (0, 2), "thalach": (40, 300),
        "exang": (0, 1), "oldpeak": (-5, 15), "slope": (1, 3), "ca": (0, 3), "thal": (3, 7),
    }
    for feature, (lower, upper) in limits.items():
        value = float(frame.iloc[0][feature])
        if not lower <= value <= upper:
            raise PatientHistoryError(f"{DISPLAY_NAMES[feature]} must be between {lower} and {upper}.")
        if feature in {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"} and value != int(value):
            raise PatientHistoryError(f"{DISPLAY_NAMES[feature]} must be a valid coded category.")
    return frame


def explain_prediction(model, frame: pd.DataFrame, probability: float) -> list[dict[str, Any]]:
    """Return transparent input-context indicators, not causal explanations."""
    coefficients = getattr(model[-1], "coef_", None) if hasattr(model, "__getitem__") else None
    if coefficients is None:
        return []
    try:
        names = model[:-1].get_feature_names_out()
        values = model[:-1].transform(frame).toarray() if hasattr(model[:-1].transform(frame), "toarray") else model[:-1].transform(frame)
        effects = values[0] * coefficients[0]
        order = np.argsort(np.abs(effects))[::-1][:5]
        return [{"feature": str(names[index]), "direction": "higher model score" if effects[index] > 0 else "lower model score", "magnitude": round(float(abs(effects[index])), 4)} for index in order]
    except Exception:
        return []


def predict_history(model, payload: dict[str, Any]) -> dict[str, Any]:
    frame = validate_history_input(payload)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "classification": "Model-estimated elevated risk" if prediction else "Model-estimated lower risk",
        "probability": round(probability, 6),
        "probability_label": "Estimated possibility from the UCI-trained model; not a calibrated clinical probability.",
        "input_summary": {DISPLAY_NAMES[feature]: float(frame.iloc[0][feature]) for feature in FEATURE_NAMES},
        "feature_indicators": explain_prediction(model, frame, probability),
        "target_definition": TARGET_DEFINITION,
        "disclaimer": "This is a dataset-based research estimate, not a medical diagnosis. Do not delay emergency care.",
    }
