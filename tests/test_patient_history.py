import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ecg_demo"))

from patient_history import PatientHistoryError, load_history_artifact, predict_history


VALID_INPUT = {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 3,
}


def test_saved_history_model_loads():
    model, info = load_history_artifact()
    assert model is not None
    assert info["available"] is True
    assert len(info["features"]) == 13


def test_history_prediction_response():
    model, _ = load_history_artifact()
    result = predict_history(model, VALID_INPUT)
    assert 0 <= result["probability"] <= 1
    assert result["classification"].startswith("Model-estimated")
    assert "not a medical diagnosis" in result["disclaimer"].lower()


def test_history_validation_rejects_bad_and_missing_values():
    model, _ = load_history_artifact()
    with pytest.raises(PatientHistoryError, match="Age"):
        predict_history(model, {**VALID_INPUT, "age": 999})
    with pytest.raises(PatientHistoryError, match="Missing required fields"):
        predict_history(model, {"age": 55})
