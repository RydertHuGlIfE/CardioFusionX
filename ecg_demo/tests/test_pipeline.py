import io
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from werkzeug.datastructures import FileStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NUM_LEADS, SIGNAL_LENGTH
from inference import InputValidationError, read_mat_upload
from model_loader import ModelConfigurationError, load_model, load_thresholds
from signal_quality import analyze_signal


def upload(payload, filename="sample.mat"):
    return FileStorage(stream=io.BytesIO(payload), filename=filename)


def mat_payload(**values):
    stream = io.BytesIO()
    savemat(stream, values)
    return stream.getvalue()


def test_valid_mat_shape():
    signal, name = read_mat_upload(upload(mat_payload(val=np.zeros((NUM_LEADS, SIGNAL_LENGTH), dtype=np.float32))))
    assert signal.shape == (12, 5000)
    assert name == "sample.mat"


def test_missing_val_key():
    with pytest.raises(InputValidationError, match="Missing val key"):
        read_mat_upload(upload(mat_payload(other=np.zeros((12, 5000)))))


def test_invalid_shape_and_non_mat():
    with pytest.raises(InputValidationError, match="Incorrect ECG shape"):
        read_mat_upload(upload(mat_payload(val=np.zeros((2, 10)))))
    with pytest.raises(InputValidationError, match="Invalid file format"):
        read_mat_upload(upload(b"not matlab", "sample.txt"))


def test_non_finite_signal():
    signal = np.zeros((12, 5000), dtype=np.float32)
    signal[0, 0] = np.nan
    with pytest.raises(InputValidationError, match="NaN"):
        read_mat_upload(upload(mat_payload(val=signal)))


def test_quality_metrics():
    quality = analyze_signal(np.zeros((12, 5000), dtype=np.float32))
    assert quality["flatline_leads"] == 12
    assert quality["status"] in {"Poor", "Unreliable"}


def test_fixed_thresholds():
    thresholds, source = load_thresholds(["label_a", "label_b"], "missing/model.pth", "fixed")
    assert source is None
    assert np.allclose(thresholds, [0.5, 0.5])


def test_missing_checkpoint_is_clear():
    with pytest.raises(ModelConfigurationError, match="Checkpoint is missing"):
        load_model("production_resnetse", [f"label_{index}" for index in range(94)])
