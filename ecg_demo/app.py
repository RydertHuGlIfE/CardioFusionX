import numpy as np
from pathlib import Path

from flask import Flask, jsonify, render_template
from flask import request

from config import LEAD_NAMES, SAMPLE_RATE, available_models, canonical_labels, diagnosis_mapping
from export_utils import prediction_csv, prediction_json
from inference import InputValidationError, normalize_signal, predict_signal, prepare_prediction, read_mat_upload
from model_loader import ModelConfigurationError, load_model, load_thresholds
from robustness import apply_noise, changed_labels
from signal_quality import analyze_signal
from patient_history import load_history_artifact, PatientHistoryError, predict_history


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MODEL_PATH = Path("/run/media/Ryder/Coding/Coding/CardioFusionX/models/ECG_model/model1.pth")
THRESHOLD_PATH = Path("/run/media/Ryder/Coding/Coding/CardioFusionX/models/ECG_model/thresholds.json")
app.config["MODEL_PATH"] = str(MODEL_PATH)
app.config["THRESHOLD_PATH"] = str(THRESHOLD_PATH)

try:
    LABELS = canonical_labels()
    MAPPING = diagnosis_mapping()
    CONFIG_ERROR = None
except Exception as exc:
    LABELS, MAPPING, CONFIG_ERROR = [], {}, str(exc)

STATS = {"uploaded": 0, "processed": 0, "predictions": 0}
HISTORY_MODEL, HISTORY_INFO = load_history_artifact()


def render_dashboard(error=None, result=None, signal=None, quality=None, history_result=None, history_error=None):
    return render_template(
        "index.html",
        error=error,
        result=result,
        signal=signal,
        quality=quality,
        model_options=available_models(),
        stats=STATS,
        lead_names=LEAD_NAMES,
        sample_rate=SAMPLE_RATE,
        config_error=CONFIG_ERROR,
        history_info=HISTORY_INFO,
        history_result=history_result,
        history_error=history_error,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_dashboard()
    uploaded = request.files.get("ecg_file")
    model_key = request.form.get("model_key") or ("resnet_asl" if MODEL_PATH.is_file() else "optimized_cnn")
    strategy = request.form.get("threshold_strategy") or ("tuned" if THRESHOLD_PATH.is_file() else "fixed")
    STATS["uploaded"] += 1
    if not uploaded or not uploaded.filename:
        return render_dashboard("Please upload a .mat ECG file.")
    if not LABELS:
        return render_dashboard(f"Configuration error: {CONFIG_ERROR}")
    try:
        raw_signal, result, _ = prepare_prediction(uploaded, model_key, strategy, MAPPING, LABELS)
        quality = analyze_signal(raw_signal, SAMPLE_RATE)
        result["lead_names"] = LEAD_NAMES
        STATS["processed"] += 1
        STATS["predictions"] += result["detected_count"]
        return render_dashboard(result=result, signal=raw_signal.tolist(), quality=quality)
    except (InputValidationError, ModelConfigurationError, RuntimeError, ValueError) as exc:
        return render_dashboard(str(exc))
    except Exception:
        app.logger.exception("Unexpected ECG processing failure")
        return render_dashboard("The ECG could not be processed. Check the file and server logs.")


@app.post("/predict-history")
def predict_history_route():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    if HISTORY_MODEL is None:
        message = HISTORY_INFO.get("message", "Patient-history model is unavailable.")
        return jsonify({"error": message, "module": "patient_history"}), 503
    try:
        return jsonify(predict_history(HISTORY_MODEL, payload))
    except PatientHistoryError as exc:
        return jsonify({"error": str(exc), "module": "patient_history"}), 400
    except Exception:
        app.logger.exception("Patient-history prediction failure")
        return jsonify({"error": "Patient history could not be processed.", "module": "patient_history"}), 500


@app.post("/api/stress-test")
def stress_test():
    try:
        uploaded = request.files.get("ecg_file")
        payload = request.get_json(silent=True) or {}
        noise_type = request.form.get("noise_type", payload.get("noise_type", "gaussian"))
        intensity = float(request.form.get("intensity", payload.get("intensity", "0.3")))
        if uploaded and uploaded.filename:
            raw_signal, filename = read_mat_upload(uploaded)
        elif payload.get("signal") is not None:
            raw_signal = np.asarray(payload["signal"], dtype=np.float32)
            if raw_signal.shape != (12, 5000):
                raise ValueError("Signal payload must be shaped (12, 5000).")
            if not np.isfinite(raw_signal).all():
                raise ValueError("Signal payload contains invalid numeric values.")
            filename = payload.get("filename", "current-signal.mat")
        else:
            raise ValueError("Please upload a .mat ECG file or provide a valid signal payload.")
        model_key = request.form.get("model_key") or payload.get("model_key") or ("resnet_asl" if MODEL_PATH.is_file() else "optimized_cnn")
        strategy = request.form.get("threshold_strategy") or payload.get("threshold_strategy") or ("tuned" if THRESHOLD_PATH.is_file() else "fixed")
        model, info = load_model(model_key, LABELS)
        thresholds, _ = load_thresholds(LABELS, info["path"], strategy)
        original = predict_signal(model, normalize_signal(raw_signal), LABELS, thresholds, MAPPING, info, strategy)
        noisy_signal = apply_noise(raw_signal, noise_type, intensity)
        noisy = predict_signal(model, normalize_signal(noisy_signal), LABELS, thresholds, MAPPING, info, strategy)
        return jsonify({"filename": filename, "noise_type": noise_type, "intensity": intensity, "original": original, "noisy": noisy, "changed_labels": changed_labels(original, noisy), "quality_before": analyze_signal(raw_signal, SAMPLE_RATE), "quality_after": analyze_signal(noisy_signal, SAMPLE_RATE), "signal": noisy_signal.tolist()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/counterfactual")
def counterfactual():
    try:
        uploaded = request.files.get("ecg_file")
        payload = request.get_json(silent=True) or {}
        if uploaded and uploaded.filename:
            raw_signal, _ = read_mat_upload(uploaded)
        elif payload.get("signal") is not None:
            raw_signal = np.asarray(payload["signal"], dtype=np.float32)
            if raw_signal.shape != (12, 5000):
                raise ValueError("Signal payload must be shaped (12, 5000).")
            if not np.isfinite(raw_signal).all():
                raise ValueError("Signal payload contains invalid numeric values.")
        else:
            raise ValueError("Please upload a .mat ECG file or provide a valid signal payload.")
        start = max(0, int(request.form.get("start", payload.get("start", 0))))
        end = min(raw_signal.shape[1], int(request.form.get("end", payload.get("end", raw_signal.shape[1]))))
        if start >= end:
            raise ValueError("Counterfactual segment must have a positive duration.")
        modified = raw_signal.copy()
        replacement = np.mean(modified[:, max(0, start - 1):min(modified.shape[1], end + 1)], axis=1, keepdims=True)
        modified[:, start:end] = replacement
        model_key = request.form.get("model_key") or payload.get("model_key") or ("resnet_asl" if MODEL_PATH.is_file() else "optimized_cnn")
        strategy = request.form.get("threshold_strategy") or payload.get("threshold_strategy") or ("tuned" if THRESHOLD_PATH.is_file() else "fixed")
        model, info = load_model(model_key, LABELS)
        thresholds, _ = load_thresholds(LABELS, info["path"], strategy)
        original = predict_signal(model, normalize_signal(raw_signal), LABELS, thresholds, MAPPING, info, strategy)
        changed = predict_signal(model, normalize_signal(modified), LABELS, thresholds, MAPPING, info, strategy)
        return jsonify({"start_sample": start, "end_sample": end, "original": original, "modified": changed, "changed_labels": changed_labels(original, changed)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/export/json")
def export_json():
    payload = request.get_json(silent=True) or {}
    response = app.response_class(prediction_json(payload.get("result", {}), payload.get("quality", {}), payload.get("model", {})), mimetype="application/json")
    response.headers["Content-Disposition"] = "attachment; filename=cardiofusionx-result.json"
    return response


@app.post("/api/export/csv")
def export_csv():
    payload = request.get_json(silent=True) or {}
    response = app.response_class(prediction_csv(payload.get("result", {})), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cardiofusionx-predictions.csv"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
