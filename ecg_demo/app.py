
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from flask import Flask, render_template, request

from model import ECGCNN

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 94
NUM_LEADS = 12
SIGNAL_LENGTH = 5000
THRESHOLD = 0.65

# Replace/add mappings as needed. Unlisted labels remain readable as their code.
DIAGNOSIS_NAMES = {
    "426177001": "Sinus Bradycardia",
    "426783006": "Sinus Rhythm",
    "164890007": "Atrial Flutter",
    "427084000": "Sinus Tachycardia",
    "164934002": "T Wave Abnormal",
    "59931005": "T Wave Inversion",
    "427393009": "Sinus Arrhythmia",
    "164889003": "Atrial Fibrillation",
    "39732003": "Left Axis Deviation",
    "284470004": "Premature Atrial Contraction",
    "426761007": "Supraventricular Tachycardia",
    "59118001": "Right Bundle Branch Block",
    "111975006": "Prolonged QT Interval",
    "164947007": "Prolonged PR Interval",
    "427172004": "Premature Ventricular Contractions",
    "164917005": "Q Wave Abnormal",
    "47665007": "Right Axis Deviation",
    "17338001": "Ventricular Premature Beats",
    "164909002": "Left Bundle Branch Block",
    "63593006": "Supraventricular Premature Beats",
    "425856008": "Paroxysmal Ventricular Tachycardia",
}

def label_name(label):
    code = label.replace("label_", "")
    return DIAGNOSIS_NAMES.get(code, f"Code {code} (mapping unavailable)")

def load_model():
    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint.get("epoch", "unknown")

def prepare_signal(file_storage):
    temp_path = BASE_DIR / "_uploaded_sample.mat"
    file_storage.save(temp_path)
    try:
        mat = loadmat(temp_path)
        if "val" not in mat:
            raise ValueError("The MAT file must contain a variable named 'val'.")
        signal = mat["val"].astype(np.float32)
    finally:
        temp_path.unlink(missing_ok=True)

    if signal.shape != (NUM_LEADS, SIGNAL_LENGTH):
        raise ValueError(
            f"Expected signal shape {(NUM_LEADS, SIGNAL_LENGTH)}, got {tuple(signal.shape)}."
        )

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    normalized = (signal - mean) / std
    return signal, normalized

def predict(model, normalized_signal):
    tensor = torch.tensor(normalized_signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()

    labels = [f"label_{x}" for x in [
        "426177001", "426783006", "164890007", "427084000", "164934002",
        "55827005", "55930002", "59931005", "427393009", "164889003",
        "429622005", "39732003", "284470004", "10370003", "428750005",
        "270492004", "713427006", "427172004", "164917005", "251146004",
        "47665007", "164930006", "698252002", "426761007", "61721007",
        "59118001", "164873001", "365413008", "111975006", "6374002",
        "445118002", "428417006", "713422000", "17338001", "713426002",
        "233917008", "164909002", "251223006", "106068003", "733534002",
        "164931005", "251199005", "164912004", "29320008", "164937009",
        "164865005", "13640000", "89792004", "425856008", "251205003",
        "81898007", "251198002", "27885002", "426995002", "74390002",
        "195042002", "251170000", "50799005", "164896001", "54329005",
        "75532003", "164947007", "57054005", "446358003", "67751000119106",
        "5609005", "54016002", "233897008", "426648003", "49578007",
        "251187003", "251166008", "233892002", "61277005", "195060002",
        "426664006", "251164006", "63593006", "251180001", "446813000",
        "17366009", "426627000", "111288001", "426183003", "251120003",
        "65778007", "445211001", "418818005", "251173003", "164942001",
        "11157007", "195101003", "77867006", "67741000119109"
    ]]

    results = []
    for label, probability in zip(labels, probabilities):
        if probability >= THRESHOLD:
            results.append({
                "code": label,
                "name": label_name(label),
                "probability": round(float(probability), 4),
            })
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

MODEL, MODEL_EPOCH = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    signal = None
    if request.method == "POST":
        uploaded = request.files.get("ecg_file")
        if not uploaded or not uploaded.filename:
            error = "Please upload a .mat ECG file."
        elif not uploaded.filename.lower().endswith(".mat"):
            error = "Only .mat files are supported in this demo."
        else:
            try:
                raw_signal, normalized_signal = prepare_signal(uploaded)
                predictions = predict(MODEL, normalized_signal)
                signal = raw_signal.tolist()
                result = {
                    "predictions": predictions,
                    "threshold": THRESHOLD,
                    "epoch": MODEL_EPOCH,
                    "shape": list(raw_signal.shape),
                }
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", result=result, signal=signal, error=error)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
