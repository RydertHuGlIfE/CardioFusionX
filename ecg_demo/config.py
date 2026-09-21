from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

NUM_CLASSES = 94
NUM_LEADS = 12
SIGNAL_LENGTH = 5000
SAMPLE_RATE = 500
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_THRESHOLD = 0.50
NEAR_THRESHOLD_MARGIN = 0.10
LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Paths are resolved from the repository root and only exposed when they exist.
MODEL_CATALOG = {
    "demo_cnn": {
        "name": "Demo ECGCNN",
        "architecture": "ecg_cnn",
        "path": BASE_DIR / "model.pth",
    },
    "resnet_asl": {
        "name": "ResNet-SE + Asymmetric Loss (ASL)",
        "architecture": "ecg_resnet_se",
        "path": PROJECT_ROOT / "models" / "ECG_model" / "model1.pth",
    },
    "optimized_cnn": {
        "name": "Optimized ECGCNN",
        "architecture": "ecg_cnn",
        "path": PROJECT_ROOT / "experiments/optimized_v1/best_macro_f1/model.pth",
    },
    "targeted_cnn": {
        "name": "Targeted fine-tuned ECGCNN",
        "architecture": "ecg_cnn",
        "path": PROJECT_ROOT / "experiments/targeted_finetune_v1/best_macro_f1/model.pth",
    },
    "production_resnetse": {
        "name": "Production ECGResNetSE",
        "architecture": "ecg_resnet_se",
        "path": PROJECT_ROOT / "experiments/best_production_resnetse_asl/best_macro_f1/model.pth",
    },
}


def available_models():
    return {
        key: {**spec, "path": str(spec["path"]), "available": spec["path"].is_file()}
        for key, spec in MODEL_CATALOG.items()
        if spec["path"].is_file()
    }


def canonical_labels():
    """Return the training label order without importing the ML runtime."""
    import csv

    for csv_path in (PROJECT_ROOT / "train_split.csv", PROJECT_ROOT / "metadata_94_labels.csv"):
        if csv_path.is_file():
            with csv_path.open(newline="") as handle:
                labels = [name for name in next(csv.reader(handle)) if name.startswith("label_")]
            if len(labels) == NUM_CLASSES:
                return labels
    raise FileNotFoundError("No canonical 94-label CSV was found.")


def diagnosis_mapping():
    import csv

    mapping = {}
    path = PROJECT_ROOT / "diagnosis_code_mapping.csv"
    if not path.is_file():
        return mapping
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            code = str(row.get("diagnosis_code", "")).strip()
            name = str(row.get("diagnosis_name", "")).strip()
            if code and name:
                mapping[code] = name
                mapping[f"label_{code}"] = name
    return mapping


def label_display_name(label, mapping):
    if label in {None, ""}:
        return "Unknown - Verify"
    code = str(label).replace("label_", "")
    if label in mapping:
        return mapping[label]
    if code in mapping:
        return mapping[code]
    return "Unknown - Verify"
