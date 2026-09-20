"""Train the separate UCI patient-history cardiac risk model.

Run from the CardioFusionX project root:
    venv/bin/python train_patient_history.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecg_demo.patient_history import (
    ARTIFACT_DIR, CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES,
    TARGET_DEFINITION, TARGET_NAME,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "heart+disease"
DATA_FILES = [
    DATA_DIR / "processed.cleveland.data",
    DATA_DIR / "processed.hungarian.data",
    DATA_DIR / "processed.switzerland.data",
    DATA_DIR / "processed.va.data",
]
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> tuple[pd.DataFrame, dict]:
    rows = []
    sources = {}
    for path in DATA_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Required UCI file not found: {path}")
        frame = pd.read_csv(path, header=None, names=FEATURE_NAMES + [TARGET_NAME], na_values=["?", -9, -9.0], skipinitialspace=True)
        if frame.shape[1] != 14:
            raise ValueError(f"Unexpected column count in {path}: {frame.shape[1]}")
        frame["source"] = path.name
        rows.append(frame)
        sources[path.name] = int(len(frame))
    data = pd.concat(rows, ignore_index=True)
    for column in FEATURE_NAMES + [TARGET_NAME]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[TARGET_NAME]).copy()
    original_targets = sorted(data[TARGET_NAME].astype(int).unique().tolist())
    if not set(original_targets).issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected UCI target values: {original_targets}")
    data["target_binary"] = (data[TARGET_NAME] > 0).astype(int)
    metadata = {
        "dataset": "UCI Heart Disease / processed Cleveland, Hungarian, Switzerland, VA files",
        "source_directory": str(DATA_DIR),
        "source_files": sources,
        "rows": int(len(data)),
        "features": FEATURE_NAMES,
        "original_target": TARGET_NAME,
        "original_target_values": original_targets,
        "target_definition": TARGET_DEFINITION,
        "missing_tokens": ["?", "-9", "-9.0"],
        "missing_values_after_parse": {feature: int(data[feature].isna().sum()) for feature in FEATURE_NAMES},
    }
    return data, metadata


def make_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def specificity(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if tn + fp else 0.0


def evaluate(y_true, probabilities, threshold=0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "specificity": specificity(y_true, predictions),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "positive_count": int(predictions.sum()),
        "sample_count": int(len(y_true)),
    }


def main() -> None:
    data, dataset_metadata = load_dataset()
    x = data[FEATURE_NAMES]
    y = data["target_binary"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model = make_pipeline()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, probabilities)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_result = cross_validate(model, x, y, cv=cv, scoring={"roc_auc": "roc_auc", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}, return_train_score=False)
    metrics["cross_validation"] = {
        metric: {"mean": float(np.mean(cv_result[f"test_{metric}"])), "std": float(np.std(cv_result[f"test_{metric}"])), "folds": [float(value) for value in cv_result[f"test_{metric}"]]}
        for metric in ("roc_auc", "f1", "balanced_accuracy")
    }
    metrics["evaluation_split"] = "stratified 75% train / 25% holdout test"
    metrics["test_positive_rate"] = float(y_test.mean())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": ARTIFACT_DIR / "patient_history_model.joblib",
        "metadata": ARTIFACT_DIR / "feature_metadata.json",
        "metrics": ARTIFACT_DIR / "metrics.json",
        "config": ARTIFACT_DIR / "training_config.json",
        "confusion_matrix": ARTIFACT_DIR / "confusion_matrix.png",
        "roc_curve": ARTIFACT_DIR / "roc_curve.png",
    }
    joblib.dump(model, paths["model"])
    metadata = {
        **dataset_metadata,
        "feature_names": FEATURE_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "display_target": "Model-estimated cardiac risk",
        "supports_free_text_symptoms": False,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    paths["metrics"].write_text(json.dumps(metrics, indent=2))
    paths["config"].write_text(json.dumps({"model": "LogisticRegression", "random_state": RANDOM_STATE, "class_weight": "balanced", "preprocessing": "median/mode imputation, StandardScaler numeric, OneHotEncoder categorical", "data_leakage_control": "preprocessing fitted inside Pipeline on each training fold", "threshold": 0.5}, indent=2))

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4, 4)); axis.imshow(matrix, cmap="Blues"); axis.set_title("UCI holdout confusion matrix"); axis.set_xlabel("Predicted"); axis.set_ylabel("Actual")
    for row in range(2):
        for col in range(2): axis.text(col, row, matrix[row, col], ha="center", va="center")
    fig.tight_layout(); fig.savefig(paths["confusion_matrix"], dpi=150); plt.close(fig)

    from sklearn.metrics import RocCurveDisplay
    fig, axis = plt.subplots(figsize=(5, 4)); RocCurveDisplay.from_predictions(y_test, probabilities, ax=axis); axis.set_title("UCI holdout ROC curve"); fig.tight_layout(); fig.savefig(paths["roc_curve"], dpi=150); plt.close(fig)
    print(json.dumps({"artifacts": {key: str(value) for key, value in paths.items()}, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
