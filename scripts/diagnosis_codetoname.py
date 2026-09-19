from pathlib import Path
import pandas as pd

# Input and output files
INPUT_FILE = "diagnosis_code_distribution.csv"
OUTPUT_FILE = "diagnosis_code_mapping.csv"

# Verified code → diagnosis name mapping
# Add more codes only after verification.
CODE_MAPPING = {
    "426177001": "Sinus Bradycardia",
    "426783006": "Sinus Rhythm",
    "164890007": "Atrial Flutter",
    "427084000": "Sinus Tachycardia",
    "164934002": "T Wave Abnormal",
    "164889003": "Atrial Fibrillation",
    "427393009": "Sinus Arrhythmia",
    "426761007": "Supraventricular Tachycardia",
    "59118001": "Right Bundle Branch Block",
    "39732003": "Left Axis Deviation",
    "164909002": "Left Bundle Branch Block",
    "284470004": "Premature Atrial Contraction",
    "427172004": "Premature Ventricular Contractions",
    "164947007": "Prolonged PR Interval",
    "111975006": "Prolonged QT Interval",
    "164917005": "Q Wave Abnormal",
    "47665007": "Right Axis Deviation",
    "63593006": "Supraventricular Premature Beats",
    "17338001": "Ventricular Premature Beats",
    "59931005": "T Wave Inversion",
}

# Read frequency data
df = pd.read_csv(INPUT_FILE, dtype={"diagnosis_code": str})

# Map codes to names
df["diagnosis_name"] = df["diagnosis_code"].map(CODE_MAPPING)

# Keep unknown codes visible instead of guessing
df["diagnosis_name"] = df["diagnosis_name"].fillna("Unknown - Verify")

# Export
df.to_csv(OUTPUT_FILE, index=False)

print("Mapping completed!")
print("Saved:", OUTPUT_FILE)

print("\nMapped codes:",
      (df["diagnosis_name"] != "Unknown - Verify").sum())

print("Unmapped codes:",
      (df["diagnosis_name"] == "Unknown - Verify").sum())

print("\nPreview:")
print(df.head(20).to_string(index=False))