from pathlib import Path
import pandas as pd
from collections import Counter

DATASET_DIR = Path("ecg-arrhythmia-data")
OUTPUT_FILE = "metadata_94_labels.csv"

hea_files = list(DATASET_DIR.rglob("*.hea"))

# Collect all diagnosis codes
all_codes = set()
records = []

for hea_path in hea_files:
    diagnosis_codes = []

    with open(hea_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#Dx:"):
                diagnosis_codes = [
                    code.strip()
                    for code in line.split(":", 1)[1].split(",")
                ]
                break

    all_codes.update(diagnosis_codes)

    mat_path = hea_path.with_suffix(".mat")

    records.append({
        "mat_path": str(mat_path),
        "hea_path": str(hea_path),
        "diagnosis_codes": diagnosis_codes
    })

# Sort all 94 codes
all_codes = sorted(all_codes)

# Create binary labels
rows = []

for record in records:
    row = {
        "mat_path": record["mat_path"],
        "hea_path": record["hea_path"]
    }

    diagnosis_set = set(record["diagnosis_codes"])

    for code in all_codes:
        row[f"label_{code}"] = int(code in diagnosis_set)

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("Metadata created!")
print("Total ECG records:", len(df))
print("Total diagnosis codes:", len(all_codes))
print("Total columns:", len(df.columns))
print("Saved to:", OUTPUT_FILE)
