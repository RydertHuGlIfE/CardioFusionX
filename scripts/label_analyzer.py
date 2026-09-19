from pathlib import Path
from collections import Counter

ROOT = Path("ecg-arrhythmia-data")

hea_files = sorted(ROOT.rglob("*.hea"))
print("Total HEA files:", len(hea_files))

diagnosis_counter = Counter()

records_with_diagnosis = 0
records_without_diagnosis = 0

for index, hea_path in enumerate(hea_files):

    # Progress every 5000 files
    if (index + 1) % 5000 == 0:
        print(f"Processing: {index + 1}/{len(hea_files)}")

    try:
        content = hea_path.read_text(errors="ignore")
        lines = content.splitlines()

        # Find diagnosis line
        diagnosis_line = next(
            (line for line in lines if line.startswith("#Dx:")),
            None
        )

        if diagnosis_line:
            diagnosis_text = diagnosis_line.split(":", 1)[1].strip()
            diagnosis_codes = diagnosis_text.split(",")
            for code in diagnosis_codes:
                code = code.strip()

                if code:
                    diagnosis_counter[code] += 1

            records_with_diagnosis += 1

        else:
            records_without_diagnosis += 1

    except Exception as e:
        print(f"Error reading {hea_path}: {e}")



print("\n========== SUMMARY ==========")

print("Total records:", len(hea_files))
print("Records with diagnosis:", records_with_diagnosis)
print("Records without diagnosis:", records_without_diagnosis)

print("\n========== TOP 20 CODES ==========")

for code, count in diagnosis_counter.most_common(20):
    print(f"Code: {code} | Count: {count}")

print("\nUnique diagnosis codes:", len(diagnosis_counter))