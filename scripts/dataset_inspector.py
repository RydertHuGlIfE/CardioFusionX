import pathlib

from scipy.io import loadmat

ROOT = pathlib.Path("ecg-arrhythmia-data")

mat_files = sorted(ROOT.glob("*.mat"))   #find files recursviley
hea_files = sorted(ROOT.glob("*.hea"))

print(f"Mat files: {len(mat_files)}")
print(f"Hea files: {len(hea_files)}")

if not mat_files:
    raise FileNotFoundError(f"No .mat files found under {ROOT.resolve()}")

mat_file = mat_files[0]
hea_file = hea_files[0]


#print some content :D
print(f"Mat file: {mat_file}")
print(f"Hea file: {hea_file}")

print(f"Mat file content")

try:
    data = loadmat(mat_file)

    for key, value in data.items():
        if not key.startswith("__"):
            print(
                key,
                {
                    "shape": getattr(value, "shape", None),
                    "dtype": getattr(value, "dtype", None),
                },
            )

except Exception as error:
    print("MAT reading error:", error)

print("\n--- HEADER CONTENT ---")

if hea_file:
    print(hea_file.read_text(errors="ignore")[:3000])

