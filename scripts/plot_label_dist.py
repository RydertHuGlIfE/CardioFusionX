import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("diagnosis_code_distribution.csv")

# Select top 20 codes
top_20 = df.head(20)

# Plot
plt.figure(figsize=(14, 6))

plt.bar(
    top_20["diagnosis_code"].astype(str),
    top_20["occurrence_count"]
)

plt.title("Top 20 Diagnosis Codes")
plt.xlabel("Diagnosis Code")
plt.ylabel("Occurrence Count")

plt.xticks(rotation=75)
plt.tight_layout()

# Save image
plt.savefig("top_20_diagnosis_codes.png", dpi=300)

plt.show()