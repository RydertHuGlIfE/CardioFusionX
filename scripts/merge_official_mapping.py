import pandas as pd

freq_df = pd.read_csv(
    "diagnosis_code_distribution.csv",
    dtype={"diagnosis_code": str}
)

mapping_df = pd.read_csv(
    "ecg-arrhythmia-data/ConditionNames_SNOMED-CT.csv",
    dtype={"Snomed_CT": str}
)

mapping_df = mapping_df.rename(
    columns={"Snomed_CT": "diagnosis_code"}
)

# Remove duplicate codes before merging
mapping_df = mapping_df.drop_duplicates(
    subset=["diagnosis_code"],
    keep="first"
)

merged_df = freq_df.merge(
    mapping_df[["diagnosis_code", "Acronym Name", "Full Name"]],
    on="diagnosis_code",
    how="left",
    validate="one_to_one"
)

print("Mapping completed!")
print("Total codes:", len(merged_df))
print("Mapped codes:", merged_df["Full Name"].notna().sum())
print("Unmapped codes:", merged_df["Full Name"].isna().sum())

print("\nTop 20 mapped codes:")
print(merged_df.head(20).to_string(index=False))
