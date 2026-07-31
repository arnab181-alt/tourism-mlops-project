"""Data Registration.

Validates that the raw dataset lives in the repository and is well-formed
before the rest of the pipeline runs. In the GitHub Actions workflow this is
the first job; it uploads the validated CSV as an artifact for later stages.
"""
import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


def main():
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing expected columns: {missing}"
    print("All expected columns are present.")

    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(dropna=False))

    nulls = df.isna().sum()
    print("\nMissing values per column:")
    print(nulls[nulls > 0] if nulls.sum() else "None")

    print("\nData registration successful.")


if __name__ == "__main__":
    main()
