"""Data Preparation.

Loads the registered dataset, applies cleaning, and produces stratified
train/test splits. The splits are written to the working directory root so the
GitHub Actions workflow can pass them to the training job as artifacts.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"


def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)
    # Drop index / identifier columns that carry no predictive signal
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])
    # Fix known data-entry issues
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    return df


def main():
    df = load_and_clean()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Saved at repo root -> uploaded as the "data-splits" artifact by the workflow
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape} | Xtest: {Xtest.shape}")
    print("Train target balance:")
    print(ytrain.value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
