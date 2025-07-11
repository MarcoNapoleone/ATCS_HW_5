#!/usr/bin/env python
"""
Download Titanic CSV (if absent), clean/encode, split train/test, save artifacts:
  data/titanic_train.csv
  data/titanic_test.csv
  data/preprocess_ct.joblib
"""
import os, urllib.request, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from titanic_influence.utils import default_preprocessor

URL     = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
DATA_D  = Path("data")
RAW_CSV = DATA_D / "titanic.csv"
TRAIN_CSV = DATA_D / "titanic_train.csv"
TEST_CSV  = DATA_D / "titanic_test.csv"
CT_PATH   = DATA_D / "preprocess_ct.joblib"

DROP_COLS = ["Cabin", "Ticket", "Name", "PassengerId"]

# --------------------------------------------------------------------------- #
def download():
    DATA_D.mkdir(exist_ok=True)
    print("⬇️  Downloading Titanic CSV...")
    urllib.request.urlretrieve(URL, RAW_CSV)

# --------------------------------------------------------------------------- #
def main():
    if not RAW_CSV.exists():
        download()

    df = pd.read_csv(RAW_CSV)
    df.drop(columns=DROP_COLS, inplace=True)

    # Preprocess / split
    ct = default_preprocessor(df)
    X, y = None, None  # just to use ct.fit below
    _ = ct.fit(df.drop(columns=["Survived"]))

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Survived"]
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    joblib.dump(ct, CT_PATH)

    print("✅ Preprocessing finished")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Artifacts → {CT_PATH}, {TRAIN_CSV}, {TEST_CSV}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
