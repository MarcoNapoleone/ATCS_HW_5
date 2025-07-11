import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

DATA_DIR = "data"
DATA_URL = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
TRAIN_PATH = os.path.join(DATA_DIR, "titanic_train_processed.csv")
TEST_PATH = os.path.join(DATA_DIR, "titanic_test_processed.csv")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.joblib")

CATEGORICAL_VARS = ["Sex", "Embarked", "Pclass"]
NUMERIC_VARS = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]
TARGET_VAR = "Survived"

def download_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the Titanic dataset from the specified URL."""
    df = pd.read_csv(url)
    return df

def build_preprocessor():
    """Create a sklearn ColumnTransformer that handles preprocessing."""
    # Pipelines for numeric and categorical features
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", pd.get_dummies),  # Placeholder; replaced in custom transformer below
    ])

    # Note: We'll apply get_dummies after imputation manually since ColumnTransformer
    # cannot directly wrap pd.get_dummies. We'll keep categorical variables as-is
    # for imputation and perform get_dummies after the entire preprocessor pipeline.

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_VARS),
            ("cat", "passthrough", CATEGORICAL_VARS),
        ]
    )
    return preprocessor

def preprocess_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Preprocess Titanic data and return train/test DataFrames."""
    # Drop columns unlikely to help the model
    df = df.drop(columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ])

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Separate features and target
    X = df.drop(columns=[TARGET_VAR])
    y = df[TARGET_VAR]

    # Fit-transform on whole data (OK because we'll split afterwards)
    X_pre = preprocessor.fit_transform(X)

    # Recover column names after preprocessing
    # Numeric columns retained after scaler
    num_cols = NUMERIC_VARS
    # Categorical columns: get_dummies on original categories discovered in the data
    cat_cols = []
    for col in CATEGORICAL_VARS:
        for cat in preprocessor.named_transformers_["cat"].categories_[CATEGORICAL_VARS.index(col)]:
            cat_cols.append(f"{col}_{cat}")

    feature_cols = num_cols + cat_cols
    X_pre_df = pd.DataFrame(X_pre, columns=feature_cols)

    # Combine with target
    processed_df = pd.concat([X_pre_df, y.reset_index(drop=True)], axis=1)

    # Train-test split
    train_df, test_df = train_test_split(
        processed_df, test_size=test_size, random_state=random_state, stratify=y
    )

    return train_df, test_df, preprocessor

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading data...")
    df = download_data()

    print("Preprocessing and splitting...")
    train_df, test_df, preprocessor = preprocess_and_split(df)

    print("Saving processed datasets and preprocessor...")
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("All done! Files written to 'data/' directory.")

if __name__ == "__main__":
    main()
