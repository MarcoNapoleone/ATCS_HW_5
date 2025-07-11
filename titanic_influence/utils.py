"""
Shared helpers: build_dataset(), build_model().
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
TARGET       = "Survived"

# --------------------------------------------------------------------------- #
def default_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Fit a preprocessing pipeline on the full DataFrame."""
    num_cols  = df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET)
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler",  StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot",  OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [("num", numeric_pipeline, num_cols),
         ("cat", categorical_pipeline, cat_cols)]
    )

# --------------------------------------------------------------------------- #
def build_dataset(df: pd.DataFrame,
                  transformer: ColumnTransformer,
                  fit: bool = False):
    """Apply transformer → return (X, y) numpy arrays."""
    X = transformer.fit_transform(df.drop(columns=[TARGET])) if fit \
        else transformer.transform(df.drop(columns=[TARGET]))
    y = df[TARGET].values.astype("float32")
    return X.toarray() if hasattr(X, "toarray") else X, y  # handle sparse

# --------------------------------------------------------------------------- #
def build_model(input_dim: int):
    """Simple 3-layer MLP for binary classification."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model
