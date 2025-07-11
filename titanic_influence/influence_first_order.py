"""Compute First-Order influence scores using logistic regression.

This implementation avoids TensorFlow by fitting a simple logistic regression
model on the preprocessed Titanic training data. Influence scores for each test
instance are computed using the classical influence function formula:

    s_{ij} = grad_test_j^T @ H^{-1} @ grad_train_i

where ``grad_test_j`` is the gradient of the test loss with respect to the
model parameters and ``H`` is the Hessian of the training loss.
The scores are saved under ``results/influence_first_order``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.linear_model import LogisticRegression

DATA_D = Path("data")
OUT_D = Path("results") / "influence_first_order"


def _load_data():
    ct = joblib.load(DATA_D / "preprocess_ct.joblib")
    train_df = pd.read_csv(DATA_D / "titanic_train.csv")
    test_df = pd.read_csv(DATA_D / "titanic_test.csv")

    X_train = ct.transform(train_df.drop(columns=["Survived"]))
    y_train = train_df["Survived"].values.astype(float)
    X_test = ct.transform(test_df.drop(columns=["Survived"]))
    y_test = test_df["Survived"].values.astype(float)
    return X_train, y_train, X_test, y_test


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _compute_h_inv(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    X_ext = _add_bias(X)
    p = model.predict_proba(X)[:, 1]
    r = p * (1 - p)
    H = (X_ext.T * r) @ X_ext
    return pinv(H)


def _gradients(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)[:, 1]
    return (p - y)[:, None] * _add_bias(X)


def compute_influence(test_indices: list[int], k: int = 20, output_dir: Path | None = None) -> dict[int, pd.DataFrame]:
    """Compute influences and return a mapping ``{index: df}``."""
    X_tr, y_tr, X_te, y_te = _load_data()
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    model.fit(X_tr, y_tr)

    h_inv = _compute_h_inv(model, X_tr)
    grads_train = _gradients(model, X_tr, y_tr)

    output_dir = OUT_D if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for idx in test_indices:
        grad_t = _gradients(model, X_te[idx : idx + 1], y_te[idx : idx + 1])[0]
        scores = grads_train @ h_inv @ grad_t
        order = np.argsort(-scores)
        df = pd.DataFrame(
            {
                "train_index": order,
                "score": scores[order],
                "rank": np.arange(1, len(order) + 1),
            }
        )
        df.to_csv(output_dir / f"test_{idx}.csv", index=False)
        results[idx] = df.head(k)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-indices", nargs="+", type=int, default=[0])
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args(argv)
    compute_influence(args.test_indices, args.k)


if __name__ == "__main__":
    main()
