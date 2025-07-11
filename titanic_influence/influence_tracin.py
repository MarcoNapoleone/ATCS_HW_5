"""Compute TracIn influence scores using logistic regression.

The implementation is a simplified variant of the TracIn method: the influence
of a training example is defined as the dot product between its gradient and the
gradient of the test example, both evaluated at the final trained parameters.
The scores are saved under ``results/influence_tracin``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_D = Path("data")
OUT_D = Path("results") / "influence_tracin"


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


def _gradients(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)[:, 1]
    return (p - y)[:, None] * _add_bias(X)


def compute_influence(test_indices: list[int], k: int = 20, output_dir: Path | None = None) -> dict[int, pd.DataFrame]:
    """Compute influences and return a mapping ``{index: df}``."""
    X_tr, y_tr, X_te, y_te = _load_data()
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    model.fit(X_tr, y_tr)

    grads_train = _gradients(model, X_tr, y_tr)

    output_dir = OUT_D if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for idx in test_indices:
        grad_t = _gradients(model, X_te[idx : idx + 1], y_te[idx : idx + 1])[0]
        scores = grads_train @ grad_t
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
