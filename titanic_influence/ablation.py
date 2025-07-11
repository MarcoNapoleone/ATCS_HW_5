#!/usr/bin/env python
"""
Remove top-k influential rows (selected by --method) and retrain.
"""
import argparse, glob, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from titanic_influence.utils import build_dataset, build_model

DATA_D   = Path("data")
BASE_MET = json.load(open("results/metrics.json"))
OUT_ROOT = Path("results/ablation")

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["first_order", "tracin"], required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--test-indices", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()

def rows_to_drop(method, idx, k):
    df = pd.read_csv(f"results/influence_{method}/test_{idx}.csv").head(k)
    return df["train_index"].tolist()

def main():
    args = parse()
    drop = set()
    for idx in args.test_indices:
        drop.update(rows_to_drop(args.method, idx, args.k))

    # Load data
    train_df = pd.read_csv(DATA_D / "titanic_train.csv").drop(index=drop)
    test_df  = pd.read_csv(DATA_D / "titanic_test.csv")
    ct       = joblib.load(DATA_D / "preprocess_ct.joblib")

    X, y = build_dataset(train_df, ct, fit=False)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    model = build_model(X.shape[1])
    cb_es = tf.keras.callbacks.EarlyStopping(patience=4,
                                             restore_best_weights=True)
    model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=args.batch_size,
              verbose=2,
              callbacks=[cb_es])

    # Evaluate
    X_test, y_test = build_dataset(test_df, ct, fit=False)
    _, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    metrics = {"accuracy": float(acc), "auc": float(auc)}

    out_d = OUT_ROOT / f"{args.method}_k{args.k}"
    out_d.mkdir(parents=True, exist_ok=True)
    model.save(out_d / "model.h5")
    with open(out_d / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("--- Baseline vs Ablation ---")
    print("Baseline:", BASE_MET)
    print("Ablated :", metrics)

if __name__ == "__main__":
    main()
