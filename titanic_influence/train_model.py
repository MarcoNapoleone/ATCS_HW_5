#!/usr/bin/env python
"""
Train MLP, save best model + metrics + checkpoints.
"""
import json, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from titanic_influence.utils import build_dataset, build_model

DATA_D   = Path("data")
RESULT_D = Path("results")
CP_D     = RESULT_D / "checkpoints"
CP_D.mkdir(parents=True, exist_ok=True)

def main(epochs=40, batch_size=128):
    train_df = pd.read_csv(DATA_D / "titanic_train.csv")
    test_df  = pd.read_csv(DATA_D / "titanic_test.csv")
    ct       = joblib.load(DATA_D / "preprocess_ct.joblib")

    X, y = build_dataset(train_df, ct, fit=False)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    model = build_model(X.shape[1])

    cb_es = tf.keras.callbacks.EarlyStopping(patience=4,
                                             restore_best_weights=True)
    cb_cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CP_D / "cp-{epoch:02d}.h5"),
        save_weights_only=False,
        save_best_only=False
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_es, cb_cp],
        verbose=2
    )

    # Final test metrics
    X_test, y_test = build_dataset(test_df, ct, fit=False)
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    metrics = {"accuracy": float(acc), "auc": float(auc)}
    (RESULT_D / "titanic_model.h5").write_bytes(
        model.to_json().encode()
    )  # light save; full weights kept in checkpoints
    with open(RESULT_D / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics:", metrics)

if __name__ == "__main__":
    import tensorflow as tf
    main()
