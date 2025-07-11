#!/usr/bin/env python
"""
Compare First-Order vs TracIn ranked lists → Kendall τ, Spearman ρ, Jaccard.
Also plot score drop-off curves.
"""
import argparse, json, pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

RESULT_D = Path("results")
OUT_D    = RESULT_D / "analysis"
plt.switch_backend("Agg")  # headless

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--test-indices", nargs="+", type=int, default=[0])
    p.add_argument("--k", type=int, default=20)
    return p.parse_args()

def load(method, idx):
    return pd.read_csv(
        RESULT_D / f"influence_{method}" / f"test_{idx}.csv"
    ).head(k)

def main():
    args = parse()
    OUT_D.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx in args.test_indices:
        fo = load("first_order", idx)
        tr = load("tracin", idx)

        # Align by train_index for correlation
        merged = fo.merge(tr, on="train_index", suffixes=("_fo", "_tr"))
        tau, _  = kendalltau(merged["rank_fo"], merged["rank_tr"])
        rho, _  = spearmanr(merged["rank_fo"], merged["rank_tr"])
        jaccard = len(set(fo.train_index) & set(tr.train_index)) / args.k

        summary.append(
            {"test_index": idx, "kendall_tau": tau,
             "spearman_rho": rho, "jaccard_topk": jaccard}
        )

        # plot drop-off
        for m, df in zip(["first_order", "tracin"], [fo, tr]):
            plt.figure()
            plt.plot(df["rank"], df["score"])
            plt.xlabel("Rank"); plt.ylabel("Influence score")
            plt.title(f"{m} — test {idx}")
            plt.savefig(OUT_D / f"{m}_test{idx}.png")
            plt.close()

    with open(OUT_D / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(pd.DataFrame(summary))

if __name__ == "__main__":
    main()
