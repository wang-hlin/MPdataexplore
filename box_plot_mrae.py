#!/usr/bin/env python3
"""
Plot test MRAE distributions for train-from-scratch vs. finetune runs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHOD_LABELS = {
    "alignn": "ALIGNN-Z",
    "alignn_z": "ALIGNN-Z",
    "alignn_prop": "ALIGNN-Prop",
    "cgcnn": "CGCNN",
    "cartnet": "CartNet",
    "chgnet": "CHGNet-Z",
    "chgnet_prop": "CHGNet-Prop",
    "leftnet_z": "LEFTNet-Z",
    "leftnet_prop": "LEFTNet-Prop",
    "linear_regression": "LR",
    "random_forest": "RF",
    "svm": "SVR",
}

MODEL_ORDER = [
    "ALIGNN-Z",
    "ALIGNN-Prop",
    "CGCNN",
    "CartNet",
    "CHGNet-Z",
    "CHGNet-Prop",
    "LEFTNet-Z",
    "LEFTNet-Prop",
    "LR",
    "RFR",
    "SVR",
]


def load_source(df: pd.DataFrame, split_dir: str, source_label: str):
    subset = df[df["split_dir"] == split_dir].copy()
    subset = subset.dropna(subset=["test_mre"])
    subset["Model"] = subset["method"].map(METHOD_LABELS)
    subset = subset[subset["Model"].notna()]
    subset["Source"] = source_label
    subset = subset.rename(columns={"test_mre": "Score"})
    return subset[["Model", "Score", "Source"]]


def main(metrics_csv="output/metrics_summary.csv", outfile="output/box_plot_mrae.pdf"):
    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise SystemExit(f"Metrics file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    scratch = load_source(df, "train_from_scratch", "From Scratch")
    finetune = load_source(df, "finetune", "Pretrained")

    combined = pd.concat([scratch, finetune], ignore_index=True)
    if combined.empty:
        raise SystemExit("No data available for the requested splits.")

    combined["Model"] = pd.Categorical(combined["Model"], categories=MODEL_ORDER, ordered=True)
    combined = combined.dropna(subset=["Model"])

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=combined,
        x="Model",
        y="Score",
        hue="Source",
        palette="Set2",
        width=0.6,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Test MRAE")
    ax.set_xlabel("Model")
    ax.set_ylim(0.2, 0.45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title=None)
    plt.tight_layout()

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved box plot to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot box plot comparing pretrained vs scratch MRAE.")
    ap.add_argument("--metrics-csv", default="output/metrics_summary.csv")
    ap.add_argument("--outfile", default="output/box_plot_mrae.pdf")
    args = ap.parse_args()
    main(metrics_csv=args.metrics_csv, outfile=args.outfile)
