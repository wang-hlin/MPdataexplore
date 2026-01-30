#!/usr/bin/env python3
"""
Plot a bar chart of test MRE per method for the OOD split.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

METHOD_LABELS = {
    "alignn_prop": "ALIGNN-Prop",
    "alignn_z": "ALIGNN-Z",
    "cartnet": "CARTNet",
    "cgcnn": "CGCNN",
    "chgnet": "CHGNet",
    "chgnet_prop": "CHGNet-Prop",
    "leftnet_prop": "LEFTNet-Prop",
    "leftnet_z": "LEFTNet-Z",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
}

METHOD_ORDER = [
    "linear_regression",
    "random_forest",
    "svm",
    "cgcnn",
    "cartnet",
    "alignn_z",
    "alignn_prop",
    "chgnet",
    "chgnet_prop",
]


def friendly(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def build_order(methods):
    priority = [m for m in METHOD_ORDER if m in methods]
    remaining = [m for m in methods if m not in priority]
    # keep remaining sorted for determinism
    return priority + sorted(remaining)


def main(metrics_csv="output/metrics_summary.csv", outfile="output/ood_mre_bar.pdf", max_y=None):
    path = Path(metrics_csv)
    if not path.exists():
        raise SystemExit(f"Metrics file not found: {path}")

    df = pd.read_csv(path)
    ood = df[df["split_dir"] == "ood"].copy()
    if ood.empty:
        raise SystemExit("No OOD rows found in metrics file.")

    agg = (
        ood.groupby("method")["test_mre"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .dropna(subset=["mean"])
    )
    if agg.empty:
        raise SystemExit("No test MRE values available for OOD split.")

    method_order = build_order(agg["method"].tolist())
    agg["label"] = agg["method"].map(friendly)
    agg["order"] = agg["method"].apply(lambda m: method_order.index(m))
    agg = agg.sort_values("order")

    fig, ax = plt.subplots(figsize=(1.2 * len(agg), 4.5))
    cmap = plt.get_cmap("viridis")
    norm_data = agg[agg["method"] != "linear_regression"]["mean"]
    norm = Normalize(vmin=norm_data.min(), vmax=norm_data.max())
    colors = cmap(norm(agg["mean"].values))
    x = range(len(agg))
    ax.bar(x, agg["mean"], color=colors)
    ax.errorbar(
        x=x,
        y=agg["mean"],
        yerr=agg["std"].fillna(0),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["label"], rotation=45, ha="right")
    ax.set_ylabel("Test MRE")
    ax.set_xlabel("Method")

    y_max = max_y if max_y is not None else (agg["mean"].max() * 1.05)
    if max_y is not None:
        ax.set_ylim(0, max_y)
    y_offset = 0.02 * y_max
    for xi, (_, row) in zip(x, agg.iterrows()):
        val = row["mean"]
        label = f"{val:.3f}"
        if row["method"] == "linear_regression":
            label = f"{val:.3e}"
        if max_y is not None and val > max_y:
            y_pos = max_y * 0.98
            va = "top"
            color = "red"
        else:
            y_pos = val + y_offset
            va = "bottom"
            color = "black"
        ax.text(
            xi,
            y_pos,
            label,
            ha="center",
            va=va,
            fontsize=8,
            rotation=90,
            color=color,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Test MRE", pad=0.01)
    fig.tight_layout()

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved OOD bar chart to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot OOD test MRE bar chart.")
    ap.add_argument("--metrics-csv", default="output/metrics_summary.csv")
    ap.add_argument("--outfile", default="output/ood_mre_bar.pdf")
    ap.add_argument("--max-y", type=float, default=None, help="Optional upper y-axis limit")
    args = ap.parse_args()
    main(metrics_csv=args.metrics_csv, outfile=args.outfile, max_y=args.max_y)
