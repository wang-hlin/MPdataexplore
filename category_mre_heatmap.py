#!/usr/bin/env python3
"""
Generate an MRE heatmap for the category split using metrics_summary.csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

# Desired GNN order for the main figure
METHOD_ORDER = [
    "cgcnn",
    "cartnet",
    "alignn_z",
    "alignn_prop",
    "chgnet",
    "chgnet_prop",
    "leftnet_z",
    "leftnet_prop",
]

BASELINE_METHODS = ["linear_regression", "random_forest", "svm"]


def friendly_method_name(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def derive_label(row):
    split_name = str(row.get("split_name") or "").strip()
    if split_name:
        return split_name
    token = str(row.get("fold_token") or "").strip()
    if token:
        return f"Fold {token}"
    idx = row.get("fold_idx")
    if pd.notna(idx):
        return f"Fold {int(idx)}"
    return None


def build_heatmap(
    df: pd.DataFrame,
    metric: str,
    phase: str,
    split_dir: str,
    methods=None,
    split_names=None,
) -> pd.DataFrame:
    metric_col = f"{phase}_{metric}"
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in metrics file")

    cat_df = df[df["split_dir"] == split_dir].copy()
    if split_names:
        split_set = {s.strip() for s in split_names}
        cat_df = cat_df[cat_df["split_name"].isin(split_set)]
    cat_df["label"] = cat_df.apply(derive_label, axis=1)
    cat_df = cat_df[cat_df["label"].notna()]

    if methods:
        cat_df = cat_df[cat_df["method"].isin(methods)]
        if cat_df.empty:
            raise ValueError("No rows left after filtering by methods")

    pivot = (
        cat_df.pivot_table(
            index="label",
            columns="method",
            values=metric_col,
            aggfunc="mean",
        )
        .sort_index()
    )
    pivot = pivot.sort_index(axis=1)

    # Column ordering
    order_priority = methods if methods else METHOD_ORDER
    if order_priority:
        ordered_cols = [c for c in order_priority if c in pivot.columns]
        remaining = [c for c in pivot.columns if c not in ordered_cols]
        pivot = pivot[ordered_cols + sorted(remaining)]

    pivot = pivot.rename(columns={c: friendly_method_name(c) for c in pivot.columns})
    return pivot


def plot_heatmap(pivot: pd.DataFrame, metric: str, phase: str, outfile: Path):
    plt.figure(figsize=(1.6 + 0.8 * len(pivot.columns), 1.2 + 0.5 * len(pivot.index)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": f"{phase.title()} {metric.upper()}"},
    )
    plt.xlabel("Method")
    plt.ylabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot category heatmap for metrics_summary.csv")
    ap.add_argument("--metrics-csv", default="output/metrics_summary.csv")
    ap.add_argument("--metric", default="mre", help="metric suffix (e.g., mae, mre, r2)")
    ap.add_argument("--phase", choices=["val", "test"], default="test")
    ap.add_argument("--split-dir", default="category", help="split directory name (e.g., category, ood)")
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="optional list of methods to keep; default = GNN methods in fixed order",
    )
    ap.add_argument(
        "--include-traditional-ml",
        action="store_true",
        help="Include Linear Regression, Random Forest, and SVM in the heatmap.",
    )
    ap.add_argument(
        "--split-names",
        nargs="*",
        default=None,
        help="optional list of split_name values to include (e.g., chemsys composition)",
    )
    ap.add_argument("--outfile", default="output/category_mre_heatmap.pdf")
    args = ap.parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # Decide which methods to pass into build_heatmap
    methods = args.methods
    if methods is None:
        # Default: your desired GNN order
        methods = METHOD_ORDER.copy()
        # Optionally append baselines at the end
        if args.include_traditional_ml:
            methods += BASELINE_METHODS
    else:
        # If user specified methods explicitly and did NOT request baselines,
        # strip out the baseline ones.
        if not args.include_traditional_ml:
            methods = [m for m in methods if m not in BASELINE_METHODS]

    pivot = build_heatmap(
        df,
        args.metric,
        args.phase,
        args.split_dir,
        methods=methods,
        split_names=args.split_names,
    )
    if pivot.empty:
        raise SystemExit("No data found for requested filters.")
    plot_heatmap(pivot, args.metric, args.phase, Path(args.outfile))
    print(f"[OK] Saved heatmap to {args.outfile}")


if __name__ == "__main__":
    main()