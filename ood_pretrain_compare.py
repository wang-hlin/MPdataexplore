#!/usr/bin/env python3
"""
Plot OOD Test MRAE (mean ± std) for pretrained vs. non-pretrained runs.

Reads aggregated metrics (output/metrics_agg.csv by default) and creates a
two-panel figure comparing the OOD split (`split_dir == ood`) against the
non-pretrained counterpart (`split_dir == ood_no_pretrain`).
Optionally overlays a baseline point showing the mean relative difference
between computational and experimental band gaps for the OOD cluster
(`splits_feature_autoood/ood.csv` by default).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

LABEL_MAP = {
    "alignn_prop": "ALIGNN-Prop",
    "alignn_z": "ALIGNN-Z",
    "cartnet": "CARTNet",
    "cgcnn": "CGCNN",
    "chgnet": "CHGNet-Z",
    "chgnet_prop": "CHGNet-Prop",
    "leftnet_prop": "LEFTNet-Prop",
    "leftnet_z": "LEFTNet-Z",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "svm": "SVR",
}

# Methods to drop from the plot entirely.
EXCLUDE_METHODS = {"random_forest", "svm"}

# Tab10 + a few extras for consistent colors across both panels.
COLOR_MAP = {
    "cgcnn": "#2ca02c",
    "cartnet": "#d62728",
    "alignn_z": "#9467bd",
    "alignn_prop": "#8c564b",
    "chgnet": "#e377c2",
    "chgnet_prop": "#7f7f7f",
    "leftnet_z": "#bcbd22",
    "leftnet_prop": "#17becf",
    "linear_regression": "#4c4c4c",
}

DEFAULT_ORDER = [
    "cgcnn",
    "cartnet",
    "alignn_z",
    "alignn_prop",
    "chgnet",
    "chgnet_prop",
    "leftnet_z",
    "leftnet_prop",
]

BASELINE_KEY = "__baseline__"
BASELINE_LABEL = "Baseline (DFT vs Exp OOD rel. diff)"
BASELINE_COLOR = "black"
Y_LIM = (0.25, 0.8)


def build_order(include_linear: bool) -> list[str]:
    order = [m for m in DEFAULT_ORDER if m not in EXCLUDE_METHODS]
    if include_linear and "linear_regression" not in order:
        order.insert(0, "linear_regression")
    return order


def prepare_split(df: pd.DataFrame, split_dir: str, include_linear: bool) -> pd.DataFrame:
    subset = df[df["split_dir"] == split_dir].copy()
    subset = subset[~subset["method"].isin(EXCLUDE_METHODS)]
    if not include_linear:
        subset = subset[subset["method"] != "linear_regression"]
    subset = subset.dropna(subset=["test_mre_mean"])
    subset["Mean"] = pd.to_numeric(subset["test_mre_mean"], errors="coerce")
    subset["Std"] = pd.to_numeric(subset["test_mre_std"], errors="coerce").fillna(0)
    subset["Model"] = subset["method"].map(LABEL_MAP)
    subset = subset[subset["Model"].notna()]
    return subset[["method", "Model", "Mean", "Std"]]


def compute_ylim(dfs: list[pd.DataFrame]) -> tuple[float, float]:
    return Y_LIM


def plot_single_setting(ax, df_setting, plot_order, y_lim, title_suffix, baseline=None, baseline_std=None):
    model_to_pos = {m: i for i, m in enumerate(plot_order)}
    for _, row in df_setting.iterrows():
        m = row["method"]
        x = model_to_pos[m]
        ax.errorbar(
            x,
            row["Mean"],
            yerr=row["Std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=COLOR_MAP[m],
            linestyle="none",
        )

    if baseline is not None and BASELINE_KEY in model_to_pos:
        x = model_to_pos[BASELINE_KEY]
        ax.errorbar(
            x,
            baseline,
            yerr=baseline_std if baseline_std is not None else 0,
            fmt="o",
            markersize=7,
            capsize=3,
            color=BASELINE_COLOR,
            linestyle="none",
        )

    xticks = list(range(len(plot_order)))
    xticklabels = [LABEL_MAP.get(m, BASELINE_LABEL if m == BASELINE_KEY else m) for m in plot_order]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_ylim(*y_lim)
    ax.set_ylabel("Test MRAE")
    ax.set_title(title_suffix)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def add_legend(fig, plot_order, present_models, baseline_label=None):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=COLOR_MAP[m],
            label=LABEL_MAP[m],
        )
        for m in plot_order
        if m in present_models
    ]
    if baseline_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=BASELINE_COLOR,
                label=baseline_label,
            )
        )
    fig.legend(
        handles=handles,
        title="Model",
        bbox_to_anchor=(0.98, 0.98),
        loc="upper left",
        borderaxespad=0.0,
    )


def load_baseline(ood_csv: Path) -> tuple[float, float]:
    """
    Compute mean and std of the relative difference between computational
    and experimental band gaps for the OOD cluster.
    """
    df = pd.read_csv(ood_csv)
    if not {"bg_mp", "bg_exp"}.issubset(df.columns):
        raise ValueError(f"Required columns bg_mp and bg_exp not found in {ood_csv}")
    rel_diff = (df["bg_mp"] - df["bg_exp"]).abs() / df["bg_exp"].abs()
    rel_diff = rel_diff.dropna()
    if rel_diff.empty:
        raise ValueError("No valid rows to compute baseline difference.")
    return rel_diff.mean(), rel_diff.std()


def main(metrics_csv: str, outfile: str, include_linear: bool, ood_csv: str | None, with_baseline: bool):
    metrics_path = Path(metrics_csv)
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    pretrain = prepare_split(df, "ood", include_linear)
    scratch = prepare_split(df, "ood_no_pretrain", include_linear)
    if pretrain.empty or scratch.empty:
        raise SystemExit("No data found for requested OOD splits.")

    plot_order = build_order(include_linear)
    y_lim = compute_ylim([pretrain, scratch])
    present_models = set(pd.concat([pretrain, scratch])["method"])

    baseline = None
    baseline_std = None
    baseline_label = None
    if with_baseline and ood_csv:
        csv_path = Path(ood_csv)
        if csv_path.exists():
            try:
                baseline, baseline_std = load_baseline(csv_path)
                baseline_label = BASELINE_LABEL
            except Exception as exc:
                print(f"[WARN] Could not compute baseline from {csv_path}: {exc}")
        else:
            print(f"[WARN] OOD CSV not found at {csv_path}; skipping baseline line.")

    if baseline is not None:
        COLOR_MAP[BASELINE_KEY] = BASELINE_COLOR
        LABEL_MAP[BASELINE_KEY] = BASELINE_LABEL
        plot_order = plot_order + [BASELINE_KEY]
        present_models = present_models | {BASELINE_KEY}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_single_setting(
        axes[0],
        pretrain,
        plot_order,
        y_lim,
        "Pretrained (OOD)",
        baseline=baseline,
        baseline_std=baseline_std,
    )
    plot_single_setting(
        axes[1],
        scratch,
        plot_order,
        y_lim,
        "No Pretrain (OOD)",
        baseline=baseline,
        baseline_std=baseline_std,
    )
    add_legend(fig, plot_order, present_models, baseline_label=baseline_label)
    fig.tight_layout(rect=(0, 0, 0.92, 1))

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved OOD pretrain comparison to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot OOD MRAE for pretrained vs non-pretrained models.")
    ap.add_argument("--metrics-csv", default="output/metrics_agg.csv", help="Path to aggregated metrics CSV.")
    ap.add_argument("--outfile", default="output/ood_pretrain_vs_nonpretrain.pdf", help="Output figure path.")
    ap.add_argument(
        "--include-linear",
        action="store_true",
        help="Include linear regression (extreme outlier) in the plot.",
    )
    ap.add_argument(
        "--ood-csv",
        default="splits_feature_autoood/ood.csv",
        help="OOD cluster CSV containing bg_mp and bg_exp for baseline line.",
    )
    ap.add_argument(
        "--with-baseline",
        action="store_true",
        help="Overlay the baseline point for computational vs experimental gap difference.",
    )
    args = ap.parse_args()
    main(
        metrics_csv=args.metrics_csv,
        outfile=args.outfile,
        include_linear=args.include_linear,
        ood_csv=args.ood_csv,
        with_baseline=args.with_baseline,
    )
