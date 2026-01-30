#!/usr/bin/env python3
"""
Parse pretrain logs and plot Test MRAE for each model (single-run, no error bars).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Maps log filename prefix -> friendly label
LABEL_MAP = {
    "alignn": "ALIGNN-Z",
    "alignn_prop": "ALIGNN-Prop",
    "cartnet": "CARTNet",
    "cgcnn": "CGCNN",
    "chgnet": "CHGNet-Z",
    "chgnet_prop": "CHGNet-Prop",
    "leftnet_z": "LEFTNet-Z",
    "leftnet_prop": "LEFTNet-Prop",
}

PLOT_ORDER = [
    "alignn",
    "alignn_prop",
    "cartnet",
    "cgcnn",
    "chgnet",
    "chgnet_prop",
    "leftnet_z",
    "leftnet_prop",
]

# Colors aligned to earlier plots
COLOR_MAP = {
    "alignn": "#9467bd",
    "alignn_prop": "#8c564b",
    "cartnet": "#d62728",
    "cgcnn": "#2ca02c",
    "chgnet": "#e377c2",
    "chgnet_prop": "#7f7f7f",
    "leftnet_z": "#bcbd22",
    "leftnet_prop": "#17becf",
}

TEST_MRE_RE = re.compile(r"'test_mre':\s*([0-9eE+\-.]+)")


def extract_test_mre(log_path: Path) -> float | None:
    text = log_path.read_text()
    m = TEST_MRE_RE.search(text)
    return float(m.group(1)) if m else None


def load_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for log_file in sorted(log_dir.glob("*_pretrain.log")):
        stem = log_file.name.replace("_pretrain.log", "")
        if stem not in LABEL_MAP:
            continue
        mre = extract_test_mre(log_file)
        if mre is None:
            print(f"[WARN] No test_mre found in {log_file}")
            continue
        rows.append(
            {
                "method": stem,
                "Model": LABEL_MAP[stem],
                "Mean": mre,
            }
        )
    return pd.DataFrame(rows)


def compute_ylim(values: pd.Series, y_min=None, y_max=None):
    if y_min is not None and y_max is not None:
        return (y_min, y_max)
    vmin, vmax = values.min(), values.max()
    span = max(0.0, vmax - vmin)
    pad = max(0.01, 0.15 * span)
    return (max(0.0, vmin - pad), vmax + pad)


def plot(df: pd.DataFrame, outfile: Path, title: str, y_min=None, y_max=None):
    if df.empty:
        raise SystemExit("No data to plot.")

    plot_order = [m for m in PLOT_ORDER if m in df["method"].values]
    model_to_pos = {m: i for i, m in enumerate(plot_order)}

    plt.figure(figsize=(10, 4))
    for _, row in df.iterrows():
        m = row["method"]
        x = model_to_pos[m]
        plt.errorbar(
            x,
            row["Mean"],
            yerr=0,
            fmt="o",
            markersize=7,
            capsize=0,
            color=COLOR_MAP[m],
            linestyle="none",
        )

    xticks = list(range(len(plot_order)))
    xticklabels = [LABEL_MAP[m] for m in plot_order]
    plt.xticks(xticks, xticklabels, rotation=45, ha="right")
    plt.ylabel("Test MRAE")
    plt.ylim(compute_ylim(df["Mean"], y_min, y_max))
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    present_models = set(df["method"])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=COLOR_MAP[m], label=LABEL_MAP[m])
        for m in plot_order
        if m in present_models
    ]
    plt.legend(
        handles=handles,
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved plot to {outfile}")


def main():
    ap = argparse.ArgumentParser(description="Plot pretrain test MRAE per model.")
    ap.add_argument("--log-dir", default="output/pretrain", help="Directory containing *_pretrain.log files.")
    ap.add_argument("--outfile", default="output/pretrain_mrae.pdf", help="Output figure path.")
    ap.add_argument("--title", default="Pretrain Test MRAE", help="Figure title.")
    ap.add_argument("--y-min", type=float, default=None, help="Optional fixed y-axis min.")
    ap.add_argument("--y-max", type=float, default=None, help="Optional fixed y-axis max.")
    args = ap.parse_args()

    df = load_logs(Path(args.log_dir))
    plot(df, Path(args.outfile), args.title, y_min=args.y_min, y_max=args.y_max)


if __name__ == "__main__":
    main()
