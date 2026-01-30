"""
Compare Halides split (train/test) against OOD Halides.

Inputs:
- data/split_by_category/Halides.train.json
- data/split_by_category/Halides.test.json
- splits_feature_autoood/ood.csv (filtered to category == Halides)

Outputs:
- splits_feature_autoood/halides_train_test_vs_ood.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data" / "split_by_category"
OOD_PATH = ROOT / "ood.csv"
OUT_FIG = ROOT / "halides_train_test_vs_ood.png"


def load_halide_split(split: str) -> pd.Series:
    """Load Halides split and return band gaps as a Series."""
    path = DATA_ROOT / f"Halides.{split}.json"
    with open(path) as f:
        data = json.load(f)  # mpid -> {"bg": val}
    return pd.Series({k: v["bg"] for k, v in data.items()}, name=f"bg_{split}")


def load_ood_halides() -> pd.DataFrame:
    """Load OOD file and keep only Halides rows."""
    ood_df = pd.read_csv(OOD_PATH)
    ood_hal = ood_df[ood_df["category"] == "Halides"].copy()
    return ood_hal


def main():
    train_bg = load_halide_split("train")
    test_bg = load_halide_split("test")
    ood_hal = load_ood_halides()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Train vs test (Halides split)
    bins = 25
    axes[0].hist(train_bg, bins=bins, density=True, alpha=0.6, color="tab:blue", label=f"train (n={len(train_bg)})")
    axes[0].hist(test_bg, bins=bins, density=True, alpha=0.6, color="tab:orange", label=f"test (n={len(test_bg)})")
    axes[0].set_xlabel("Band gap (eV)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Halides band-gap distribution (train vs test)")
    axes[0].legend()

    # Halide test vs OOD Halides (using experimental band gap if available)
    axes[1].hist(test_bg, bins=bins, density=True, alpha=0.6, color="tab:blue", label=f"Halides test (n={len(test_bg)})")
    axes[1].hist(
        ood_hal["bg_exp"],
        bins=bins,
        density=True,
        alpha=0.6,
        color="tab:green",
        label=f"OOD Halides (n={len(ood_hal)})",
    )
    axes[1].set_xlabel("Band gap (eV)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Halides test vs OOD Halides (experimental band gap)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    print(f"Saved figure to {OUT_FIG}")
    print(f"Counts: train={len(train_bg)}, test={len(test_bg)}, ood_halides={len(ood_hal)}")


if __name__ == "__main__":
    main()
