"""
Replicate ood_vs_id visualizations for Halides train vs test splits.

Data sources:
- data/split_by_category/Halides.train.json (mpid -> {"bg": value})
- data/split_by_category/Halides.test.json
- splits_feature_autoood/id.csv and ood.csv for metadata (bg_mp, bg_exp, formula, etc.)

Output:
- splits_feature_autoood/halides_train_test_distributions.png
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    from pymatgen.core.composition import Composition
except Exception as exc:  # pragma: no cover
    raise SystemExit("pymatgen is required to compute HHI") from exc


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data" / "split_by_category"
ID_PATH = ROOT / "id.csv"
OOD_PATH = ROOT / "ood.csv"
OUT_FIG = ROOT / "halides_train_test_distributions.png"


def load_halide_split(split: str) -> pd.DataFrame:
    """Load Halides split json (mpid -> {bg})."""
    path = DATA_ROOT / f"Halides.{split}.json"
    with open(path) as f:
        data = json.load(f)
    rows = [{"mpid": k, "bg": v["bg"], "split": split} for k, v in data.items()]
    return pd.DataFrame(rows)


def hhi_from_formula(formulas: pd.Series) -> pd.Series:
    vals = []
    for f in formulas:
        if pd.isna(f):
            vals.append(None)
            continue
        comp = Composition(f)
        frac = comp.fractional_composition
        vals.append(sum(val**2 for val in frac.values()))
    return pd.Series(vals, index=formulas.index)


def percent_counts(series: pd.Series) -> pd.Series:
    return (series.value_counts(normalize=True) * 100).sort_values(ascending=False)


def overlay_hist(ax, data, label, color, bins=30):
    ax.hist(data.dropna(), bins=bins, density=True, alpha=0.6, label=label, color=color)


def bar_percent(ax, series, label, color, width=0.35, offset=0.0):
    x = range(len(series))
    ax.bar([i + offset for i in x], series.values, width=width, label=label, color=color)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(series.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter())


def main():
    # Load halide splits
    train_df = load_halide_split("train")
    test_df = load_halide_split("test")
    hal_df = pd.concat([train_df, test_df], ignore_index=True)

    # Load metadata (ID+OOD) to enrich with formulas, bg_mp/bg_exp, etc.
    meta = pd.concat([pd.read_csv(ID_PATH), pd.read_csv(OOD_PATH)], ignore_index=True)
    meta = meta.drop_duplicates(subset="mpids").set_index("mpids")
    hal_df = hal_df.merge(meta, left_on="mpid", right_index=True, how="left")

    # Compute HHI
    hal_df["hhi"] = hhi_from_formula(hal_df["formula"])

    # Split back into train/test
    train = hal_df[hal_df["split"] == "train"]
    test = hal_df[hal_df["split"] == "test"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    (ax_bg_mp, ax_bg_exp), (ax_cat, ax_ne), (ax_hhi, ax_src) = axes

    bins = 30

    # Band gaps (MP)
    overlay_hist(ax_bg_mp, train["bg_mp"], f"train (n={len(train)})", "tab:blue", bins=bins)
    overlay_hist(ax_bg_mp, test["bg_mp"], f"test (n={len(test)})", "tab:orange", bins=bins)
    ax_bg_mp.set_xlabel("Band gap (MP, eV)")
    ax_bg_mp.set_ylabel("Density")
    ax_bg_mp.set_title("Halides band gap (MP) distribution")
    ax_bg_mp.legend()

    # Band gaps (experimental)
    overlay_hist(ax_bg_exp, train["bg_exp"], f"train (n={len(train)})", "tab:blue", bins=bins)
    overlay_hist(ax_bg_exp, test["bg_exp"], f"test (n={len(test)})", "tab:orange", bins=bins)
    ax_bg_exp.set_xlabel("Band gap (exp, eV)")
    ax_bg_exp.set_ylabel("Density")
    ax_bg_exp.set_title("Halides band gap (exp) distribution")
    ax_bg_exp.legend()

    # Category percentages (should be all Halides)
    cat_train = percent_counts(train["category"])
    cat_test = percent_counts(test["category"]).reindex(cat_train.index, fill_value=0)
    bar_percent(ax_cat, cat_train, "train", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_cat, cat_test, "test", "tab:orange", width=0.4, offset=0.2)
    ax_cat.set_ylabel("Percent")
    ax_cat.set_title("Category distribution (%)")
    ax_cat.legend()

    # Number of elements
    ne_train = percent_counts(train["num_elements"]).sort_index()
    ne_test = percent_counts(test["num_elements"]).reindex(ne_train.index, fill_value=0)
    bar_percent(ax_ne, ne_train, "train", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_ne, ne_test, "test", "tab:orange", width=0.4, offset=0.2)
    ax_ne.set_ylabel("Percent")
    ax_ne.set_title("Number of elements per formula (%)")
    ax_ne.legend()

    # HHI
    overlay_hist(ax_hhi, train["hhi"], "train", "tab:blue", bins=bins)
    overlay_hist(ax_hhi, test["hhi"], "test", "tab:orange", bins=bins)
    ax_hhi.set_xlabel("HHI (stoichiometry concentration)")
    ax_hhi.set_ylabel("Density")
    ax_hhi.set_title("HHI distribution")
    ax_hhi.legend()

    # Source and cluster with inset
    src_train = percent_counts(train["source"])
    src_test = percent_counts(test["source"])
    sources = src_train.add(src_test, fill_value=0).sort_values(ascending=False).index
    src_train = src_train.reindex(sources, fill_value=0)
    src_test = src_test.reindex(sources, fill_value=0)
    bar_percent(ax_src, src_train, "train", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_src, src_test, "test", "tab:orange", width=0.4, offset=0.2)
    ax_src.set_ylabel("Percent")
    ax_src.set_title("Source distribution (%)")
    ax_src.legend()

    clu_train = percent_counts(train["cluster_auto"])
    clu_test = percent_counts(test["cluster_auto"])
    clusters = clu_train.add(clu_test, fill_value=0).sort_values(ascending=False).index
    clu_train = clu_train.reindex(clusters, fill_value=0)
    clu_test = clu_test.reindex(clusters, fill_value=0)
    inset = ax_src.inset_axes([0.55, 0.5, 0.4, 0.45])
    bar_percent(inset, clu_train, "train", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(inset, clu_test, "test", "tab:orange", width=0.4, offset=0.2)
    inset.set_title("Cluster (%)", fontsize=9)
    inset.tick_params(axis="x", labelrotation=45, labelsize=8)
    inset.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    print(f"Saved figure to {OUT_FIG}")
    missing = hal_df["formula"].isna().sum()
    if missing:
        print(f"Warning: {missing} entries missing metadata; they are included but may drop from some plots.")


if __name__ == "__main__":
    main()
