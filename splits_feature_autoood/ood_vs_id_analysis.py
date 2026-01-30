"""
Visualize distribution differences between ID and OOD splits.

Generates comparison plots for:
- mp band gap and experimental band gap
- material category distribution
- number of elements per formula
- Herfindahl–Hirschman Index (HHI) of stoichiometric concentration
- source distribution
- auto-cluster assignment

Output: splits_feature_autoood/ood_vs_id_distributions.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    from pymatgen.core.composition import Composition
except Exception as exc:  # pragma: no cover - defensive guard for missing dependency
    raise SystemExit("pymatgen is required to compute HHI") from exc


ROOT = Path(__file__).resolve().parent
ID_PATH = ROOT / "id.csv"
OOD_PATH = ROOT / "ood.csv"
OUT_FIG = ROOT / "ood_vs_id_distributions.png"


def hhi_from_formula(formulas: pd.Series) -> pd.Series:
    """Compute stoichiometric HHI from reduced formulas."""
    hhi_vals = []
    for f in formulas:
        comp = Composition(f)
        frac = comp.fractional_composition
        hhi_vals.append(sum(val**2 for val in frac.values()))
    return pd.Series(hhi_vals, index=formulas.index)


def percent_counts(series: pd.Series) -> pd.Series:
    """Return percentage counts sorted by descending frequency."""
    return (series.value_counts(normalize=True) * 100).sort_values(ascending=False)


def overlay_hist(ax, data, label, color, bins=30):
    ax.hist(data, bins=bins, density=True, alpha=0.55, label=label, color=color)


def bar_percent(ax, series, label, color, width=0.35, offset=0.0):
    x = range(len(series))
    ax.bar([i + offset for i in x], series.values, width=width, label=label, color=color)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(series.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter())


def main():
    id_df = pd.read_csv(ID_PATH)
    ood_df = pd.read_csv(OOD_PATH)

    # Compute HHI
    id_df["hhi"] = hhi_from_formula(id_df["formula"])
    ood_df["hhi"] = hhi_from_formula(ood_df["formula"])

    # Percent distributions
    cat_id = percent_counts(id_df["category"])
    cat_ood = percent_counts(ood_df["category"])
    cats = cat_id.add(cat_ood, fill_value=0).sort_values(ascending=False).index
    cat_id = cat_id.reindex(cats, fill_value=0)
    cat_ood = cat_ood.reindex(cats, fill_value=0)

    ne_id = percent_counts(id_df["num_elements"]).sort_index()
    ne_ood = percent_counts(ood_df["num_elements"]).reindex(ne_id.index, fill_value=0)

    nc_id = percent_counts(id_df["num_categories"]).sort_index()
    nc_ood = percent_counts(ood_df["num_categories"]).reindex(nc_id.index, fill_value=0)

    src_id = percent_counts(id_df["source"])
    src_ood = percent_counts(ood_df["source"])
    sources = src_id.add(src_ood, fill_value=0).sort_values(ascending=False).index
    src_id = src_id.reindex(sources, fill_value=0)
    src_ood = src_ood.reindex(sources, fill_value=0)

    clu_id = percent_counts(id_df["cluster_auto"])
    clu_ood = percent_counts(ood_df["cluster_auto"])
    clusters = clu_id.add(clu_ood, fill_value=0).sort_values(ascending=False).index
    clu_id = clu_id.reindex(clusters, fill_value=0)
    clu_ood = clu_ood.reindex(clusters, fill_value=0)

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    (ax_bg_mp, ax_bg_exp), (ax_cat, ax_ne), (ax_hhi, ax_src) = axes

    # Band gaps (MP)
    overlay_hist(ax_bg_mp, id_df["bg_mp"], "ID bg_mp", "tab:blue")
    overlay_hist(ax_bg_mp, ood_df["bg_mp"], "OOD bg_mp", "tab:orange")
    ax_bg_mp.set_xlabel("Band gap (MP, eV)")
    ax_bg_mp.set_ylabel("Density")
    ax_bg_mp.legend()
    ax_bg_mp.set_title("Band gap (MP) distribution")

    # Band gaps (experimental)
    overlay_hist(ax_bg_exp, id_df["bg_exp"], "ID bg_exp", "tab:blue")
    overlay_hist(ax_bg_exp, ood_df["bg_exp"], "OOD bg_exp", "tab:orange")
    ax_bg_exp.set_xlabel("Band gap (exp, eV)")
    ax_bg_exp.set_ylabel("Density")
    ax_bg_exp.legend()
    ax_bg_exp.set_title("Band gap (exp) distribution")

    # Category percentages
    bar_percent(ax_cat, cat_id, "ID", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_cat, cat_ood, "OOD", "tab:orange", width=0.4, offset=0.2)
    ax_cat.set_ylabel("Percent")
    ax_cat.legend()
    ax_cat.set_title("Category distribution (%)")

    # Number of elements
    bar_percent(ax_ne, ne_id, "ID", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_ne, ne_ood, "OOD", "tab:orange", width=0.4, offset=0.2)
    ax_ne.set_ylabel("Percent")
    ax_ne.legend()
    ax_ne.set_title("Number of elements per formula (%)")

    # HHI
    overlay_hist(ax_hhi, id_df["hhi"], "ID HHI", "tab:blue")
    overlay_hist(ax_hhi, ood_df["hhi"], "OOD HHI", "tab:orange")
    ax_hhi.set_xlabel("HHI (stoichiometry concentration)")
    ax_hhi.set_ylabel("Density")
    ax_hhi.legend()
    ax_hhi.set_title("HHI distribution")

    # Source and cluster stacked inset
    bar_percent(ax_src, src_id, "ID source", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(ax_src, src_ood, "OOD source", "tab:orange", width=0.4, offset=0.2)
    ax_src.set_ylabel("Percent")
    ax_src.legend()
    ax_src.set_title("Source distribution (%)")

    # Add cluster distribution as inset for clarity
    inset = ax_src.inset_axes([0.55, 0.5, 0.4, 0.45])
    bar_percent(inset, clu_id, "ID cluster", "tab:blue", width=0.4, offset=-0.2)
    bar_percent(inset, clu_ood, "OOD cluster", "tab:orange", width=0.4, offset=0.2)
    inset.set_title("Cluster (%)", fontsize=9)
    inset.tick_params(axis="x", labelrotation=45, labelsize=8)
    inset.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    print(f"Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
