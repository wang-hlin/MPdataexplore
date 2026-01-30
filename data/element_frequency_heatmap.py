#!/usr/bin/env python3
"""
Generate a periodic-table heatmap showing how often each element appears
in a materials dataset containing composition information.
"""

import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from pymatgen.core import Composition, Element


DEFAULT_DATA_PATH = Path("data/joint_dataset/combined_bandgap_data.csv")
DEFAULT_OUTPUT_PATH = Path("figs/element_frequency_heatmap.pdf")


def _iter_elements_from_formulas(
    formulas: Iterable[str],
) -> Iterable[Iterable[str]]:
    for formula in formulas:
        if not isinstance(formula, str):
            continue
        try:
            composition = Composition(formula)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"Skipping formula {formula!r}: {exc}")
            continue
        yield [el.symbol for el in composition.elements]


def _iter_elements_from_column(entries: Iterable) -> Iterable[Iterable[str]]:
    for entry in entries:
        if entry is None or (isinstance(entry, float) and np.isnan(entry)):
            continue
        if isinstance(entry, str):
            try:
                parsed = literal_eval(entry)
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"Could not parse element list {entry!r}: {exc}")
                continue
        else:
            parsed = entry
        if isinstance(parsed, (list, tuple)):
            yield [str(el).strip(" '\"") for el in parsed if el]


def load_element_frequencies(
    csv_path: Path,
    formula_column: Optional[str] = None,
    elements_column: Optional[str] = None,
) -> Counter:
    """Count how many formulas contain each element."""
    if not formula_column and not elements_column:
        raise ValueError("Provide at least formula_column or elements_column")

    df = pd.read_csv(csv_path)
    frequency = Counter()

    element_lists: Optional[Iterable[Iterable[str]]] = None

    sources = []
    if formula_column:
        sources.append(("formula", formula_column))
    if elements_column:
        sources.append(("elements", elements_column))

    for source_type, column in sources:
        if column not in df.columns:
            continue
        series = df[column].dropna()
        if source_type == "formula":
            element_lists = _iter_elements_from_formulas(series)
        else:
            element_lists = _iter_elements_from_column(series)
        break

    if element_lists is None:
        missing = ", ".join(repr(col) for _, col in sources) or "specified columns"
        raise KeyError(f"None of the columns {missing} were found in {csv_path}")

    for element_list in element_lists:
        for symbol in set(element_list):  # count once per compound
            frequency[symbol] += 1

    return frequency


def build_heatmap_frames(frequency: Counter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create data and label frames aligned with periodic table layout."""
    max_rows, max_cols = 9, 18
    heat = np.full((max_rows, max_cols), np.nan)
    labels = np.empty((max_rows, max_cols), dtype=object)

    period_labels = {
        1: "Period 1",
        2: "Period 2",
        3: "Period 3",
        4: "Period 4",
        5: "Period 5",
        6: "Period 6",
        7: "Period 7",
        8: "Lanthanides",
        9: "Actinides",
    }

    for atomic_number in range(1, 119):
        element = Element.from_Z(atomic_number)
        symbol = element.symbol
        count = frequency.get(symbol, 0)

        # Map lanthanides/actinides to dedicated rows.
        if 57 <= atomic_number <= 71:
            row = 8
            group = atomic_number - 54  # Align La under group 3.
        elif 89 <= atomic_number <= 103:
            row = 9
            group = atomic_number - 86  # Align Ac under group 3.
        else:
            row = element.row
            group = element.group

        if group is None or not (1 <= group <= max_cols):
            continue

        row_idx = row - 1
        col_idx = group - 1

        heat[row_idx, col_idx] = count
        labels[row_idx, col_idx] = symbol

    index = [period_labels[i + 1] for i in range(max_rows)]
    columns = list(range(1, max_cols + 1))

    heat_df = pd.DataFrame(heat, index=index, columns=columns)
    label_df = pd.DataFrame(labels, index=index, columns=columns)
    return heat_df, label_df


def plot_heatmap(
    heat_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    log_scale: bool = True,
    color_la_ac: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 8),
    show_colorbar: bool = True,
    colorbar_label: str = "Compound count",
) -> plt.Axes:
    """Render the heatmap; save if no external axis is supplied."""
    own_axis = ax is None
    if own_axis:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    heat_df = heat_df.copy()
    label_df = label_df.copy()

    if color_la_ac:
        # Aggregate lanthanide/actinide rows into their main-table placeholders.
        lan_sum = float(np.nansum(heat_df.loc["Lanthanides"].values))
        act_sum = float(np.nansum(heat_df.loc["Actinides"].values))
        heat_df.at["Period 6", 3] = lan_sum
        heat_df.at["Period 7", 3] = act_sum
        label_df.at["Period 6", 3] = "La"
        label_df.at["Period 7", 3] = "Ac"
    else:
        label_df.at["Period 6", 3] = "La"
        label_df.at["Period 7", 3] = "Ac"
        heat_df.at["Period 6", 3] = np.nan
        heat_df.at["Period 7", 3] = np.nan

    def add_spacer(df: pd.DataFrame, fill_value=np.nan, dtype=None) -> pd.DataFrame:
        idx = df.index.tolist()
        insert_pos = idx.index("Period 7") + 1
        spacer = pd.DataFrame(
            [ [fill_value] * df.shape[1] ],
            index=[" "],
            columns=df.columns,
            dtype=dtype,
        )
        return pd.concat([df.iloc[:insert_pos], spacer, df.iloc[insert_pos:]])

    display_heat_df = add_spacer(heat_df)
    display_label_df = add_spacer(label_df, fill_value=None, dtype=object)

    linear_df = display_heat_df.copy()
    plot_df = linear_df
    norm = None
    cbar_label = colorbar_label

    if log_scale:
        plot_df = linear_df.where(linear_df > 0)
        valid_values = plot_df.values[np.isfinite(plot_df.values)]
        if valid_values.size == 0:
            log_scale = False
        else:
            min_nonzero = valid_values.min()
            max_value = valid_values.max()
            if min_nonzero == max_value:
                log_scale = False
                plot_df = linear_df
            else:
                norm = LogNorm(vmin=min_nonzero, vmax=max_value)
                cbar_label = f"{colorbar_label} (log scale)"

    if not log_scale:
        plot_df = linear_df

    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad(color="white")

    mask = display_heat_df.isna()
    if show_colorbar:
        cbar_kws = {"label": cbar_label, "shrink": 0.6}
    else:
        cbar_kws = None

    ax = sns.heatmap(
        plot_df,
        mask=mask,
        cmap=cmap,
        linewidths=0,
        square=True,
        norm=norm,
        cbar=show_colorbar,
        cbar_kws=cbar_kws,
        ax=ax,
    )
    ax.set_facecolor("white")  # Keep masked areas seamless with background.

    max_value = np.nanmax(display_heat_df.values)
    midpoint = max_value * 0.5 if max_value else 0

    for row_idx in range(display_label_df.shape[0]):
        for col_idx in range(display_label_df.shape[1]):
            symbol = display_label_df.iat[row_idx, col_idx]
            if not symbol:
                continue

            value = display_heat_df.iat[row_idx, col_idx]
            if np.isnan(value):
                text_color = "black"
            else:
                text_color = "white" if value >= midpoint else "black"

            # Draw boundary and label only for real elements.
            rect = Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor="#424242",
                linewidth=0.8,
            )
            ax.add_patch(rect)

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                symbol,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )
    ax.set_ylabel("")
    ax.set_xlabel("Group")
    if title:
        ax.set_title(title)

    if own_axis:
        fig.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return ax


def print_top_elements(label: str, frequency: Counter, top_n: int = 10) -> None:
    """Print the most common elements for quick inspection."""
    print(f"Top elements ({label}):")
    for symbol, count in frequency.most_common(top_n):
        print(f"{symbol:>3}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input CSV path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output image path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--formula-column",
        type=str,
        default="formula",
        help="Column containing chemical formulas parsable by pymatgen (default: formula).",
    )
    parser.add_argument(
        "--elements-column",
        type=str,
        default=None,
        help="Column containing explicit element lists (used if formula column missing).",
    )
    parser.add_argument(
        "--color-la-ac",
        action="store_true",
        help="Color the La/Ac placeholders using the sum of their rows instead of leaving them blank.",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default="Primary dataset",
        help="Title for the primary dataset heatmap.",
    )
    parser.add_argument(
        "--secondary-csv",
        type=Path,
        default=None,
        help="Optional second CSV file to render alongside the primary dataset.",
    )
    parser.add_argument(
        "--secondary-formula-column",
        type=str,
        default=None,
        help="Formula column for the secondary dataset (defaults to primary formula column).",
    )
    parser.add_argument(
        "--secondary-elements-column",
        type=str,
        default=None,
        help="Elements column for the secondary dataset (defaults to primary elements column).",
    )
    parser.add_argument(
        "--secondary-label",
        type=str,
        default="Secondary dataset",
        help="Title for the secondary dataset heatmap.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("figs/element_frequency_heatmap_comparison.pdf"),
        help="Output image path when rendering both datasets together.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.csv
    if not data_path.exists():
        raise FileNotFoundError(f"Could not locate {data_path}")

    formula_column = args.formula_column or None
    elements_column = args.elements_column or None

    primary_frequency = load_element_frequencies(
        data_path, formula_column=formula_column, elements_column=elements_column
    )
    primary_heat_df, primary_label_df = build_heatmap_frames(primary_frequency)
    print_top_elements(args.primary_label, primary_frequency)

    if args.secondary_csv:
        secondary_path = args.secondary_csv
        if not secondary_path.exists():
            raise FileNotFoundError(f"Could not locate {secondary_path}")

        secondary_formula = (
            args.secondary_formula_column
            if args.secondary_formula_column is not None
            else formula_column
        )
        secondary_elements = (
            args.secondary_elements_column
            if args.secondary_elements_column is not None
            else elements_column
        )

        secondary_frequency = load_element_frequencies(
            secondary_path,
            formula_column=secondary_formula,
            elements_column=secondary_elements,
        )
        secondary_heat_df, secondary_label_df = build_heatmap_frames(secondary_frequency)

        print_top_elements(args.secondary_label, secondary_frequency)

        fig, axes = plt.subplots(1, 2, figsize=(24, 9))
        titles = [f"A) {args.primary_label}", f"B) {args.secondary_label}"]

        plot_heatmap(
            primary_heat_df,
            primary_label_df,
            output_path=None,
            color_la_ac=args.color_la_ac,
            ax=axes[0],
            title=titles[0],
            show_colorbar=True,
        )
        plot_heatmap(
            secondary_heat_df,
            secondary_label_df,
            output_path=None,
            color_la_ac=args.color_la_ac,
            ax=axes[1],
            title=titles[1],
            show_colorbar=True,
        )

        fig.tight_layout()
        comparison_path = args.comparison_output
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(comparison_path, dpi=300)
        plt.close(fig)
        print(f"\nComparison heatmap saved to {comparison_path}")
    else:
        plot_heatmap(
            primary_heat_df,
            primary_label_df,
            args.output,
            color_la_ac=args.color_la_ac,
            title=args.primary_label,
        )
        print(f"\nHeatmap saved to {args.output}")


if __name__ == "__main__":
    main()
