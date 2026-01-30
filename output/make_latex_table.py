#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generate a LaTeX table from metrics_agg.csv:
- Automatically discovers all split groups in the CSV.
- Groups by (split_dir, split_name).
- Bold = best, underline = second-best within each group.
- Prints mean ± std as \text{X.XXX}_\text{(Y.YYY)}.
"""

import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

PREFACE = r"""\begin{table}[t]
\centering
\caption{Performance of different models on the validation and test sets, reported as mean $\pm$ standard deviation (\textbf{best in bold}, \underline{second best} in underline). Higher values indicate better performance ($\uparrow$), otherwise ($\downarrow$).}
\scalebox{1}{
\setlength{\tabcolsep}{1.0mm}{
\begin{tabular}{l|ccc|ccc}
\toprule
\multirow{2}{*}[-0.4ex]{Model} & \multicolumn{3}{c|}{Validation set} & \multicolumn{3}{c}{Test set} \\
& MAE(eV) $\downarrow$ & MRAE $\downarrow$ & ${R^2}$ $\uparrow$ & MAE(eV) $\downarrow$ & MRAE $\downarrow$ & ${R^2}$ $\uparrow$ \\
\midrule\midrule
"""
POSTFACE = r"""\bottomrule
\end{tabular}}}
\label{tab:cv_performance}
\vspace{-15pt}
\end{table}
"""

def fmt(mu, sd):
    """Format as math: $\text{0.642}_{\text{(0.059)}}$ or NaN if missing."""
    if pd.isna(mu) or pd.isna(sd):
        return r"$\text{NaN}$"
    return rf"$\text{{{mu:.3f}}}_\text{{({sd:.3f})}}$"


def rank_and_mark(strings, values, direction):
    """
    Mark best and second best in 'strings' based on 'values'.
    direction='min' for MAE/MRAE; 'max' for R2.
    """
    arr = np.array(values, dtype=float)
    bad = np.isnan(arr)
    if direction == "min":
        arr_rank = np.where(bad, np.inf, arr)
        order = np.argsort(arr_rank)
    else:
        arr_rank = np.where(bad, -np.inf, arr)
        order = np.argsort(-arr_rank)

    marked = strings[:]
    candidates = [i for i in order if np.isfinite(arr_rank[i])]
    if len(candidates) >= 1:
        i0 = candidates[0]
        marked[i0] = r"\textbf{" + marked[i0] + "}"
    if len(candidates) >= 2:
        i1 = candidates[1]
        marked[i1] = r"\underline{" + marked[i1] + "}"
    return marked

def section_title(split_dir, split_name):
    """Human-friendly titles for each (split_dir, split_name)."""
    sd = str(split_dir).strip() if pd.notna(split_dir) else ""
    sn = str(split_name).strip() if pd.notna(split_name) else ""

    if sd == "finetune":
        return "No pre-training (fine-tuning only)"
    if sd == "ood":
        return "OOD evaluation"
    if sd == "split" and sn:
        # nicer aliases if you want
        aliases = {
            "chemsys": "Chemical system split",
            "composition": "Composition split",
            "periodictablegroups": "PT groups split",
            "sgnum": "Space group split",
            "structureid": "Structure-ID split",
        }
        return aliases.get(sn, f"{sn} split")
    # fallback
    return (sn + " " if sn else "") + sd

def sort_groups(df):
    """
    Sort (split_dir, split_name) pairs in a sensible order:
    finetune -> split:* (custom order) -> ood -> others
    """
    groups = df[["split_dir","split_name"]].drop_duplicates()

    pref_split_order = ["chemsys", "composition", "periodictablegroups", "sgnum", "structureid"]
    def score(row):
        sd = row["split_dir"]
        sn = row["split_name"] if pd.notna(row["split_name"]) else ""
        if sd == "finetune": return (0, 0)
        if sd == "split":
            idx = pref_split_order.index(sn) if sn in pref_split_order else 999
            return (1, idx)
        if sd == "ood": return (2, 0)
        return (3, 0)
    groups = groups.copy()
    groups["__ord__"] = groups.apply(score, axis=1)
    groups = groups.sort_values(["__ord__", "split_dir", "split_name"], na_position="last").drop(columns="__ord__")
    return [tuple(x) for x in groups[["split_dir","split_name"]].values]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to metrics_agg.csv")
    ap.add_argument("--out", dest="out_tex", required=True, help="Output LaTeX file")
    ap.add_argument("--name-map", type=str, default="{}", help="JSON mapping method->display name")
    ap.add_argument("--gray-rows", action="store_true", help="Alternate gray row shading (requires mygray color)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    name_map = json.loads(args.name_map)

    # Ensure numeric (keep NaN)
    numeric_cols = [
        "val_mae_mean","val_mae_std","val_mre_mean","val_mre_std","val_r2_mean","val_r2_std",
        "test_mae_mean","test_mae_std","test_mre_mean","test_mre_std","test_r2_mean","test_r2_std"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Auto-detect all groups
    groups = sort_groups(df)

    lines = [PREFACE]

    for (split_dir, split_name) in groups:
        # subset safely (treat NaN consistently)
        mask = (df["split_dir"].astype(str).str.strip() == str(split_dir).strip())
        if "split_name" in df.columns:
            if pd.isna(split_name) or str(split_name).strip() == "":
                mask &= (
                    df["split_name"].isna()
                    | (df["split_name"].astype(str).str.strip() == "")
                    | (df["split_name"].astype(str).str.lower().str.strip() == "none")
                )
            else:
                mask &= (df["split_name"].astype(str).str.strip() == str(split_name).strip())

        g = df.loc[mask].copy()
        # Skip empty groups completely
        if g.empty:
            continue

        # section header
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{section_title(split_dir, split_name)}}}}} \\ \midrule")

        # sort methods for stable order
        g = g.sort_values("method").reset_index(drop=True)

        # Pre-format cells
        val_mae = [fmt(m, s) for m, s in zip(g["val_mae_mean"],  g["val_mae_std"])]
        val_mre = [fmt(m, s) for m, s in zip(g["val_mre_mean"],  g["val_mre_std"])]
        val_r2  = [fmt(m, s) for m, s in zip(g["val_r2_mean"],   g["val_r2_std"])]
        tst_mae = [fmt(m, s) for m, s in zip(g["test_mae_mean"], g["test_mae_std"])]
        tst_mre = [fmt(m, s) for m, s in zip(g["test_mre_mean"], g["test_mre_std"])]
        tst_r2  = [fmt(m, s) for m, s in zip(g["test_r2_mean"],  g["test_r2_std"])]

        # Apply best/second-best markers per metric within this group
        val_mae = rank_and_mark(val_mae, g["val_mae_mean"].values,  "min")
        val_mre = rank_and_mark(val_mre, g["val_mre_mean"].values,  "min")
        val_r2  = rank_and_mark(val_r2,  g["val_r2_mean"].values,   "max")
        tst_mae = rank_and_mark(tst_mae, g["test_mae_mean"].values, "min")
        tst_mre = rank_and_mark(tst_mre, g["test_mre_mean"].values, "min")
        tst_r2  = rank_and_mark(tst_r2,  g["test_r2_mean"].values,  "max")

        # Emit rows
        for i in range(len(g)):
            m = name_map.get(g.loc[i, "method"], g.loc[i, "method"])
            row = f"{m} & {val_mae[i]} & {val_mre[i]} & {val_r2[i]} & {tst_mae[i]} & {tst_mre[i]} & {tst_r2[i]} \\\\"
            if args.gray_rows and (i % 2 == 1):
                row = r"\rowcolor{mygray}" + "\n" + row
            lines.append(row)

        lines.append(r"\midrule\midrule")

    lines.append(POSTFACE)

    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote LaTeX table to {args.out_tex}")

if __name__ == "__main__":
    main()
