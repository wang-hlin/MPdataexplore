#!/usr/bin/env python3
"""
Aggregate metrics over folds.
Reads a CSV produced by collect_metrics_v3.py (metrics_summary.csv) and
writes an aggregated CSV with mean and std across folds for each
(split_dir, split_name, method).

Usage:
    python aggregate_metrics.py --in ./output/metrics_summary.csv --out ./output/metrics_agg.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(in_path="./output/metrics_summary.csv", out_path=None):
    in_csv = Path(in_path)
    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(in_csv)

    # numeric columns we aggregate
    num_cols = ["val_mae","val_mre","val_r2","test_mae","test_mre","test_r2"]
    # Convert to numeric (coerce errors -> NaN)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["split_dir","split_name","method"]
    grouped = df.groupby(group_cols, dropna=False)

    # compute mean and std while ignoring NaNs
    agg = grouped[num_cols].agg(['mean','std','count'])
    # flatten MultiIndex columns
    agg.columns = ['{}_{}'.format(a,b) for a,b in agg.columns]
    agg = agg.reset_index()

    # order columns nicely
    ordered_cols = (
        group_cols +
        ["val_mae_mean","val_mae_std",
         "val_mre_mean","val_mre_std",
         "val_r2_mean","val_r2_std",
         "test_mae_mean","test_mae_std",
         "test_mre_mean","test_mre_std",
         "test_r2_mean","test_r2_std"] +
        ["val_mae_count","val_mre_count","val_r2_count",
         "test_mae_count","test_mre_count","test_r2_count"]
    )
    # keep only existing columns in case some are missing
    ordered_cols = [c for c in ordered_cols if c in agg.columns]

    out_csv = Path(out_path) if out_path else (in_csv.parent / "metrics_agg.csv")
    agg.to_csv(out_csv, index=False, columns=ordered_cols if ordered_cols else None)
    print(f"[OK] Wrote aggregate table to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default="./output/metrics_summary.csv")
    ap.add_argument("--out", dest="out_path", type=str, default=None)
    args = ap.parse_args()
    main(in_path=args.in_path, out_path=args.out_path)