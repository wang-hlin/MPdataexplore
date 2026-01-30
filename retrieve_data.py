import json
import pandas as pd

def df_to_keyed_json(df, out_path, id_col="material_id", bg_col="bg", comp_col="composition"):
    # Keep only needed columns and clean up
    need = [id_col, bg_col, comp_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[need].dropna(subset=[id_col, bg_col])
    df[bg_col] = pd.to_numeric(df[bg_col], errors="coerce")
    df = df.dropna(subset=[bg_col])

    # Build mapping: "mp-xxxx": {"bg": ..., "composition": "..."}
    out_dict = {
        str(row[id_col]): {"bg": float(row[bg_col]), "composition": str(row[comp_col])}
        for _, row in df.iterrows()
    }

    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=4, sort_keys=True)
    print(f"Saved {len(out_dict)} entries to {out_path}")


# --- usage ---
df = pd.read_csv("filtered_data/test.csv", low_memory=False)
df = df.drop(columns=["elements"], errors="ignore")  # remove if present
df_to_keyed_json(df, "filtered_data/test.json")
