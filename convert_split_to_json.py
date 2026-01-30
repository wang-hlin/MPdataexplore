from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


def load_split(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load a split CSV and return {mpid: {"bg": bg_exp}}."""
    mapping: Dict[str, Dict[str, float]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or all(
            name not in reader.fieldnames for name in ("mpids", "mpid", "id")
        ):
            # Skip summary-style CSVs that do not contain per-mpid entries.
            return mapping
        for row in reader:
            mpid = row.get("mpids") or row.get("mpid") or row.get("id")
            if not mpid:
                # Skip rows without identifiers instead of aborting the conversion.
                continue
            raw_bg = (row.get("bg_exp") or "").strip()
            if not raw_bg:
                # Skip entries with no experimental band gap.
                continue
            try:
                band_gap = float(raw_bg)
            except ValueError as exc:
                raise ValueError(f"Invalid bg_exp '{raw_bg}' in {csv_path}") from exc
            mapping[mpid] = {"bg": band_gap}
    return mapping


def convert_all_splits(split_root: Path, output_root: Path) -> None:
    for split_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        target_dir = output_root / split_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(split_dir.glob("*.csv")):
            mapping = load_split(csv_path)
            output_path = target_dir / (csv_path.stem + ".json")
            with output_path.open("w") as handle:
                json.dump(mapping, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    SPLIT_ROOT = Path("data/split")
    OUTPUT_ROOT = Path("data/split_json")
    convert_all_splits(SPLIT_ROOT, OUTPUT_ROOT)
