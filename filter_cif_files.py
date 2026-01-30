#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


def load_mpids(csv_path: Path) -> set[str]:
    """Return the set of mpids referenced in the dataset CSV."""
    mpids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "mpids" not in reader.fieldnames:
            raise ValueError("Expected an 'mpids' column in the CSV header.")
        for row in reader:
            mpid = (row.get("mpids") or "").strip()
            if mpid:
                mpids.add(mpid)
    return mpids


def filter_cif_files(
    mpids: set[str],
    source_dir: Path,
    target_dir: Path,
) -> tuple[int, int]:
    """
    Copy CIF files whose stem matches the provided mpids into the target directory.

    Returns:
        copied_count: Number of CIF files copied.
        missing_count: Number of mpids without corresponding CIF files.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source CIF directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for mpid in sorted(mpids):
        cif_name = f"{mpid}.cif"
        cif_path = source_dir / cif_name
        if cif_path.is_file():
            shutil.copy2(cif_path, target_dir / cif_name)
            copied += 1
        else:
            missing += 1
    return copied, missing


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "joint_dataset" / "combined_bandgap_data.csv"
    source_dir = project_root / "cif_file"
    target_dir = project_root / "cif_file_filtered"

    mpids = load_mpids(csv_path)
    copied, missing = filter_cif_files(mpids, source_dir, target_dir)

    print(f"Unique mpids in CSV: {len(mpids)}")
    print(f"CIF files copied: {copied}")
    print(f"mpids without CIF files: {missing}")


if __name__ == "__main__":
    main()
