#!/usr/bin/env python3
"""
Simple .mat to .npy converter.

Usage:
    python convert_mat_to_npy.py
"""

import h5py
import numpy as np
from pathlib import Path


def convert_mat_to_npy(mat_file):
    """Convert a .mat file to .npy files."""
    mat_path = Path(mat_file)

    # Create output directory: smh_train.mat → smh_train/
    output_dir = mat_path.parent / mat_path.stem

    # Skip if already converted
    if output_dir.exists() and any(output_dir.glob("*.npy")):
        print(f"\n{mat_path.name} → ⏭️  Already converted, skipping")
        return

    output_dir.mkdir(exist_ok=True)

    print(f"\n{mat_path.name} →")

    with h5py.File(mat_path, 'r') as f:
        for key in f.keys():
            data = np.array(f[key])
            output_file = output_dir / f"{key}.npy"
            np.save(output_file, data)
            print(f"  ✓ {key}.npy  {data.shape}")


if __name__ == "__main__":
    data_dir = Path("data")

    # Convert all .mat files
    for mat_file in data_dir.rglob("*.mat"):
        convert_mat_to_npy(mat_file)

    print("\n✓ Done!")