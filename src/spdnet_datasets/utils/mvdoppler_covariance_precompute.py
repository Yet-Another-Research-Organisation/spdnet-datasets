#!/usr/bin/env python3
"""
Precompute covariance matrices for MVDoppler dataset.

Each .h5 file contains radar_dat of shape (128, 256, 2). The 2 channels are
I/Q components. We compute magnitude, crop center 75% of y (128→96), and
compute per-sample covariance along the frequency axis.

Usage:
    python -m spdnet_datasets.utils.mvdoppler_covariance_precompute \
        --data_dir /DATA/MVDoppler
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

N_FEATURES = 96  # 75% of 128
CROP_FRACTION = 0.75
MIN_FILE_SIZE = 10 * 1024  # 10 KB — skip corrupt/empty files


def compute_scm(X: np.ndarray) -> np.ndarray:
    """Compute Sample Covariance Matrix from (n_samples, n_features)."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / n


def geometric_mean_eigenvalues(cov: np.ndarray) -> float:
    """Geometric mean of positive eigenvalues."""
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-15]
    if len(eigs) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(eigs))))


def load_h5_sample(h5_path: str) -> np.ndarray | None:
    """
    Load and process a single MVDoppler .h5 file.

    Returns:
        X of shape (n_time, n_features) or None if invalid.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'radar_dat' not in f:
                return None
            data = f['radar_dat'][:]  # (128, 256, 2)
    except Exception:
        return None

    if data.ndim != 3 or data.shape[2] < 2:
        return None

    # Compute magnitude from I/Q
    mag = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2).astype(np.float64)
    # mag shape: (128, 256) = (freq, time)

    freq_h = mag.shape[0]
    target_h = int(freq_h * CROP_FRACTION)
    start = (freq_h - target_h) // 2
    mag = mag[start:start + target_h, :]  # (96, 256)

    # mag: (n_freq, n_time) → transpose for covariance
    return mag.T  # (256, 96) = (n_time, n_features)


def extract_class_from_filename(filename: str) -> str:
    """
    Extract class name from MVDoppler filename.
    Pattern: {class}_{number}.h5
    """
    stem = Path(filename).stem
    # Split on last underscore (before the number)
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def precompute_mvdoppler(
    data_dir: str,
    force: bool = False,
    verify: bool = True,
):
    """
    Precompute covariance matrices for MVDoppler.
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / "cov"

    # Find all valid .h5 files
    all_h5 = sorted(data_dir.rglob("*.h5"))
    print(f"Total .h5 files found: {len(all_h5)}")

    # Filter by file size
    valid_h5 = [f for f in all_h5 if f.stat().st_size >= MIN_FILE_SIZE]
    print(f"Valid files (>= {MIN_FILE_SIZE // 1024} KB): {len(valid_h5)}")

    # Build sample list with class info
    all_samples = []
    for h5_file in valid_h5:
        class_name = extract_class_from_filename(h5_file.name)
        all_samples.append({
            'path': h5_file,
            'class_name': class_name,
        })

    # Print class distribution
    from collections import Counter
    class_counts = Counter(s['class_name'] for s in all_samples)
    print("Class distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    if not all_samples:
        print("No valid samples!")
        return

    # ---- PASS 1: estimate scaling factor ----
    print("\n=== PASS 1: Estimating scaling factor ===")
    geo_means = []
    subset = all_samples[:: max(1, len(all_samples) // 200)]

    for s in tqdm(subset, desc="Pass 1"):
        X = load_h5_sample(str(s['path']))
        if X is None:
            continue
        cov = compute_scm(X)
        gm = geometric_mean_eigenvalues(cov)
        geo_means.append(gm)

    avg_gm = float(np.mean(geo_means))
    scaling_factor = 1.0 / np.sqrt(avg_gm) if avg_gm > 0 else 1.0
    print(f"Average geometric mean: {avg_gm:.6e}")
    print(f"Scaling factor: {scaling_factor:.6e}")

    # ---- PASS 2: compute and save ----
    print("\n=== PASS 2: Computing scaled covariances ===")
    processed = 0
    existing = 0
    skipped = 0

    for s in tqdm(all_samples, desc="Pass 2"):
        class_name = s['class_name']
        stem = s['path'].stem
        out_class_dir = out_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_class_dir / f"{stem}.pt"

        if out_path.exists() and not force:
            existing += 1
            continue

        X = load_h5_sample(str(s['path']))
        if X is None:
            skipped += 1
            continue

        X_scaled = X * scaling_factor
        cov = compute_scm(X_scaled)

        torch.save(torch.from_numpy(cov).float(), out_path)
        processed += 1

    print(f"Processed: {processed}, Already existing: {existing}, "
          f"Skipped: {skipped}")

    # Save metadata
    meta = {
        'scaling_factor': scaling_factor,
        'n_features': N_FEATURES,
        'crop_fraction': CROP_FRACTION,
        'total_samples': processed + existing,
        'avg_geometric_mean_before_scaling': avg_gm,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(meta, out_dir / "precompute_meta.pt")
    print(f"Metadata saved to {out_dir / 'precompute_meta.pt'}")

    # ---- Verification ----
    if verify:
        print("\n=== Verification ===")
        pt_files = list(out_dir.rglob("*.pt"))
        pt_files = [f for f in pt_files if f.name != "precompute_meta.pt"]
        check = pt_files[:: max(1, len(pt_files) // 10)]
        gm_after = []
        for f in check:
            cov = torch.load(f, weights_only=True)
            eigs = torch.linalg.eigvalsh(cov).numpy()
            eigs = eigs[eigs > 1e-15]
            gm = float(np.exp(np.mean(np.log(eigs))))
            gm_after.append(gm)
        print(f"Geometric mean after scaling (avg over {len(check)} samples): "
              f"{np.mean(gm_after):.4f} (target ~1.0)")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute covariance matrices for MVDoppler"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of MVDoppler"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip verification")
    args = parser.parse_args()

    precompute_mvdoppler(
        data_dir=args.data_dir,
        force=args.force,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
