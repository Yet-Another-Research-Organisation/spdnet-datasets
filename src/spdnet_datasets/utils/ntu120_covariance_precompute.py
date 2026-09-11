#!/usr/bin/env python3
"""
Precompute covariance matrices for NTU RGB+D 120 skeleton dataset.

Each .npy file contains a dict with skel_body0 of shape (nframes, 25, 3).
Reshape to (nframes, 75) features, filter by min_timesteps, compute SCM.
Two-pass approach: estimate scaling factor, then compute scaled covariances.

Usage:
    python -m spdnet_datasets.utils.ntu120_covariance_precompute \
        --data_dir /DATA/NTU_RGBD_120
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_skeleton(npy_path: str) -> np.ndarray:
    """
    Load skeleton data from .npy dict file.

    Returns:
        Array of shape (nframes, 75) or None if invalid.
    """
    data = np.load(npy_path, allow_pickle=True)
    if data.ndim == 0:
        d = data.item()
    else:
        return None

    if not isinstance(d, dict) or 'skel_body0' not in d:
        return None

    skel = d['skel_body0']  # (nframes, 25, 3)
    nframes = skel.shape[0]
    return skel.reshape(nframes, -1).astype(np.float64)  # (nframes, 75)


def compute_scm(X: np.ndarray) -> np.ndarray:
    """Compute Sample Covariance Matrix from (n_samples, n_features)."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / n


def geometric_mean_eigenvalues(cov: np.ndarray) -> float:
    """Geometric mean of positive eigenvalues: exp(mean(log(eigs)))."""
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-15]
    if len(eigs) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(eigs))))


def precompute_ntu120(
    data_dir: str,
    min_timesteps: int = 150,
    force: bool = False,
    verify: bool = True,
):
    """
    Precompute covariance matrices for NTU RGB+D 120.

    Args:
        data_dir: Root directory (contains npy_skeletons_by_class/)
        min_timesteps: Minimum number of frames to keep a sample
        force: Recompute even if output exists
        verify: Verify results after computation
    """
    data_dir = Path(data_dir)
    src_dir = data_dir / "npy_skeletons_by_class"
    out_dir = data_dir / "cov"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    class_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in {src_dir}")

    # Collect all valid samples
    all_samples = []
    skipped_short = 0
    skipped_invalid = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        for npy_file in sorted(class_dir.glob("*.npy")):
            skel = load_skeleton(str(npy_file))
            if skel is None:
                skipped_invalid += 1
                continue
            if skel.shape[0] < min_timesteps:
                skipped_short += 1
                continue
            all_samples.append({
                'path': npy_file,
                'class_name': class_name,
                'nframes': skel.shape[0],
            })

    print(f"Valid samples: {len(all_samples)}")
    print(f"Skipped (< {min_timesteps} frames): {skipped_short}")
    print(f"Skipped (invalid format): {skipped_invalid}")

    if len(all_samples) == 0:
        print("No valid samples found!")
        return

    # ---- PASS 1: estimate scaling factor ----
    print("\n=== PASS 1: Estimating scaling factor ===")
    geo_means = []
    subset = all_samples[:: max(1, len(all_samples) // 200)]  # sample up to 200
    for s in tqdm(subset, desc="Pass 1"):
        skel = load_skeleton(str(s['path']))
        cov = compute_scm(skel)
        gm = geometric_mean_eigenvalues(cov)
        geo_means.append(gm)

    avg_gm = float(np.mean(geo_means))
    scaling_factor = 1.0 / np.sqrt(avg_gm) if avg_gm > 0 else 1.0
    print(f"Average geometric mean: {avg_gm:.6e}")
    print(f"Scaling factor (for data): {scaling_factor:.6e}")

    # ---- PASS 2: compute and save scaled covariances ----
    print("\n=== PASS 2: Computing scaled covariances ===")
    processed = 0
    existing = 0

    for s in tqdm(all_samples, desc="Pass 2"):
        class_name = s['class_name']
        stem = s['path'].stem
        out_class_dir = out_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_class_dir / f"{stem}.pt"

        if out_path.exists() and not force:
            existing += 1
            continue

        skel = load_skeleton(str(s['path']))
        skel_scaled = skel * scaling_factor
        cov = compute_scm(skel_scaled)

        torch.save(torch.from_numpy(cov).float(), out_path)
        processed += 1

    print(f"Processed: {processed}, Already existing: {existing}")

    # Save metadata
    meta = {
        'scaling_factor': scaling_factor,
        'min_timesteps': min_timesteps,
        'n_features': 75,
        'total_samples': len(all_samples),
        'avg_geometric_mean_before_scaling': avg_gm,
    }
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
            is_sym = torch.allclose(cov, cov.T, atol=1e-6)
            is_psd = eigs.min() > -1e-6
            if not (is_sym and is_psd):
                print(f"  WARNING: {f.name} sym={is_sym} psd={is_psd}")
        print(f"Geometric mean after scaling (avg over {len(check)} samples): "
              f"{np.mean(gm_after):.4f} (target ~1.0)")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute covariance matrices for NTU RGB+D 120"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing npy_skeletons_by_class/"
    )
    parser.add_argument(
        "--min_timesteps", type=int, default=150,
        help="Minimum number of frames per sample (default: 150)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip verification")
    args = parser.parse_args()

    precompute_ntu120(
        data_dir=args.data_dir,
        min_timesteps=args.min_timesteps,
        force=args.force,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
