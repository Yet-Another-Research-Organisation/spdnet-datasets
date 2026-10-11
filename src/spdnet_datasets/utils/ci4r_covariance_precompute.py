#!/usr/bin/env python3
"""
Precompute covariance matrices for CI4R-MULTI3 dataset.

Processes 77GHz and Xethru spectrograms. Covariance is computed along the
frequency axis (y): each column is a time sample, each row a frequency bin.
Center 75% of y is cropped. Per-sample SCM with geometric-mean scaling.

Usage:
    python -m spdnet_datasets.utils.ci4r_covariance_precompute \
        --data_dir /DATA/CI4R-MULTI3
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

MODALITIES = {
    '77GHz': '77GHz',
    'Xethru': 'Xethru',
}

TARGET_HEIGHT = 471  # common target after 75% crop
FREQ_SUBSAMPLE = 2  # subsample frequency to improve conditioning


def load_spectrogram(img_path: str) -> np.ndarray:
    """
    Load PNG spectrogram as grayscale float64 array (H, W).
    """
    img = Image.open(img_path).convert('L')
    return np.array(img, dtype=np.float64)


def crop_center_y(spec: np.ndarray, target_h: int) -> np.ndarray:
    """Crop center portion along y-axis (rows)."""
    h = spec.shape[0]
    if target_h >= h:
        return spec
    start = (h - target_h) // 2
    return spec[start:start + target_h]


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


def process_spectrogram(img_path: str, target_h: int) -> np.ndarray:
    """
    Load, crop, subsample, and prepare spectrogram for covariance.

    Returns:
        X of shape (n_time, n_freq) where n_time is the time axis
        and n_freq = target_h // FREQ_SUBSAMPLE is the frequency features.
    """
    spec = load_spectrogram(img_path)
    spec = crop_center_y(spec, target_h)
    # Subsample frequency axis by 2 for better conditioning
    spec = spec[::FREQ_SUBSAMPLE, :]
    # spec shape: (n_freq, n_time) — transpose so rows=samples, cols=features
    return spec.T  # (n_time, n_freq)


def precompute_ci4r(
    data_dir: str,
    modalities: list | None = None,
    force: bool = False,
    verify: bool = True,
):
    """
    Precompute covariance matrices for CI4R-MULTI3.

    Args:
        data_dir: Root directory of CI4R-MULTI3
        modalities: List of modality names to process (default: both)
        force: Recompute even if output exists
        verify: Verify results after computation
    """
    data_dir = Path(data_dir)
    src_dir = data_dir / "11_class_activity_data"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if modalities is None:
        modalities = list(MODALITIES.keys())

    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing modality: {modality}")
        print(f"{'='*60}")

        mod_dir = src_dir / MODALITIES[modality]
        if not mod_dir.exists():
            print(f"WARNING: {mod_dir} not found, skipping")
            continue

        out_dir = data_dir / "cov" / modality

        # Collect all samples
        all_samples = []
        class_dirs = sorted([d for d in mod_dir.iterdir() if d.is_dir()])
        print(f"Found {len(class_dirs)} classes")

        for class_dir in class_dirs:
            class_name = class_dir.name
            for img_file in sorted(class_dir.glob("*.png")):
                all_samples.append({
                    'path': img_file,
                    'class_name': class_name,
                })

        if not all_samples:
            # Try flat structure with class in filename
            for img_file in sorted(mod_dir.glob("*.png")):
                parts = img_file.stem.split('_')
                class_name = parts[0] if parts else "unknown"
                all_samples.append({
                    'path': img_file,
                    'class_name': class_name,
                })

        print(f"Total samples: {len(all_samples)}")
        if not all_samples:
            continue

        # ---- PASS 1: estimate scaling factor ----
        print("\n--- Pass 1: Estimating scaling factor ---")
        geo_means = []
        subset = all_samples[:: max(1, len(all_samples) // 100)]

        for s in tqdm(subset, desc="Pass 1"):
            X = process_spectrogram(str(s['path']), TARGET_HEIGHT)
            cov = compute_scm(X)
            gm = geometric_mean_eigenvalues(cov)
            geo_means.append(gm)

        avg_gm = float(np.mean(geo_means))
        scaling_factor = 1.0 / np.sqrt(avg_gm) if avg_gm > 0 else 1.0
        print(f"Average geometric mean: {avg_gm:.6e}")
        print(f"Scaling factor: {scaling_factor:.6e}")

        # ---- PASS 2: compute and save ----
        print("\n--- Pass 2: Computing scaled covariances ---")
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

            X = process_spectrogram(str(s['path']), TARGET_HEIGHT)
            X_scaled = X * scaling_factor
            cov = compute_scm(X_scaled)

            torch.save(torch.from_numpy(cov).float(), out_path)
            processed += 1

        print(f"Processed: {processed}, Already existing: {existing}")

        # Save metadata
        meta = {
            'scaling_factor': scaling_factor,
            'modality': modality,
            'target_height': TARGET_HEIGHT,
            'total_samples': len(all_samples),
            'avg_geometric_mean_before_scaling': avg_gm,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(meta, out_dir / "precompute_meta.pt")
        print(f"Metadata saved to {out_dir / 'precompute_meta.pt'}")

        # ---- Verification ----
        if verify:
            print("\n--- Verification ---")
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
        description="Precompute covariance matrices for CI4R-MULTI3"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of CI4R-MULTI3"
    )
    parser.add_argument(
        "--modalities", nargs='+', default=None,
        choices=list(MODALITIES.keys()),
        help="Modalities to process (default: all)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip verification")
    args = parser.parse_args()

    precompute_ci4r(
        data_dir=args.data_dir,
        modalities=args.modalities,
        force=args.force,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
