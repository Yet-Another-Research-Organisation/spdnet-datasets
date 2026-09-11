#!/usr/bin/env python3
"""
Precompute covariance matrices for Dop-NET dataset.

Dop-NET contains complex spectrograms in .mat files. Each spectrogram has
shape (800, W) complex128. We take the magnitude, crop the center of the
frequency axis (y=800), and subsample to get ~50 features. Covariance is
computed along the frequency axis.

Training data: 6 .mat files (Persons A-F) in train/
Test data: Data_For_Test_Random.mat in test/

Usage:
    python -m spdnet_datasets.utils.dopnet_covariance_precompute \
        --data_dir /DATA/Dop-NET
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

N_FEATURES = 50  # target number of frequency features
MIN_TIME_RATIO = 2.0  # need at least 2x time samples vs features
CLASS_NAMES = ['Click', 'Swipe', 'Pinch', 'Wave']


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


def process_spectrogram(spec: np.ndarray, n_features: int) -> np.ndarray | None:
    """
    Process a complex spectrogram to prepare for covariance.

    Args:
        spec: Complex spectrogram of shape (freq, time)
        n_features: Target number of frequency features

    Returns:
        X of shape (n_time, n_features) or None if too short.
    """
    mag = np.abs(spec).astype(np.float64)  # (freq, time)
    freq_h, time_w = mag.shape

    # Crop center of frequency axis to 2*n_features, then subsample by 2
    crop_h = min(freq_h, 2 * n_features)
    start = (freq_h - crop_h) // 2
    mag = mag[start:start + crop_h, :]

    # Subsample frequency to reach n_features
    step = max(1, mag.shape[0] // n_features)
    # Take every step-th row from the center crop
    indices = np.arange(0, mag.shape[0], step)[:n_features]
    mag = mag[indices, :]  # (n_features, time)

    actual_features = mag.shape[0]
    time_w = mag.shape[1]

    # Check time/features ratio
    if time_w < MIN_TIME_RATIO * actual_features:
        return None

    return mag.T  # (time, n_features)


def load_training_data(data_dir: Path) -> list[dict]:
    """Load training samples from individual person .mat files."""
    samples = []

    # Try multiple possible directory names
    for dirname in ["Training Data", "train", "training", "."]:
        train_dir = data_dir / dirname if dirname != "." else data_dir
        if train_dir.exists():
            mat_files = sorted(train_dir.glob("*.mat"))
            mat_files = [f for f in mat_files if "Test" not in f.name]
            if mat_files:
                break
    else:
        print("  WARNING: No training directory found")
        return samples

    for mat_file in mat_files:
        print(f"  Loading {mat_file.name}...")
        try:
            mat = scipy.io.loadmat(str(mat_file))
        except Exception as e:
            print(f"  WARNING: Cannot load {mat_file.name}: {e}")
            continue

        # Find data key containing training data
        data_key = None
        for k in mat.keys():
            if not k.startswith('_') and 'training' in k.lower():
                data_key = k
                break
        if data_key is None:
            for k in mat.keys():
                if not k.startswith('_'):
                    data_key = k
                    break
        if data_key is None:
            print(f"  WARNING: No data key found in {mat_file.name}")
            continue

        data = mat[data_key]

        # Structure: data[0,0]['Doppler_Signals'] is (1, 4) for 4 classes
        try:
            doppler = data[0, 0]['Doppler_Signals']
        except (IndexError, KeyError, ValueError):
            try:
                doppler = data['Doppler_Signals']
                if hasattr(doppler, 'item'):
                    doppler = doppler.item()
            except Exception:
                print(f"  WARNING: Cannot extract Doppler_Signals from {data_key}")
                continue

        for class_idx in range(min(doppler.shape[1], len(CLASS_NAMES))):
            class_name = CLASS_NAMES[class_idx]
            signals = doppler[0, class_idx]  # (N, 1) array of spectrograms

            for i in range(signals.shape[0]):
                try:
                    spec = signals[i, 0]
                    if spec.ndim == 2 and spec.shape[0] > 10:
                        samples.append({
                            'spec': spec,
                            'class_name': class_name,
                            'source': mat_file.stem,
                            'idx': i,
                            'split': 'train',
                        })
                except Exception:
                    continue

    return samples


def load_test_data(data_dir: Path) -> list[dict]:
    """Load test samples from test .mat file."""
    samples = []

    test_file = None
    for candidate in [
        data_dir / "Test Data" / "Data_For_Test_Random.mat",
        data_dir / "test" / "Data_For_Test_Random.mat",
        data_dir / "Data_For_Test_Random.mat",
    ]:
        if candidate.exists():
            test_file = candidate
            break

    if test_file is None:
        found = list(data_dir.rglob("*Test*.mat"))
        if found:
            test_file = found[0]

    if test_file is None:
        print("  WARNING: No test file found")
        return samples

    print(f"  Loading {test_file.name}...")
    mat = scipy.io.loadmat(str(test_file))

    # Find data key
    data_key = None
    for k in mat.keys():
        if 'rand' in k.lower() or 'test' in k.lower():
            data_key = k
            break
    if data_key is None:
        for k in mat.keys():
            if not k.startswith('_'):
                data_key = k
                break

    if data_key is None:
        return samples

    data = mat[data_key]  # (N, 2): [spec_nested, label_string]

    for i in range(data.shape[0]):
        try:
            spec_nested = data[i, 0]
            label_str = str(data[i, 1]).strip()

            # Extract spectrogram from nested structure
            if hasattr(spec_nested, 'shape') and spec_nested.shape == (1, 1):
                spec = spec_nested[0, 0]
            else:
                spec = spec_nested

            # Parse class from label string: "Click 20 Person F"
            class_name = label_str.split()[0] if label_str else "unknown"
            # Normalize label
            for cn in CLASS_NAMES:
                if cn.lower() in label_str.lower():
                    class_name = cn
                    break

            if spec.ndim == 2 and spec.shape[0] > 10:
                samples.append({
                    'spec': spec,
                    'class_name': class_name,
                    'source': 'test',
                    'idx': i,
                    'split': 'test',
                })
        except Exception:
            continue

    return samples


def precompute_dopnet(
    data_dir: str,
    n_features: int = N_FEATURES,
    force: bool = False,
    verify: bool = True,
):
    """
    Precompute covariance matrices for Dop-NET.
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / "cov"

    # Load all samples
    print("Loading training data...")
    train_samples = load_training_data(data_dir)
    print(f"  Training samples: {len(train_samples)}")

    print("Loading test data...")
    test_samples = load_test_data(data_dir)
    print(f"  Test samples: {len(test_samples)}")

    all_raw = train_samples + test_samples

    # Process spectrograms and filter
    all_samples = []
    skipped = 0
    for s in all_raw:
        X = process_spectrogram(s['spec'], n_features)
        if X is None:
            skipped += 1
            continue
        s['X'] = X
        all_samples.append(s)

    print(f"\nValid samples after filtering: {len(all_samples)}")
    print(f"Skipped (too short): {skipped}")
    actual_n_features = all_samples[0]['X'].shape[1] if all_samples else n_features

    if not all_samples:
        print("No valid samples!")
        return

    # ---- PASS 1: estimate scaling factor ----
    print("\n=== PASS 1: Estimating scaling factor ===")
    geo_means = []
    subset = all_samples[:: max(1, len(all_samples) // 200)]

    for s in tqdm(subset, desc="Pass 1"):
        cov = compute_scm(s['X'])
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

    for s in tqdm(all_samples, desc="Pass 2"):
        split = s['split']
        class_name = s['class_name']
        source = s['source']
        idx = s['idx']
        fname = f"{source}_{idx:04d}"

        out_split_dir = out_dir / split / class_name
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_split_dir / f"{fname}.pt"

        if out_path.exists() and not force:
            existing += 1
            continue

        X_scaled = s['X'] * scaling_factor
        cov = compute_scm(X_scaled)

        torch.save(torch.from_numpy(cov).float(), out_path)
        processed += 1

    print(f"Processed: {processed}, Already existing: {existing}")

    # Save metadata
    meta = {
        'scaling_factor': scaling_factor,
        'n_features': actual_n_features,
        'total_samples': len(all_samples),
        'avg_geometric_mean_before_scaling': avg_gm,
        'class_names': CLASS_NAMES,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(meta, out_dir / "precompute_meta.pt")

    # Free memory
    for s in all_samples:
        s.pop('X', None)
        s.pop('spec', None)

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
        description="Precompute covariance matrices for Dop-NET"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of Dop-NET"
    )
    parser.add_argument(
        "--n_features", type=int, default=N_FEATURES,
        help=f"Target number of frequency features (default: {N_FEATURES})"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip verification")
    args = parser.parse_args()

    precompute_dopnet(
        data_dir=args.data_dir,
        n_features=args.n_features,
        force=args.force,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
