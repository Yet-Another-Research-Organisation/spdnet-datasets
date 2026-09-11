#!/usr/bin/env python3
"""
Comprehensive verification of all dataset loaders.

Creates per-dataset output directories with:
- Text report: per-class sample counts, eigenvalue stats, data stats
- Figures: covariance imshow + eigenvalue decay per class
- Dataset summary: dimensions, train/test split, class list

Usage:
    python tests/verify_all_datasets.py [--output_dir tests/dataset_reports]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all datasets to register them
import spdnet_datasets  # noqa: F401
from spdnet_datasets.manager import DatasetManager

# ---- Dataset configurations ----
# Each entry: (name, {config_overrides}, [list of mode dicts])
# mode dicts are extra kwargs that define distinct "modes" to verify
DATASET_CONFIGS = [
    ("rices90", {"path": "/DATA/Rices_90"}, [{}]),
    ("hyperleaf", {"path": "/DATA/HyperLeaf2024"}, [
        {"split": "train", "task": "cultivar", "mode": "cov"},
        {"split": "train", "task": "fertilizer", "mode": "cov"},
    ]),
    ("hdm05", {"path": "/DATA/HDM05"}, [{}]),
    ("uav", {"path": "/DATA/Hyperspectral/UAV-HSI-Crop"}, [
        {"mode": "cov"},
    ]),
    ("gsoff", {"path": "/DATA/Hyperspectral/GSOFF"}, [
        {"mode": "cov"},
    ]),
    ("chikusei", {"path": "/DATA/Hyperspectral/Chikusei"}, [
        {"mode": "cov"},
    ]),
    ("deephsfruit", {"path": "/DATA/DeepHS-Fruit"}, [
        {"mode": "cov", "classification_mode": "fruit"},
    ]),
    ("kaggle_wheat", {"path": "/DATA/Kaggle_Prepared"}, [
        {"split": "train"},
    ]),
    ("ntu120", {"path": "/DATA/NTU_RGBD_120"}, [{}]),
    ("ci4r", {"path": "/DATA/CI4R-MULTI3"}, [
        {"modality": "77GHz"},
        {"modality": "Xethru"},
    ]),
    ("dopnet", {"path": "/DATA/Dop-NET"}, [
        {"split": "train"},
        {"split": "test"},
        {"split": "all"},
    ]),
    ("mvdoppler", {"path": "/DATA/MVDoppler"}, [{}]),
]


def mode_suffix(mode_kwargs: dict) -> str:
    """Generate a readable suffix from mode kwargs."""
    if not mode_kwargs:
        return "default"
    parts = []
    for k, v in sorted(mode_kwargs.items()):
        parts.append(f"{k}={v}")
    return "_".join(parts)


def compute_eigenvalues(cov: torch.Tensor) -> np.ndarray:
    """Compute sorted (descending) eigenvalues of a covariance matrix."""
    eigs = torch.linalg.eigvalsh(cov).numpy()
    return np.sort(eigs)[::-1]


def verify_dataset(name: str, base_config: dict, mode_kwargs: dict,
                   output_dir: Path, max_samples_check: int = 500):
    """
    Verify a single dataset configuration.

    Args:
        name: Dataset registration name
        base_config: Base config dict (path, etc.)
        mode_kwargs: Mode-specific kwargs
        output_dir: Where to write reports and figures
        max_samples_check: Max samples to load for stats (for speed)
    """
    suffix = mode_suffix(mode_kwargs)
    report_dir = output_dir / name / suffix
    report_dir.mkdir(parents=True, exist_ok=True)

    config = {"name": name, **base_config, **mode_kwargs}
    report_lines = []
    report_lines.append(f"{'='*70}")
    report_lines.append(f"Dataset: {name}")
    report_lines.append(f"Mode: {suffix}")
    report_lines.append(f"Config: {config}")
    report_lines.append(f"{'='*70}\n")

    try:
        ds = DatasetManager.create_dataset(config)
    except Exception as e:
        report_lines.append(f"ERROR loading dataset: {e}")
        _write_report(report_dir / "report.txt", report_lines)
        print(f"  FAIL [{name}/{suffix}]: {e}")
        return False

    n_samples = len(ds)
    n_classes = ds.get_num_classes() if hasattr(ds, 'get_num_classes') else len(ds.classes)
    classes = ds.classes if hasattr(ds, 'classes') else []

    report_lines.append(f"Total samples: {n_samples}")
    report_lines.append(f"Number of classes: {n_classes}")
    report_lines.append(f"Classes: {classes}\n")

    # Collect per-class stats
    class_data = defaultdict(lambda: {
        'count': 0,
        'eig_mins': [], 'eig_maxs': [], 'eig_means': [],
        'eig_geo_means': [],
        'data_mins': [], 'data_maxs': [], 'data_means': [],
        'sample_cov': None,  # store one example
        'sample_eigs': None,
    })

    step = max(1, n_samples // max_samples_check)
    indices = list(range(0, n_samples, step))
    cov_dim = None

    for i in indices:
        try:
            result = ds[i]
            if isinstance(result, tuple) and len(result) >= 2:
                cov, label = result[0], result[1]
            else:
                continue

            if isinstance(label, int):
                class_name = classes[label] if label < len(classes) else str(label)
            else:
                class_name = str(label)

            cov_np = cov.numpy() if isinstance(cov, torch.Tensor) else np.array(cov)

            if cov_dim is None:
                cov_dim = cov_np.shape
                report_lines.append(f"Covariance matrix dimension: {cov_dim}")

            eigs = compute_eigenvalues(cov)
            eigs_pos = eigs[eigs > 1e-15]

            cd = class_data[class_name]
            cd['count'] += 1
            cd['eig_mins'].append(eigs[-1])
            cd['eig_maxs'].append(eigs[0])
            cd['eig_means'].append(np.mean(eigs))
            if len(eigs_pos) > 0:
                cd['eig_geo_means'].append(
                    float(np.exp(np.mean(np.log(eigs_pos))))
                )
            cd['data_mins'].append(cov_np.min())
            cd['data_maxs'].append(cov_np.max())
            cd['data_means'].append(cov_np.mean())

            if cd['sample_cov'] is None:
                cd['sample_cov'] = cov_np
                cd['sample_eigs'] = eigs

        except Exception as e:
            report_lines.append(f"  Error at index {i}: {e}")
            continue

    report_lines.append(f"Samples checked: {len(indices)}\n")

    # Per-class report
    report_lines.append(f"\n{'='*70}")
    report_lines.append("PER-CLASS STATISTICS")
    report_lines.append(f"{'='*70}\n")

    sorted_classes = sorted(class_data.keys())

    for cls in sorted_classes:
        cd = class_data[cls]
        report_lines.append(f"--- Class: {cls} ---")
        report_lines.append(f"  Count (checked): {cd['count']}")

        if cd['eig_mins']:
            report_lines.append("  Eigenvalues:")
            report_lines.append(f"    min:      {np.mean(cd['eig_mins']):.6e} "
                                f"(range [{np.min(cd['eig_mins']):.6e}, "
                                f"{np.max(cd['eig_mins']):.6e}])")
            report_lines.append(f"    max:      {np.mean(cd['eig_maxs']):.6e} "
                                f"(range [{np.min(cd['eig_maxs']):.6e}, "
                                f"{np.max(cd['eig_maxs']):.6e}])")
            report_lines.append(f"    mean:     {np.mean(cd['eig_means']):.6e}")
            if cd['eig_geo_means']:
                report_lines.append(f"    geo_mean: {np.mean(cd['eig_geo_means']):.6e}")

        if cd['data_mins']:
            report_lines.append("  Data values:")
            report_lines.append(f"    min:  {np.mean(cd['data_mins']):.6e}")
            report_lines.append(f"    max:  {np.mean(cd['data_maxs']):.6e}")
            report_lines.append(f"    mean: {np.mean(cd['data_means']):.6e}")
        report_lines.append("")

    # Global summary
    all_geo_means = []
    for cd in class_data.values():
        all_geo_means.extend(cd['eig_geo_means'])
    if all_geo_means:
        report_lines.append(f"\nGlobal average geometric mean of eigenvalues: "
                            f"{np.mean(all_geo_means):.6e}")

    # ---- Generate figures ----
    n_plot_classes = min(len(sorted_classes), 12)
    plot_classes = sorted_classes[:n_plot_classes]

    # Figure 1: Sample covariance matrices (imshow)
    if plot_classes:
        n_cols = min(4, n_plot_classes)
        n_rows = (n_plot_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_plot_classes == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, cls in enumerate(plot_classes):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r, c]
            cd = class_data[cls]
            if cd['sample_cov'] is not None:
                im = ax.imshow(cd['sample_cov'], aspect='auto', cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{cls}", fontsize=8)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for idx in range(n_plot_classes, n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r, c].set_visible(False)

        fig.suptitle(f"{name} / {suffix} - Sample Covariance Matrices", fontsize=10)
        plt.tight_layout()
        fig.savefig(report_dir / "covariance_samples.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Figure 2: Eigenvalue decay plots
    if plot_classes:
        fig, ax = plt.subplots(figsize=(10, 6))
        for cls in plot_classes:
            cd = class_data[cls]
            if cd['sample_eigs'] is not None:
                eigs = cd['sample_eigs']
                ax.semilogy(range(len(eigs)), np.maximum(eigs, 1e-20),
                            label=cls, linewidth=0.8)
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.set_title(f'{name} / {suffix} - Eigenvalue Decay')
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(report_dir / "eigenvalue_decay.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Write report
    _write_report(report_dir / "report.txt", report_lines)
    print(f"  OK  [{name}/{suffix}]: {n_samples} samples, "
          f"{n_classes} classes, dim={cov_dim}")
    return True


def _write_report(path: Path, lines: list[str]):
    """Write report lines to a text file."""
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Verify all dataset loaders"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="tests/dataset_reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--datasets", nargs='*', default=None,
        help="Specific datasets to verify (default: all)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Max samples to check per dataset/mode (default: 500)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Registered datasets: {DatasetManager.get_available_datasets()}")
    print()

    results = {}
    for name, base_config, modes in DATASET_CONFIGS:
        if args.datasets and name not in args.datasets:
            continue

        print(f"\n--- Verifying: {name} ---")

        # Check data path exists
        data_path = base_config.get("path", "")
        if not Path(data_path).exists():
            print(f"  SKIP [{name}]: data path not found: {data_path}")
            results[name] = "SKIPPED"
            continue

        for mode_kwargs in modes:
            suffix = mode_suffix(mode_kwargs)
            try:
                ok = verify_dataset(
                    name, base_config, mode_kwargs,
                    output_dir, args.max_samples
                )
                results[f"{name}/{suffix}"] = "OK" if ok else "FAIL"
            except Exception as e:
                print(f"  FAIL [{name}/{suffix}]: {e}")
                results[f"{name}/{suffix}"] = f"ERROR: {e}"

    # Print summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    for key, status in results.items():
        print(f"  {key:40s} {status}")

    # Write summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DATASET VERIFICATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for key, status in results.items():
            f.write(f"{key:40s} {status}\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
