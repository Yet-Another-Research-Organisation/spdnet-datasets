"""
Hyperspectral Placenta Dataset Preprocessing
Extracts windows from hyperspectral TIFF files with segmentation masks.
Window size: 12x12, Purity threshold: 80%
"""

import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from .placenta_utils import read_stiff, read_mtiff

# Configuration
WINDOW_SIZE = 24
THRESHOLD = 0.90

# Class mapping: merge ICG variants with base classes
CLASS_MAPPING = {
    'Artery': 'Artery',
    'Artery, ICG': 'Artery',
    'Stroma': 'Stroma',
    'Stroma, ICG': 'Stroma',
    'Umbilical cord': 'Umbilical',
    'Umbilical cord, ICG': 'Umbilical',
    'Suture': 'Suture',
    'Suture, ICG': 'Suture',
    'Vein': 'Vein',
    'Vein, ICG': 'Vein',
    # Ignore Specular reflection
}

# Dye types for different folders
DYE_FOLDERS = {
    'Placenta P007 - P030 red blue': 'red_blue',
    'Placenta P031 - P053 red blue': 'red_blue',
    'Placenta P054 - P077 ICG': 'ICG',
    'Placenta P078 - P101 ICG': 'ICG',
}


def check_window_purity(mask_window):
    """
    Check if a mask window contains more than THRESHOLD% pixels of True.

    Args:
        mask_window: Boolean array of shape (H, W)

    Returns:
        is_pure: Boolean indicating if window meets purity threshold
    """
    if mask_window.size == 0:
        return False

    true_count = np.sum(mask_window)
    purity = true_count / mask_window.size

    return purity >= THRESHOLD


def extract_windows_from_image(data_cube, masks, dye_type, sample_id):
    """
    Extract windows from a single hyperspectral image.

    Args:
        data_cube: Hyperspectral image (H, W, C)
        masks: Dictionary of class_name -> boolean mask
        dye_type: 'red_blue' or 'ICG'
        sample_id: Sample identifier (e.g., 'P007')

    Returns:
        windows_data: List of extracted windows
        windows_labels: List of class labels
        windows_metadata: List of metadata dicts
    """
    height, width, n_bands = data_cube.shape

    windows_data = []
    windows_labels = []
    windows_metadata = []

    # Process each class
    for class_name, mask in masks.items():
        # Skip specular reflection
        if 'Specular' in class_name or 'specular' in class_name:
            continue

        # Map to unified class name
        unified_class = CLASS_MAPPING.get(class_name)
        if unified_class is None:
            print(f"  Warning: Unknown class '{class_name}' - skipping")
            continue

        # Extract windows with sliding window
        window_count = 0

        for y in range(0, height - WINDOW_SIZE + 1, WINDOW_SIZE):
            for x in range(0, width - WINDOW_SIZE + 1, WINDOW_SIZE):
                mask_window = mask[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]

                if check_window_purity(mask_window):
                    data_window = data_cube[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, :]

                    windows_data.append(data_window)
                    windows_labels.append(unified_class)
                    windows_metadata.append({
                        'sample_id': sample_id,
                        'dye_type': dye_type,
                        'class': unified_class,
                        'original_class': class_name,
                        'position': (y, x)
                    })
                    window_count += 1

        if window_count > 0:
            print(f"    {class_name} -> {unified_class}: {window_count} windows")

    return windows_data, windows_labels, windows_metadata


def process_folder(folder_path, dye_type, output_dir):
    """
    Process all samples in a folder.

    Args:
        folder_path: Path to folder containing .tif files
        dye_type: 'red_blue' or 'ICG'
        output_dir: Output directory for extracted windows

    Returns:
        all_windows: List of all extracted windows
        all_labels: List of all labels
        all_metadata: List of all metadata
    """
    print(f"\nProcessing folder: {folder_path.name} (dye: {dye_type})")

    all_windows = []
    all_labels = []
    all_metadata = []
    expected_bands = None  # Will be set from first image

    # Get all .tif files (not mask files)
    tif_files = sorted([f for f in folder_path.glob("P*.tif") if "mask" not in f.name])

    for tif_file in tqdm(tif_files, desc=f"  Processing {dye_type}"):
        sample_id = tif_file.stem  # e.g., 'P007'
        mask_file = tif_file.parent / f"{sample_id}, masks.tif"

        if not mask_file.exists():
            print(f"  Warning: Mask file not found for {sample_id}")
            continue

        try:
            # Load data
            data_cube, center_wavelengths, preview_image, metadata = read_stiff(str(tif_file))
            masks = read_mtiff(str(mask_file))

            # Check number of bands
            n_bands = data_cube.shape[-1]
            if expected_bands is None:
                expected_bands = n_bands
                print(f"  Expected number of bands: {expected_bands}")
            elif n_bands != expected_bands:
                print(f"  Warning: {sample_id} has {n_bands} bands, expected {expected_bands} - skipping")
                continue

            # Extract windows
            windows, labels, meta = extract_windows_from_image(
                data_cube, masks, dye_type, sample_id
            )

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_metadata.extend(meta)

        except Exception as e:
            print(f"  Error processing {sample_id}: {e}")
            continue

    return all_windows, all_labels, all_metadata


def save_processed_data(all_windows, all_labels, all_metadata, output_dir):
    """
    Save processed data to disk.

    Args:
        all_windows: List of windows
        all_labels: List of labels
        all_metadata: List of metadata dicts
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to arrays
    windows_array = np.array(all_windows, dtype=np.float32)
    labels_array = np.array(all_labels)

    # Save arrays
    np.save(output_dir / 'placenta_windows_data.npy', windows_array)
    np.save(output_dir / 'placenta_windows_labels.npy', labels_array)

    # Save metadata
    with open(output_dir / 'placenta_windows_metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Create summary
    unique_labels = sorted(set(all_labels))
    class_counts = {label: all_labels.count(label) for label in unique_labels}

    # Count by dye type
    dye_counts = {'red_blue': 0, 'ICG': 0}
    dye_class_counts = {'red_blue': {}, 'ICG': {}}

    for meta in all_metadata:
        dye = meta['dye_type']
        cls = meta['class']
        dye_counts[dye] += 1
        dye_class_counts[dye][cls] = dye_class_counts[dye].get(cls, 0) + 1

    summary = {
        'total_windows': len(windows_array),
        'window_size': WINDOW_SIZE,
        'threshold': THRESHOLD,
        'shape': list(windows_array.shape),
        'n_bands': windows_array.shape[-1],
        'dataset': 'Hyperspectral_Placenta',
        'classes': unique_labels,
        'n_classes': len(unique_labels),
        'class_counts': class_counts,
        'dye_counts': dye_counts,
        'dye_class_counts': dye_class_counts
    }

    with open(output_dir / 'placenta_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Extraction Summary:")
    print(f"{'='*80}")
    print(f"Total windows: {len(windows_array)}")
    print(f"Window shape: {windows_array.shape}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"\nClass distribution:")
    for label in unique_labels:
        count = class_counts[label]
        pct = 100 * count / len(windows_array)
        print(f"  {label:15s}: {count:5d} windows ({pct:5.1f}%)")
    print(f"\nDye type distribution:")
    for dye, count in dye_counts.items():
        pct = 100 * count / len(windows_array)
        print(f"  {dye:10s}: {count:5d} windows ({pct:5.1f}%)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract windows from Hyperspectral Placenta dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/media/mgallet/BACK UP/DATASET/Hyperspectral_Placenta_Dataset',
        help='Directory containing Placenta folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/media/mgallet/BACK UP/DATASET/Hyperspectral_Placenta_Dataset/imagettes',
        help='Output directory for extracted windows'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("Hyperspectral Placenta Dataset Preprocessing")
    print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}, Threshold: {THRESHOLD*100}%")
    print("="*80)

    all_windows = []
    all_labels = []
    all_metadata = []

    # Process each folder
    for folder_name, dye_type in DYE_FOLDERS.items():
        folder_path = data_dir / folder_name

        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue

        windows, labels, metadata = process_folder(folder_path, dye_type, output_dir)
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_metadata.extend(metadata)

    # Save all data
    if len(all_windows) > 0:
        save_processed_data(all_windows, all_labels, all_metadata, output_dir)
    else:
        print("\nNo windows extracted!")

    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()
