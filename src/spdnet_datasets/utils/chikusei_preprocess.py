"""
Chikusei Dataset Preprocessing
Extracts windows from hyperspectral .mat files.
Simplified folder naming: just 'Chikusei'
"""

import numpy as np
import h5py
import scipy.io
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Configuration
WINDOW_SIZE = 16
THRESHOLD = 0.75


def load_chikusei_data(data_dir):
    """Load Chikusei data (hyperspectral image and ground truth)."""
    print("Loading Chikusei data...")

    hyperspectral_path = data_dir / 'HyperspecVNIR_Chikusei_20140729.mat'
    gt_path = data_dir / 'HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'

    # Load hyperspectral image with h5py
    with h5py.File(hyperspectral_path, 'r') as f:
        hyperspectral_data = f['chikusei'][:]  # Shape: (128, 2335, 2517)

    # Load ground truth
    gt_data = scipy.io.loadmat(str(gt_path))
    gt_struct = gt_data['GT'][0, 0]
    gt_image = gt_struct['gt']
    class_names = gt_struct['classnames']

    # Convert to (height, width, bands)
    hyperspectral_data = np.transpose(hyperspectral_data, (1, 2, 0))

    print(f"  Hyperspectral shape: {hyperspectral_data.shape}")
    print(f"  GT shape: {gt_image.shape}")

    class_names_list = [str(item[0]) for item in class_names[0]] if class_names is not None else None

    return hyperspectral_data, gt_image, class_names_list


def check_window_purity(gt_window):
    """Check if a GT window contains more than THRESHOLD% pixels of the same class."""
    valid_pixels = gt_window[gt_window > 0]

    if len(valid_pixels) == 0:
        return False, 0

    unique, counts = np.unique(valid_pixels, return_counts=True)
    max_count = np.max(counts)
    dominant_class = unique[np.argmax(counts)]

    purity = max_count / len(valid_pixels)

    if purity >= THRESHOLD:
        return True, dominant_class

    return False, 0


def find_best_offsets_per_class(gt_image):
    """Find the best offset for each class."""
    height, width = gt_image.shape
    unique_classes = np.unique(gt_image[gt_image > 0])

    print(f"  Classes detected: {len(unique_classes)}")

    best_offsets = {}

    for class_id in tqdm(unique_classes, desc="Finding best offsets"):
        best_count = 0
        best_offset = (0, 0)

        for offset_y in range(0, WINDOW_SIZE):
            for offset_x in range(0, WINDOW_SIZE):
                count = 0
                y = offset_y
                while y + WINDOW_SIZE <= height:
                    x = offset_x
                    while x + WINDOW_SIZE <= width:
                        gt_window = gt_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                        is_pure, label = check_window_purity(gt_window)
                        if is_pure and label == class_id:
                            count += 1
                        x += WINDOW_SIZE
                    y += WINDOW_SIZE

                if count > best_count:
                    best_count = count
                    best_offset = (offset_y, offset_x)

        best_offsets[class_id] = {'offset': best_offset, 'count': best_count}

    return best_offsets


def extract_and_save(hyperspectral_data, gt_image, best_offsets, output_dir, class_names_list):
    """Extract and save windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []

    for class_id, offset_data in tqdm(best_offsets.items(), desc="Extracting windows"):
        offset_y, offset_x = offset_data['offset']

        y = offset_y
        while y + WINDOW_SIZE <= hyperspectral_data.shape[0]:
            x = offset_x
            while x + WINDOW_SIZE <= hyperspectral_data.shape[1]:
                gt_window = gt_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                is_pure, label = check_window_purity(gt_window)

                if is_pure and label == class_id:
                    data_window = hyperspectral_data[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, :]
                    all_windows.append(data_window)
                    all_labels.append(class_id)

                x += WINDOW_SIZE
            y += WINDOW_SIZE

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)

    np.save(output_dir / 'chikusei_windows_data.npy', all_windows)
    np.save(output_dir / 'chikusei_windows_labels.npy', all_labels)

    metadata = {
        'total_windows': len(all_windows),
        'window_size': WINDOW_SIZE,
        'threshold': THRESHOLD,
        'shape': list(all_windows.shape),
        'dataset': 'Chikusei',
        'class_names': class_names_list
    }

    with open(output_dir / 'chikusei_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Total extracted windows: {len(all_windows)}")
    print(f"  Windows shape: {all_windows.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract windows from Chikusei dataset")
    parser.add_argument('--data-dir', type=str,
                        default='/media/mgallet/BACK UP/DATASET/Hyperspectral/Chikusei',
                        help='Directory containing .mat files')
    parser.add_argument('--output-dir', type=str,
                        default='/media/mgallet/BACK UP/DATASET/Hyperspectral/Chikusei',
                        help='Output directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("Chikusei Dataset Preprocessing")
    print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}, Threshold: {THRESHOLD*100}%")
    print("="*80)

    # Load data
    hyperspectral_data, gt_image, class_names_list = load_chikusei_data(data_dir)

    # Find best offsets
    best_offsets = find_best_offsets_per_class(gt_image)

    # Extract and save
    extract_and_save(hyperspectral_data, gt_image, best_offsets, output_dir, class_names_list)

    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()
