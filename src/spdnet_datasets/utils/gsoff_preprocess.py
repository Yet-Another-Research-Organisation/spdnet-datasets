"""
GSOFF (Gaofeng State Owned Forest Farm) Dataset Preprocessing
Extracts windows from hyperspectral GeoTIFF imagery.
Simplified folder naming: just 'GSOFF'
"""

import numpy as np
import rasterio
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Configuration
WINDOW_SIZE = 12
THRESHOLD = 0.8


def load_images(test_path, gt_path):
    """Load test and GT images with rasterio."""
    print("Loading images...")
    print(f"  Test image: {test_path}")
    print(f"  GT image: {gt_path}")

    with rasterio.open(test_path) as src:
        test_image = src.read()
        test_profile = src.profile

    with rasterio.open(gt_path) as src:
        gt_image = src.read(1)

    # Convert to (height, width, bands)
    test_image = np.transpose(test_image, (1, 2, 0))

    print(f"  Test image shape: {test_image.shape}")
    print(f"  GT image shape: {gt_image.shape}")

    return test_image, gt_image


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


def extract_and_save(test_image, gt_image, best_offsets, output_dir):
    """Extract and save windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []

    for class_id, offset_data in tqdm(best_offsets.items(), desc="Extracting windows"):
        offset_y, offset_x = offset_data['offset']

        y = offset_y
        while y + WINDOW_SIZE <= test_image.shape[0]:
            x = offset_x
            while x + WINDOW_SIZE <= test_image.shape[1]:
                gt_window = gt_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                is_pure, label = check_window_purity(gt_window)

                if is_pure and label == class_id:
                    data_window = test_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, :]
                    all_windows.append(data_window)
                    all_labels.append(class_id)

                x += WINDOW_SIZE
            y += WINDOW_SIZE

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)

    np.save(output_dir / 'gaofeng_windows_data.npy', all_windows)
    np.save(output_dir / 'gaofeng_windows_labels.npy', all_labels)

    metadata = {
        'total_windows': len(all_windows),
        'window_size': WINDOW_SIZE,
        'threshold': THRESHOLD,
        'shape': list(all_windows.shape),
        'dataset': 'GSOFF'
    }

    with open(output_dir / 'gaofeng_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Total extracted windows: {len(all_windows)}")
    print(f"  Windows shape: {all_windows.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract windows from GSOFF dataset")
    parser.add_argument('--data-dir', type=str,
                        default='/media/mgallet/BACK UP/DATASET/Hyperspectral/Gaofeng State Owned Forest Farm/data/testarea2_new',
                        help='Directory containing GeoTIFF files')
    parser.add_argument('--output-dir', type=str,
                        default='/media/mgallet/BACK UP/DATASET/Hyperspectral/GSOFF',
                        help='Output directory')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("GSOFF Dataset Preprocessing")
    print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}, Threshold: {THRESHOLD*100}%")
    print("="*80)

    # Load images
    test_path = data_dir / 'sg_quac_test2.tif'
    gt_path = data_dir / 'sg_quac_test2_gt.tif'

    test_image, gt_image = load_images(test_path, gt_path)

    # Find best offsets
    best_offsets = find_best_offsets_per_class(gt_image)

    # Extract and save
    extract_and_save(test_image, gt_image, best_offsets, output_dir)

    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()
