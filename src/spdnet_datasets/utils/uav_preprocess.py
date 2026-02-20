"""
UAV-HSI-Crop Dataset Preprocessing
Extracts windows from UAV hyperspectral patches and saves them for training.
Simplified folder naming: just the scene name (MJK_N, MJK_S, XJM)
"""

import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Configuration
WINDOW_SIZE = 15
THRESHOLD = 0.8  # 80% of pixels must belong to a class
BAND_STEP = 1  # Select every BAND_STEP band


def reconstruct_image(group_name, base_path):
    """Reconstruct full image and GT from patches for a given group."""
    print(f"Reconstructing images for {group_name}...")

    gt_patches = {}
    rs_patches = {}

    base_dirs = [
        base_path / 'Test',
        base_path / 'Train' / 'Training'
    ]

    for base_dir in base_dirs:
        gt_dir = base_dir / 'gt'
        rs_dir = base_dir / 'rs'

        if not gt_dir.exists() or not rs_dir.exists():
            continue

        for file in gt_dir.iterdir():
            if file.name.startswith(f'{group_name}_patch_') and file.name.endswith('.npy'):
                # Parse row and col from filename
                suffix = file.name[len(f'{group_name}_patch_'):-4]
                numbers = suffix.split('_')
                row = int(numbers[0])
                col = int(numbers[1])

                gt_patch = np.load(str(file))
                rs_patch = np.load(str(rs_dir / file.name))

                # Apply band selection
                rs_patch = rs_patch[:, :, ::BAND_STEP]

                gt_patches[(row, col)] = gt_patch
                rs_patches[(row, col)] = rs_patch

    if not gt_patches:
        raise ValueError(f"No patches found for {group_name}")

    # Get actual patch size from first patch
    first_patch = gt_patches[next(iter(gt_patches))]
    patch_size = first_patch.shape[0]

    # Find dimensions
    max_row = max(r for r, c in gt_patches.keys())
    max_col = max(c for r, c in gt_patches.keys())

    height = (max_row + 1) * patch_size
    width = (max_col + 1) * patch_size
    bands = rs_patches[next(iter(rs_patches))].shape[2]

    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Reconstructed image size: {height}x{width}x{bands}")

    # Create large arrays
    gt_image = np.zeros((height, width), dtype=gt_patches[next(iter(gt_patches))].dtype)
    test_image = np.zeros((height, width, bands), dtype=rs_patches[next(iter(rs_patches))].dtype)

    # Place patches
    for (row, col), gt_patch in gt_patches.items():
        rs_patch = rs_patches[(row, col)]
        y_start = row * patch_size
        x_start = col * patch_size
        gt_image[y_start:y_start+patch_size, x_start:x_start+patch_size] = gt_patch
        test_image[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = rs_patch

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
    """Find the best offset for each class by testing on GT only."""
    height, width = gt_image.shape
    unique_classes = np.unique(gt_image[gt_image > 0])

    print(f"  Classes detected: {len(unique_classes)} ({unique_classes.tolist()})")

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

        best_offsets[class_id] = {
            'offset': best_offset,
            'count': best_count
        }
        print(f"    Class {class_id}: {best_count} windows (offset: {best_offset})")

    return best_offsets


def extract_all_windows(test_image, gt_image, best_offsets):
    """Extract all hyperspectral windows using the best offsets."""
    height, width, bands = test_image.shape
    windows_per_class = {}

    for class_id, offset_data in tqdm(best_offsets.items(), desc="Extracting windows"):
        offset_y, offset_x = offset_data['offset']
        windows = []

        y = offset_y
        while y + WINDOW_SIZE <= height:
            x = offset_x
            while x + WINDOW_SIZE <= width:
                gt_window = gt_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                is_pure, label = check_window_purity(gt_window)

                if is_pure and label == class_id:
                    data_window = test_image[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE, :]
                    windows.append((data_window, (y, x)))

                x += WINDOW_SIZE
            y += WINDOW_SIZE

        windows_per_class[class_id] = {
            'windows': windows,
            'offset': offset_data['offset'],
            'count': len(windows)
        }

    return windows_per_class


def save_windows(windows_per_class, output_dir, scene_name):
    """Save the extracted windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving windows...")

    all_windows = []
    all_labels = []
    all_positions = []
    metadata = {}

    for class_id, class_data in tqdm(windows_per_class.items(), desc="Saving"):
        windows = class_data['windows']

        for window_data, position in windows:
            all_windows.append(window_data)
            all_labels.append(class_id)
            all_positions.append(position)

        metadata[int(class_id)] = {
            'count': class_data['count'],
            'offset': class_data['offset']
        }

    # Convert to numpy arrays
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_positions = np.array(all_positions)

    # Save
    np.save(output_dir / 'uav_windows_data.npy', all_windows)
    np.save(output_dir / 'uav_windows_labels.npy', all_labels)
    np.save(output_dir / 'uav_windows_positions.npy', all_positions)

    # Save metadata
    metadata['total_windows'] = len(all_windows)
    metadata['window_size'] = WINDOW_SIZE
    metadata['threshold'] = THRESHOLD
    metadata['band_step'] = BAND_STEP
    metadata['bands'] = all_windows.shape[-1]
    metadata['shape'] = list(all_windows.shape)
    metadata['dataset'] = 'UAV-HSI-Crop'
    metadata['scene'] = scene_name

    with open(output_dir / 'uav_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Total extracted windows: {len(all_windows)}")
    print(f"  Windows shape: {all_windows.shape}")
    print(f"  Files saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract windows from UAV-HSI-Crop dataset")
    parser.add_argument('--data-dir', type=str, default='/media/mgallet/BACK UP/DATASET/Hyperspectral/UAV-HSI-Crop-Dataset',
                        help='Path to UAV-HSI-Crop-Dataset folder')
    parser.add_argument('--output-dir', type=str, default='/media/mgallet/BACK UP/DATASET/Hyperspectral/UAV-HSI-Crop',
                        help='Output directory')

    args = parser.parse_args()

    base_path = Path(args.data_dir)
    output_base = Path(args.output_dir)

    scenes = ['MJK_N', 'MJK_S', 'XJM']

    print("="*80)
    print(f"UAV-HSI-Crop Dataset Preprocessing")
    print(f"Window size: {WINDOW_SIZE}x{WINDOW_SIZE}, Threshold: {THRESHOLD*100}%")
    print("="*80)

    for scene in scenes:
        print(f"\n{'='*80}")
        print(f"Processing scene: {scene}")
        print(f"{'='*80}")

        try:
            # Reconstruct images
            test_image, gt_image = reconstruct_image(scene, base_path)

            # Find best offsets
            best_offsets = find_best_offsets_per_class(gt_image)

            # Extract windows
            windows_per_class = extract_all_windows(test_image, gt_image, best_offsets)

            # Save with simplified naming (just scene name)
            output_dir = output_base / scene
            save_windows(windows_per_class, output_dir, scene)

            print(f"  Scene {scene} completed!")

        except Exception as e:
            print(f"  Error processing {scene}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All scenes processed!")
    print("="*80)


if __name__ == "__main__":
    main()
