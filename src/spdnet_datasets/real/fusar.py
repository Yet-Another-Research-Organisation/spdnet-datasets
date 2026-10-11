"""
FUSAR-Ship dataset loader (Gaofen-3 SAR ship chips, single channel).

Reference: Hou et al., "FUSAR-Ship: a high-resolution SAR-AIS matchup dataset
of Gaofen-3 for ship detection and recognition", Science China Information
Sciences, 2020.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from spdnet_datasets.base import BaseDataset
from spdnet_datasets.manager import DatasetManager


@DatasetManager.register_dataset("fusar")
class FUSARShipDataset(BaseDataset):
    """
    FUSAR-Ship SAR ship chips (Gaofen-3), single channel, for CNN backbones.

    Expected structure::

        data_dir/
            <Category>/<SubCategory>/Ship_CxxSyyNzzzz.tiff   # 512x512 uint8

    The classes are the categories (``class_level="basic"``: Cargo, Fishing,
    Tanker, ...) or the ``Category_SubCategory`` pairs (``"fine"``). The
    dataset is very imbalanced (from about 2100 Cargo chips down to 3
    DiveVessel ones): classes with fewer than ``min_samples_per_class``
    chips are dropped, then ``max_samples_per_class`` caps the others
    (drawn with ``seed``). Use a stratified split
    (``create_dataloaders(stratified=True)``).

    Each chip is returned as a float32 tensor ``(1, image_size, image_size)``:
    ``log1p`` of the intensity (SAR amplitudes are heavy tailed), resized
    with antialiasing, then standardized per image (mean 0, std 1).
    """

    def __init__(
        self,
        class_level: str = "basic",
        min_samples_per_class: int = 30,
        exclude_classes: list[str] | None = None,
        image_size: int = 224,
        preload: bool = False,
        **kwargs,
    ):
        """
        Initialize the FUSAR-Ship dataset.

        Args:
            class_level: 'basic' (categories) or 'fine' (category_subcategory)
            min_samples_per_class: classes with fewer chips are dropped
            exclude_classes: class names to drop (e.g. ['Other', 'Unspecified'])
            image_size: side of the square output image, in pixels
            preload: load every chip into memory at construction
            **kwargs: BaseDataset arguments (data_dir, max_samples_per_class,
                max_classes, classes, seed, verbose)
        """
        if class_level not in ("basic", "fine"):
            raise ValueError(
                f"class_level must be 'basic' or 'fine', got {class_level}"
            )
        self.class_level = class_level
        self.min_samples_per_class = min_samples_per_class
        self.exclude_classes = set(exclude_classes or [])
        self.image_size = image_size
        self.preload = preload
        self.target_size = (1, image_size, image_size)
        self._cache: dict[int, torch.Tensor] = {}
        super().__init__(**kwargs)
        self._load_data()
        if self.preload:
            for idx in range(len(self.samples)):
                self._cache[idx] = self._load_image(self.samples[idx][0])

    def _class_of(self, path: Path) -> str:
        category, subcategory = path.relative_to(self.data_dir).parts[:2]
        return category if self.class_level == "basic" else f"{category}_{subcategory}"

    def _load_data(self):
        paths_by_class: dict[str, list[Path]] = {}
        for path in sorted(self.data_dir.glob("*/*/*.tif*")):
            paths_by_class.setdefault(self._class_of(path), []).append(path)
        if not paths_by_class:
            raise FileNotFoundError(f"No FUSAR TIFF chips found under {self.data_dir}")

        kept = sorted(
            name
            for name, paths in paths_by_class.items()
            if len(paths) >= self.min_samples_per_class
            and name not in self.exclude_classes
        )
        if self.verbose:
            dropped = sorted(set(paths_by_class) - set(kept))
            print(
                f"FUSAR ({self.class_level}): {len(kept)} classes kept, "
                f"{len(dropped)} dropped (< {self.min_samples_per_class} chips "
                f"or excluded): {dropped}"
            )
        self.classes = self._limit_classes(kept)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        samples_by_class = {
            name: [(path, self.class_to_idx[name]) for path in paths_by_class[name]]
            for name in self.classes
        }
        self.samples = self._limit_samples_per_class(samples_by_class)

    def _load_image(self, path: Path) -> torch.Tensor:
        array = tifffile.imread(path)
        if array.ndim == 3:  # a few chips may carry a trailing channel axis
            array = array[..., 0]
        image = torch.from_numpy(np.log1p(array.astype(np.float32)))[None, None]
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        image = image[0]
        return (image - image.mean()) / (image.std() + 1e-6)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = self._cache.get(idx)
        if image is None:
            image = self._load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.classes)
