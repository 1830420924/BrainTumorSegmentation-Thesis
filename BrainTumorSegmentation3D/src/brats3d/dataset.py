from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .config import DEFAULT_CONFIG
from .manifest import get_split_records
from .utils import LoadedVolumePair, load_volume_pair


def _dataset_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = deepcopy(DEFAULT_CONFIG["dataset"])
    if config is None:
        return defaults
    provided = config.get("dataset", config)
    if not isinstance(provided, Mapping):
        raise ValueError("config.dataset must be a mapping.")
    for key, value in provided.items():
        if key == "augmentation" and isinstance(value, Mapping):
            defaults["augmentation"].update(value)
        else:
            defaults[key] = deepcopy(value)
    return defaults


def _spatial_tuple(value: Sequence[int], name: str = "patch_size") -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three integers in D,H,W order.")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers, received {result}.")
    return result


class _BaseBraTS3DDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path | Mapping[str, Any],
        split: str,
        config: Mapping[str, Any] | None = None,
        cache_size: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.config = _dataset_config(config)
        self.records = get_split_records(manifest_path, split)
        if not self.records:
            raise RuntimeError(f"No patients found in manifest split '{split}'.")

        configured_cache = int(self.config.get("cache_size", 0))
        self.cache_size = configured_cache if cache_size is None else int(cache_size)
        if self.cache_size < 0:
            raise ValueError("cache_size cannot be negative.")
        self.normalization_epsilon = float(self.config.get("normalization_epsilon", 1.0e-6))
        self._cache: OrderedDict[str, LoadedVolumePair] = OrderedDict()

    def _load_pair(self, record: Mapping[str, str]) -> LoadedVolumePair:
        patient_id = str(record["patient_id"])
        cached = self._cache.get(patient_id)
        if cached is not None:
            self._cache.move_to_end(patient_id)
            return cached

        pair = load_volume_pair(
            record["image_path"],
            record["mask_path"],
            normalization_epsilon=self.normalization_epsilon,
        )
        if self.cache_size > 0:
            self._cache[patient_id] = pair
            self._cache.move_to_end(patient_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return pair

    @staticmethod
    def _metadata(record: Mapping[str, str], pair: LoadedVolumePair) -> dict[str, Any]:
        return {
            "patient_id": str(record["patient_id"]),
            "image_path": str(record["image_path"]),
            "mask_path": str(record["mask_path"]),
            "affine": torch.as_tensor(pair.affine.copy(), dtype=torch.float32),
            "original_shape": torch.as_tensor(pair.original_shape, dtype=torch.int64),
            "spatial_shape_dhw": torch.as_tensor(pair.image.shape, dtype=torch.int64),
            "normalization_mean": torch.tensor(pair.normalization["mean"], dtype=torch.float32),
            "normalization_std": torch.tensor(pair.normalization["std"], dtype=torch.float32),
        }


class BraTS3DVolumeDataset(_BaseBraTS3DDataset):
    """Validation/test dataset returning one normalized complete volume per patient.

    Each sample uses ``[C,D,H,W]`` tensor order. Use ``batch_size=1`` for full
    volumes to keep host/GPU memory bounded and to support datasets with varying
    spatial shapes.
    """

    def __init__(
        self,
        manifest_path: str | Path | Mapping[str, Any],
        split: str,
        config: Mapping[str, Any] | None = None,
        cache_size: int = 0,
    ) -> None:
        super().__init__(manifest_path, split, config=config, cache_size=cache_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        pair = self._load_pair(record)
        sample = self._metadata(record, pair)
        sample.update(
            {
                "image": torch.from_numpy(pair.image).unsqueeze(0).to(dtype=torch.float32),
                "mask": torch.from_numpy(pair.mask).unsqueeze(0).to(dtype=torch.float32),
            }
        )
        return sample


def _crop_start(center: int, patch: int, dimension: int) -> int:
    if dimension <= patch:
        return 0
    return max(0, min(int(center) - patch // 2, dimension - patch))


def _extract_patch(
    array: np.ndarray,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    starts = tuple(
        _crop_start(center[axis], patch_size[axis], array.shape[axis]) for axis in range(3)
    )
    result = np.zeros(patch_size, dtype=array.dtype)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis, (start, patch, dimension) in enumerate(zip(starts, patch_size, array.shape)):
        source_end = min(start + patch, dimension)
        source_slices.append(slice(start, source_end))
        target_slices.append(slice(0, source_end - start))
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result, starts


def _patch_contains_tumor(
    mask: np.ndarray,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> bool:
    starts = tuple(
        _crop_start(center[axis], patch_size[axis], mask.shape[axis]) for axis in range(3)
    )
    slices = tuple(
        slice(start, min(start + patch, dimension))
        for start, patch, dimension in zip(starts, patch_size, mask.shape)
    )
    return bool(np.any(mask[slices]))


def _nonzero_bounds(image: np.ndarray) -> tuple[tuple[int, int], ...]:
    foreground = image != 0
    bounds: list[tuple[int, int]] = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(np.any(foreground, axis=other_axes))
        if occupied.size == 0:
            bounds.append((0, image.shape[axis]))
        else:
            bounds.append((int(occupied[0]), int(occupied[-1]) + 1))
    return tuple(bounds)


class BraTS3DPatchDataset(_BaseBraTS3DDataset):
    """Random positive/negative 3D patch sampling for memory-safe training."""

    def __init__(
        self,
        manifest_path: str | Path | Mapping[str, Any],
        split: str = "train",
        config: Mapping[str, Any] | None = None,
        patch_size: Sequence[int] | None = None,
        patches_per_patient: int | None = None,
        positive_patch_ratio: float | None = None,
        augment: bool | None = None,
        cache_size: int | None = None,
    ) -> None:
        super().__init__(manifest_path, split, config=config, cache_size=cache_size)
        self.patch_size = _spatial_tuple(
            patch_size or self.config.get("patch_size", [64, 96, 96])
        )
        self.patches_per_patient = int(
            patches_per_patient
            if patches_per_patient is not None
            else self.config.get("patches_per_patient", 4)
        )
        if self.patches_per_patient <= 0:
            raise ValueError("patches_per_patient must be positive.")
        self.positive_patch_ratio = float(
            positive_patch_ratio
            if positive_patch_ratio is not None
            else self.config.get("positive_patch_ratio", 0.75)
        )
        if not 0.0 <= self.positive_patch_ratio <= 1.0:
            raise ValueError("positive_patch_ratio must be between 0 and 1.")

        augmentation = dict(self.config.get("augmentation", {}))
        self.augment = bool(augmentation.get("enabled", True)) if augment is None else bool(augment)
        self.flip_probability = float(augmentation.get("flip_probability", 0.5))
        self.intensity_scale = float(augmentation.get("intensity_scale", 0.10))
        self.intensity_shift = float(augmentation.get("intensity_shift", 0.10))
        self.noise_std = float(augmentation.get("noise_std", 0.01))
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("augmentation.flip_probability must be between 0 and 1.")
        if min(self.intensity_scale, self.intensity_shift, self.noise_std) < 0:
            raise ValueError("Intensity augmentation magnitudes cannot be negative.")

        self.seed = int(self.config.get("random_seed", 42))
        self.epoch = 0
        self._worker_generators: dict[int, np.random.Generator] = {}

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_patient

    def set_epoch(self, epoch: int) -> None:
        """Reset patch streams deterministically when called by the trainer."""

        self.epoch = int(epoch)
        self._worker_generators.clear()

    def _generator(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        generator = self._worker_generators.get(worker_id)
        if generator is None:
            worker_offset = (worker_id + 1) * 1_000_003
            generator = np.random.default_rng(self.seed + self.epoch * 10_000_019 + worker_offset)
            self._worker_generators[worker_id] = generator
        return generator

    @staticmethod
    def _positive_center(mask: np.ndarray, generator: np.random.Generator) -> tuple[int, int, int] | None:
        positive_flat = np.flatnonzero(mask)
        if positive_flat.size == 0:
            return None
        selected = int(positive_flat[generator.integers(0, positive_flat.size)])
        return tuple(int(value) for value in np.unravel_index(selected, mask.shape))

    def _negative_center(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        generator: np.random.Generator,
    ) -> tuple[tuple[int, int, int], bool]:
        bounds = _nonzero_bounds(image)
        fallback = tuple(shape // 2 for shape in image.shape)
        for _ in range(64):
            center = tuple(
                int(generator.integers(lower, max(lower + 1, upper))) for lower, upper in bounds
            )
            fallback = center
            if image[center] == 0 or mask[center] != 0:
                continue
            if not _patch_contains_tumor(mask, center, self.patch_size):
                return center, True
        # A tumor-free patch may not exist for very large patches; this remains a
        # background-centred sample and is explicitly identified in metadata.
        return fallback, not _patch_contains_tumor(mask, fallback, self.patch_size)

    def _augment_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        generator: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        for axis in range(3):
            if generator.random() < self.flip_probability:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis)

        foreground = image != 0
        if self.intensity_scale > 0:
            scale = generator.uniform(1.0 - self.intensity_scale, 1.0 + self.intensity_scale)
            image[foreground] *= np.float32(scale)
        if self.intensity_shift > 0:
            shift = generator.uniform(-self.intensity_shift, self.intensity_shift)
            image[foreground] += np.float32(shift)
        if self.noise_std > 0 and np.any(foreground):
            noise = generator.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
            image[foreground] += noise[foreground]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient_index = (int(index) // self.patches_per_patient) % len(self.records)
        record = self.records[patient_index]
        pair = self._load_pair(record)
        generator = self._generator()

        request_positive = generator.random() < self.positive_patch_ratio
        center = self._positive_center(pair.mask, generator) if request_positive else None
        if center is not None:
            sampling = "positive"
        else:
            center, strictly_negative = self._negative_center(pair.image, pair.mask, generator)
            sampling = "negative" if strictly_negative else "background_centered_mixed"

        image_patch, origin = _extract_patch(pair.image, center, self.patch_size)
        mask_patch, _ = _extract_patch(pair.mask, center, self.patch_size)
        image_patch = image_patch.astype(np.float32, copy=False)
        mask_patch = mask_patch.astype(np.float32, copy=False)
        if self.augment:
            # Copies isolate augmentation from the optional full-volume cache.
            image_patch, mask_patch = self._augment_patch(
                image_patch.copy(), mask_patch.copy(), generator
            )

        sample = self._metadata(record, pair)
        sample.update(
            {
                "image": torch.from_numpy(np.ascontiguousarray(image_patch)).unsqueeze(0),
                "mask": torch.from_numpy(np.ascontiguousarray(mask_patch)).unsqueeze(0),
                "patch_origin_dhw": torch.as_tensor(origin, dtype=torch.int64),
                "patch_center_dhw": torch.as_tensor(center, dtype=torch.int64),
                "sampling": sampling,
            }
        )
        return sample


class BrainTumorVolumeDataset(Dataset):
    """Compatibility facade selecting patch training or complete-volume loading."""

    def __init__(
        self,
        manifest_path: str | Path | Mapping[str, Any],
        split: str,
        training: bool | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        use_patches = split == "train" if training is None else bool(training)
        dataset_class = BraTS3DPatchDataset if use_patches else BraTS3DVolumeDataset
        self.dataset = dataset_class(manifest_path, split, config=config, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


BrainTumor3DDataset = BrainTumorVolumeDataset

