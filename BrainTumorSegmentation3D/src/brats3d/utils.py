from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class LoadedVolumePair:
    """A normalized FLAIR volume and binary mask in PyTorch D,H,W order."""

    image: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    original_shape: tuple[int, int, int]
    normalization: dict[str, float]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically save a small JSON metadata file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on the local environment.
        raise RuntimeError(
            "nibabel is required to read BraTS NIfTI volumes. "
            "Install the 3D project requirements before preparing the manifest."
        ) from exc
    return nib


def list_patient_dirs(training_root: str | Path) -> list[Path]:
    training_root = Path(training_root)
    if not training_root.exists():
        raise FileNotFoundError(f"BraTS training root not found: {training_root}")
    if not training_root.is_dir():
        raise NotADirectoryError(f"BraTS training root is not a directory: {training_root}")
    return sorted(path for path in training_root.iterdir() if path.is_dir())


def resolve_volume_path(patient_dir: str | Path, patient_id: str, suffix: str) -> Path:
    """Resolve a BraTS .nii/.nii.gz path, including known non-standard seg names."""

    patient_dir = Path(patient_dir)
    direct_candidates = (
        patient_dir / f"{patient_id}_{suffix}.nii",
        patient_dir / f"{patient_id}_{suffix}.nii.gz",
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    # A small number of public mirrors use an unusual segmentation filename.
    patterns = (
        f"*_{suffix}.nii",
        f"*_{suffix}.nii.gz",
        "*Seg*.nii",
        "*Seg*.nii.gz",
        "*seg*.nii",
        "*seg*.nii.gz",
    )
    matches: list[Path] = []
    for pattern in patterns if suffix.lower() == "seg" else patterns[:2]:
        matches.extend(patient_dir.glob(pattern))
    unique = sorted({match.resolve() for match in matches})
    if len(unique) == 1:
        return unique[0]

    expected = direct_candidates[0]
    if not unique:
        raise FileNotFoundError(f"Missing {suffix} volume for {patient_id}: expected {expected}")
    raise RuntimeError(
        f"Ambiguous {suffix} volumes for {patient_id}: "
        + ", ".join(path.name for path in unique)
    )


def load_nifti_array(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI proxy directly as float32 without creating a float64 copy."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI volume not found: {path}")
    nib = require_nibabel()
    image = nib.load(str(path), mmap=True)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {image.shape} for {path}")
    array = np.asarray(image.dataobj, dtype=np.float32)
    return array, np.asarray(image.affine, dtype=np.float32)


def normalize_nonzero_volume(
    volume: np.ndarray,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Z-score the complete volume using all finite non-zero brain voxels.

    Background remains exactly zero. Statistics are computed before any patch is
    selected, preventing patch-specific normalization and train/evaluation drift.
    """

    volume = np.asarray(volume, dtype=np.float32)
    foreground = np.isfinite(volume) & (volume != 0)
    output = np.zeros(volume.shape, dtype=np.float32)
    if not np.any(foreground):
        return output, {"mean": 0.0, "std": 1.0, "nonzero_voxels": 0.0}

    values = volume[foreground]
    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64))
    if not np.isfinite(std) or std < float(epsilon):
        std = 1.0
    output[foreground] = (values - mean) / std
    return output, {
        "mean": mean,
        "std": std,
        "nonzero_voxels": float(values.size),
    }


def xyz_to_dhw(volume: np.ndarray) -> np.ndarray:
    """Convert nibabel's X,Y,Z storage order to Conv3d's D,H,W order."""

    if volume.ndim != 3:
        raise ValueError(f"xyz_to_dhw expects a 3D array, got {volume.shape}.")
    return np.ascontiguousarray(np.transpose(volume, (2, 0, 1)))


def dhw_to_xyz(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"dhw_to_xyz expects a 3D array, got {volume.shape}.")
    return np.ascontiguousarray(np.transpose(volume, (1, 2, 0)))


def load_volume_pair(
    image_path: str | Path,
    mask_path: str | Path,
    normalization_epsilon: float = 1.0e-6,
) -> LoadedVolumePair:
    """Load FLAIR and segmentation, create ``seg > 0`` binary labels, normalize."""

    image_xyz, affine = load_nifti_array(image_path)
    segmentation_xyz, mask_affine = load_nifti_array(mask_path)
    if image_xyz.shape != segmentation_xyz.shape:
        raise ValueError(
            f"FLAIR/seg shape mismatch: {Path(image_path).name}={image_xyz.shape}, "
            f"{Path(mask_path).name}={segmentation_xyz.shape}"
        )
    if not np.allclose(affine, mask_affine, atol=1.0e-4):
        raise ValueError(
            f"FLAIR/seg affine mismatch for {Path(image_path).name} and {Path(mask_path).name}."
        )

    normalized_xyz, statistics = normalize_nonzero_volume(image_xyz, normalization_epsilon)
    binary_mask_xyz = (np.isfinite(segmentation_xyz) & (segmentation_xyz > 0)).astype(np.uint8)
    return LoadedVolumePair(
        image=xyz_to_dhw(normalized_xyz),
        mask=xyz_to_dhw(binary_mask_xyz),
        affine=affine,
        original_shape=tuple(int(value) for value in image_xyz.shape),
        normalization=statistics,
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer compatible with Windows spawn workers."""

    del worker_id
    try:
        import torch

        worker_seed = int(torch.initial_seed() % (2**32))
    except ImportError:  # pragma: no cover
        worker_seed = random.randrange(2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

