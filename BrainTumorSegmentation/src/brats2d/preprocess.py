from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from .deps import require_nibabel
from .utils import ensure_dir, save_json


EXPECTED_TRAIN_SUFFIXES = ("flair", "t1", "t1ce", "t2", "seg")
EXPECTED_VAL_SUFFIXES = ("flair", "t1", "t1ce", "t2")


@dataclass
class PatientRecord:
    patient_id: str
    flair_path: Path
    seg_path: Path | None


def list_patient_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    return sorted([path for path in root.iterdir() if path.is_dir()])


def resolve_volume_path(patient_dir: Path, patient_id: str, suffix: str) -> Path:
    nii_path = patient_dir / f"{patient_id}_{suffix}.nii"
    if nii_path.exists():
        return nii_path
    nii_gz_path = patient_dir / f"{patient_id}_{suffix}.nii.gz"
    if nii_gz_path.exists():
        return nii_gz_path

    # Some mirrors contain a small number of non-standard segmentation filenames.
    if suffix == "seg":
        fallback_candidates = (
            sorted(patient_dir.glob("*Seg*.nii"))
            + sorted(patient_dir.glob("*seg*.nii"))
            + sorted(patient_dir.glob("*Seg*.nii.gz"))
            + sorted(patient_dir.glob("*seg*.nii.gz"))
        )
        fallback_candidates = list(dict.fromkeys(fallback_candidates))
        if len(fallback_candidates) == 1:
            return fallback_candidates[0]
    return nii_path


def collect_training_records(training_root: str | Path) -> list[PatientRecord]:
    patient_dirs = list_patient_dirs(training_root)
    records: list[PatientRecord] = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        flair_path = resolve_volume_path(patient_dir, patient_id, "flair")
        seg_path = resolve_volume_path(patient_dir, patient_id, "seg")
        records.append(PatientRecord(patient_id=patient_id, flair_path=flair_path, seg_path=seg_path))
    return records


def validate_patient_files(patient_dir: Path, required_suffixes: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    patient_id = patient_dir.name
    for suffix in required_suffixes:
        expected = resolve_volume_path(patient_dir, patient_id, suffix)
        if not expected.exists():
            missing.append(expected.name)
    return missing


def load_volume(path: str | Path) -> np.ndarray:
    nib = require_nibabel()
    return nib.load(str(path)).get_fdata().astype(np.float32)


def choose_preview_slices(mask_volume: np.ndarray, preview_count: int) -> list[int]:
    positive_per_slice = mask_volume.reshape(-1, mask_volume.shape[-1]).sum(axis=0)
    ranked = np.argsort(positive_per_slice)[::-1]
    selected = [int(index) for index in ranked if positive_per_slice[index] > 0][:preview_count]
    if not selected:
        midpoint = mask_volume.shape[-1] // 2
        selected = [midpoint]
    return selected


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    non_zero = slice_2d[slice_2d > 0]
    if non_zero.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    lower = float(np.percentile(non_zero, 1))
    upper = float(np.percentile(non_zero, 99))
    if upper <= lower:
        lower = float(non_zero.min())
        upper = float(non_zero.max())
    clipped = np.clip(slice_2d, lower, upper)
    scaled = (clipped - lower) / max(upper - lower, 1e-6)
    return (scaled * 255.0).astype(np.uint8)


def resize_image_and_mask(image_slice: np.ndarray, mask_slice: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(image_slice).resize((image_size, image_size), Image.BILINEAR)
    mask = Image.fromarray(mask_slice.astype(np.uint8)).resize((image_size, image_size), Image.NEAREST)
    return np.array(image, dtype=np.uint8), np.array(mask, dtype=np.uint8)


def save_preview_figure(
    flair_volume: np.ndarray,
    binary_mask: np.ndarray,
    patient_id: str,
    preview_slices: list[int],
    output_path: str | Path,
    show: bool = False,
) -> tuple[Path, bool]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(preview_slices)
    figure, axes = plt.subplots(rows, 3, figsize=(10, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, slice_idx in enumerate(preview_slices):
        image_slice = normalize_slice(flair_volume[:, :, slice_idx])
        mask_slice = binary_mask[:, :, slice_idx].astype(np.uint8)

        axes[row_idx, 0].imshow(image_slice, cmap="gray")
        axes[row_idx, 0].set_title(f"{patient_id} slice {slice_idx}")
        axes[row_idx, 1].imshow(mask_slice, cmap="gray")
        axes[row_idx, 1].set_title("Binary mask")
        axes[row_idx, 2].imshow(image_slice, cmap="gray")
        axes[row_idx, 2].imshow(mask_slice, cmap="jet", alpha=0.35)
        axes[row_idx, 2].set_title("Overlay")

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    figure.tight_layout()
    try:
        figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    except PermissionError:
        plt.close(figure)
        try:
            save_preview_montage(flair_volume, binary_mask, preview_slices, output_path)
            if show:
                print(f"Preview figure fallback saved to: {output_path}")
            return output_path, output_path.exists()
        except PermissionError:
            if show:
                print(f"Preview image could not be written in this environment: {output_path}")
            return output_path, False
    if show:
        plt.show()
    plt.close(figure)
    return output_path, output_path.exists()


def save_preview_montage(
    flair_volume: np.ndarray,
    binary_mask: np.ndarray,
    preview_slices: list[int],
    output_path: str | Path,
) -> Path:
    rows: list[np.ndarray] = []
    for slice_idx in preview_slices:
        image_slice = normalize_slice(flair_volume[:, :, slice_idx])
        mask_slice = (binary_mask[:, :, slice_idx] > 0).astype(np.uint8) * 255

        gray_rgb = np.stack([image_slice] * 3, axis=-1)
        mask_rgb = np.stack([mask_slice] * 3, axis=-1)
        overlay_rgb = gray_rgb.copy()
        overlay_rgb[:, :, 1] = np.maximum(overlay_rgb[:, :, 1], mask_slice)

        row = np.concatenate([gray_rgb, mask_rgb, overlay_rgb], axis=1)
        rows.append(row)

    montage = np.concatenate(rows, axis=0)
    Image.fromarray(montage).save(output_path)
    return Path(output_path)


def check_dataset(config: dict[str, Any], show: bool = False) -> dict[str, Any]:
    paths = config["paths"]
    dataset_cfg = config["dataset"]

    training_root = Path(paths["training_root"])
    validation_root = Path(paths["validation_root"])
    artifacts_root = ensure_dir(Path(paths["artifacts_root"]) / "dataset_checks")

    training_patients = list_patient_dirs(training_root)
    validation_patients = list_patient_dirs(validation_root)

    if not training_patients:
        raise RuntimeError("No training patients found.")
    if not validation_patients:
        raise RuntimeError("No validation patients found.")
    if len(training_patients) != 369:
        raise RuntimeError(f"Expected 369 training patients, found {len(training_patients)}.")
    if len(validation_patients) != 125:
        raise RuntimeError(f"Expected 125 validation patients, found {len(validation_patients)}.")

    missing_training = {
        patient_dir.name: validate_patient_files(patient_dir, EXPECTED_TRAIN_SUFFIXES)
        for patient_dir in training_patients
    }
    missing_training = {name: missing for name, missing in missing_training.items() if missing}
    if missing_training:
        first_item = next(iter(missing_training.items()))
        raise RuntimeError(f"Missing training files detected, first issue: {first_item}")

    missing_validation = {
        patient_dir.name: validate_patient_files(patient_dir, EXPECTED_VAL_SUFFIXES)
        for patient_dir in validation_patients
    }
    missing_validation = {name: missing for name, missing in missing_validation.items() if missing}
    if missing_validation:
        first_item = next(iter(missing_validation.items()))
        raise RuntimeError(f"Missing validation files detected, first issue: {first_item}")

    preview_patient_id = dataset_cfg["preview_patient"]
    preview_patient_dir = training_root / preview_patient_id
    if not preview_patient_dir.exists():
        preview_patient_dir = training_patients[0]
        preview_patient_id = preview_patient_dir.name

    flair_path = resolve_volume_path(preview_patient_dir, preview_patient_id, "flair")
    seg_path = resolve_volume_path(preview_patient_dir, preview_patient_id, "seg")
    flair_volume = load_volume(flair_path)
    seg_volume = load_volume(seg_path)
    binary_mask = (seg_volume > 0).astype(np.uint8)
    preview_slices = choose_preview_slices(binary_mask, int(dataset_cfg["preview_slices"]))
    preview_path, preview_saved = save_preview_figure(
        flair_volume,
        binary_mask,
        preview_patient_id,
        preview_slices,
        artifacts_root / f"dataset_preview_{uuid4().hex[:8]}.png",
        show=show,
    )

    summary = {
        "training_patients": len(training_patients),
        "validation_patients": len(validation_patients),
        "preview_patient": preview_patient_id,
        "flair_shape": list(flair_volume.shape),
        "seg_shape": list(seg_volume.shape),
        "mask_labels": [int(label) for label in np.unique(seg_volume)],
        "preview_slices": preview_slices,
        "preview_path": str(preview_path),
        "preview_saved": preview_saved,
    }
    try:
        save_json(artifacts_root / "dataset_check_summary.json", summary)
    except PermissionError:
        pass
    return summary


def split_patients(patient_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[str]]:
    shuffled = list(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def save_processed_slice(image_array: np.ndarray, mask_array: np.ndarray, image_path: Path, mask_path: Path) -> None:
    Image.fromarray(image_array).save(image_path)
    Image.fromarray(mask_array).save(mask_path)


def clear_existing_pngs(directory: Path) -> None:
    for png_path in directory.glob("*.png"):
        png_path.unlink()


def prepare_2d_dataset(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    dataset_cfg = config["dataset"]

    training_root = Path(paths["training_root"])
    processed_root = ensure_dir(paths["processed_root"])
    splits_path = Path(paths["splits_file"])

    records = collect_training_records(training_root)
    patient_ids = [record.patient_id for record in records]
    split_mapping = split_patients(
        patient_ids=patient_ids,
        train_ratio=float(dataset_cfg["train_ratio"]),
        val_ratio=float(dataset_cfg["val_ratio"]),
        seed=int(dataset_cfg["random_seed"]),
    )
    patient_to_split = {
        patient_id: split_name
        for split_name, split_patients in split_mapping.items()
        for patient_id in split_patients
    }

    image_size = int(dataset_cfg["image_size"])
    split_counts: dict[str, int] = {split_name: 0 for split_name in split_mapping}
    slice_manifest: dict[str, list[str]] = {split_name: [] for split_name in split_mapping}

    for split_name in split_mapping:
        images_dir = ensure_dir(processed_root / split_name / "images")
        masks_dir = ensure_dir(processed_root / split_name / "masks")
        clear_existing_pngs(images_dir)
        clear_existing_pngs(masks_dir)

    iterator = tqdm(records, desc="Preparing 2D slices", unit="patient")
    for record in iterator:
        split_name = patient_to_split[record.patient_id]
        flair_volume = load_volume(record.flair_path)
        seg_volume = load_volume(record.seg_path)
        binary_mask = (seg_volume > 0).astype(np.uint8)

        if flair_volume.shape != seg_volume.shape:
            raise RuntimeError(
                f"Shape mismatch for {record.patient_id}: "
                f"flair={flair_volume.shape}, seg={seg_volume.shape}"
            )

        depth = flair_volume.shape[-1]
        for slice_idx in range(depth):
            mask_slice = binary_mask[:, :, slice_idx]
            if dataset_cfg["keep_strategy"] == "tumor_only" and int(mask_slice.sum()) == 0:
                continue

            image_slice = normalize_slice(flair_volume[:, :, slice_idx])
            mask_slice = (mask_slice > 0).astype(np.uint8) * 255
            image_array, mask_array = resize_image_and_mask(image_slice, mask_slice, image_size)

            sample_id = f"{record.patient_id}_slice_{slice_idx:03d}"
            image_path = processed_root / split_name / "images" / f"{sample_id}.png"
            mask_path = processed_root / split_name / "masks" / f"{sample_id}.png"
            save_processed_slice(image_array, mask_array, image_path, mask_path)

            split_counts[split_name] += 1
            slice_manifest[split_name].append(sample_id)

    splits_payload = {
        "seed": int(dataset_cfg["random_seed"]),
        "train_ratio": float(dataset_cfg["train_ratio"]),
        "val_ratio": float(dataset_cfg["val_ratio"]),
        "test_ratio": float(dataset_cfg["test_ratio"]),
        "patients": split_mapping,
        "slice_counts": split_counts,
        "samples": slice_manifest,
    }
    save_json(splits_path, splits_payload)
    save_json(processed_root / "preprocessing_summary.json", splits_payload)
    return splits_payload
