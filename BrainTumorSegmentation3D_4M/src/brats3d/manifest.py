from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import list_patient_dirs, load_json, resolve_volume_path, save_json


VALID_SPLITS = ("train", "val", "test")
FOUR_MODALITIES = ("t1", "t1ce", "t2", "flair")


def collect_patient_records(
    training_root: str | Path,
    modalities: Sequence[str] = FOUR_MODALITIES,
    label_suffix: str = "seg",
) -> dict[str, dict[str, Any]]:
    """Index raw NIfTI paths without copying any volume data."""

    modalities = tuple(str(modality).strip().lower() for modality in modalities)
    if modalities != FOUR_MODALITIES:
        raise ValueError(
            "The four-modality manifest requires t1, t1ce, t2, flair in that order."
        )
    records: dict[str, dict[str, Any]] = {}
    for patient_dir in list_patient_dirs(training_root):
        patient_id = patient_dir.name
        mask_path = resolve_volume_path(patient_dir, patient_id, label_suffix)
        records[patient_id] = {
            "patient_id": patient_id,
            "image_paths": {
                modality: str(resolve_volume_path(patient_dir, patient_id, modality))
                for modality in modalities
            },
            "mask_path": str(mask_path),
        }
    if not records:
        raise RuntimeError(f"No BraTS patient directories found under {training_root}.")
    return records


def _patient_id(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and item.get("patient_id"):
        return str(item["patient_id"])
    raise ValueError(f"Invalid patient split entry: {item!r}")


def _extract_patient_splits(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    section = payload.get("patients")
    if not isinstance(section, Mapping):
        section = payload.get("splits")
    if not isinstance(section, Mapping):
        raise ValueError("Split JSON must contain a 'patients' or 'splits' mapping.")

    result: dict[str, list[str]] = {}
    for split in VALID_SPLITS:
        values = section.get(split)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Split JSON is missing a list for '{split}'.")
        result[split] = [_patient_id(item) for item in values]
    return result


def _validate_patient_splits(
    splits: Mapping[str, Sequence[str]],
    available_patient_ids: set[str],
) -> None:
    seen: dict[str, str] = {}
    for split in VALID_SPLITS:
        patients = list(splits[split])
        if len(patients) != len(set(patients)):
            raise ValueError(f"Duplicate patient ID inside '{split}' split.")
        for patient_id in patients:
            previous = seen.get(patient_id)
            if previous is not None:
                raise ValueError(
                    f"Patient leakage: {patient_id} occurs in both '{previous}' and '{split}'."
                )
            seen[patient_id] = split

    assigned = set(seen)
    missing_from_dataset = assigned - available_patient_ids
    unassigned = available_patient_ids - assigned
    if missing_from_dataset:
        preview = ", ".join(sorted(missing_from_dataset)[:5])
        raise ValueError(f"Split references patients absent from the raw dataset: {preview}")
    if unassigned:
        preview = ", ".join(sorted(unassigned)[:5])
        raise ValueError(f"Raw dataset contains patients missing from the split: {preview}")


def create_patient_splits(
    patient_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    shuffled = sorted(str(patient_id) for patient_id in patient_ids)
    generator = np.random.default_rng(int(seed))
    generator.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * float(train_ratio))
    val_end = train_end + int(total * float(val_ratio))
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _resolve_splits(
    config: Mapping[str, Any],
    available_patient_ids: set[str],
    reuse_legacy: bool | None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    dataset = config["dataset"]
    paths = config["paths"]
    should_reuse = bool(dataset.get("reuse_legacy_splits", True)) if reuse_legacy is None else reuse_legacy
    legacy_path = Path(paths["legacy_splits_file"])

    if should_reuse and legacy_path.exists():
        splits = _extract_patient_splits(load_json(legacy_path))
        _validate_patient_splits(splits, available_patient_ids)
        return splits, {
            "kind": "legacy_2d_patient_split",
            "path": str(legacy_path.resolve()),
            "reused": True,
        }

    splits = create_patient_splits(
        sorted(available_patient_ids),
        train_ratio=float(dataset["train_ratio"]),
        val_ratio=float(dataset["val_ratio"]),
        seed=int(dataset["random_seed"]),
    )
    _validate_patient_splits(splits, available_patient_ids)
    return splits, {
        "kind": "deterministic_3d_patient_split",
        "path": None,
        "reused": False,
        "legacy_requested_but_missing": bool(should_reuse and not legacy_path.exists()),
    }


def validate_manifest(payload: Mapping[str, Any], check_files: bool = False) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("4M manifest must contain a 'dataset' mapping.")
    raw_modalities = dataset.get("modalities")
    if not isinstance(raw_modalities, Sequence) or isinstance(raw_modalities, (str, bytes)):
        raise ValueError("4M manifest dataset must contain an ordered 'modalities' list.")
    modalities = tuple(str(value).strip().lower() for value in raw_modalities)
    if modalities != FOUR_MODALITIES:
        raise ValueError(
            "4M manifest modalities must be exactly ['t1', 't1ce', 't2', 'flair']."
        )

    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("3D manifest must contain a 'splits' mapping.")

    seen: dict[str, str] = {}
    for split in VALID_SPLITS:
        records = splits.get(split)
        if not isinstance(records, list):
            raise ValueError(f"3D manifest is missing split '{split}'.")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid record in '{split}': {record!r}")
            missing_keys = {"patient_id", "image_paths", "mask_path"} - set(record)
            if missing_keys:
                raise ValueError(f"Record in '{split}' is missing keys: {sorted(missing_keys)}")
            image_paths = record["image_paths"]
            if not isinstance(image_paths, Mapping):
                raise ValueError(f"Record in '{split}' has a non-mapping image_paths value.")
            actual_modalities = {str(key) for key in image_paths}
            expected_modalities = set(modalities)
            if actual_modalities != expected_modalities:
                missing = sorted(expected_modalities - actual_modalities)
                extra = sorted(actual_modalities - expected_modalities)
                raise ValueError(
                    f"Record in '{split}' has invalid image_paths modalities; "
                    f"missing={missing}, extra={extra}."
                )
            patient_id = str(record["patient_id"])
            if patient_id in seen:
                raise ValueError(
                    f"Patient leakage: {patient_id} occurs in '{seen[patient_id]}' and '{split}'."
                )
            seen[patient_id] = split
            if check_files:
                for modality in modalities:
                    path = image_paths[modality]
                    if not isinstance(path, (str, Path)) or not Path(path).exists():
                        raise FileNotFoundError(
                            f"Manifest {modality} path does not exist: {path}"
                        )
                mask_path = record["mask_path"]
                if not isinstance(mask_path, (str, Path)) or not Path(mask_path).exists():
                    raise FileNotFoundError(f"Manifest mask path does not exist: {mask_path}")

    counts = payload.get("counts")
    if isinstance(counts, Mapping):
        for split in VALID_SPLITS:
            if int(counts.get(split, -1)) != len(splits[split]):
                raise ValueError(f"Manifest count for '{split}' does not match its record list.")


def prepare_manifest(
    config: Mapping[str, Any],
    reuse_legacy: bool | None = None,
) -> dict[str, Any]:
    """Generate a lightweight 3D manifest that points to original NIfTI files."""

    paths = config["paths"]
    dataset = config["dataset"]
    records_by_id = collect_patient_records(
        paths["training_root"],
        modalities=dataset["modalities"],
        label_suffix=str(dataset.get("label_suffix", "seg")),
    )
    patient_splits, split_source = _resolve_splits(
        config,
        available_patient_ids=set(records_by_id),
        reuse_legacy=reuse_legacy,
    )

    split_records = {
        split: [records_by_id[patient_id] for patient_id in patient_splits[split]]
        for split in VALID_SPLITS
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "four_modality_whole_tumor_binary_segmentation_3d",
        "dataset": {
            "name": "BraTS2020",
            "modalities": list(dataset["modalities"]),
            "label_rule": "seg > 0",
            "nifti_order": "X,Y,Z",
            "tensor_order": "C,D,H,W",
        },
        "storage": {
            "raw_data_copied": False,
            "training_root": str(Path(paths["training_root"]).resolve()),
        },
        "split_source": split_source,
        "counts": {split: len(split_records[split]) for split in VALID_SPLITS},
        "splits": split_records,
    }
    validate_manifest(payload, check_files=True)
    save_json(paths["manifest_file"], payload)
    return payload


def load_manifest(path: str | Path, check_files: bool = False) -> dict[str, Any]:
    payload = load_json(path)
    validate_manifest(payload, check_files=check_files)
    return payload


def get_split_records(
    manifest_or_path: Mapping[str, Any] | str | Path,
    split: str,
) -> list[dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {VALID_SPLITS}.")
    payload = (
        dict(manifest_or_path)
        if isinstance(manifest_or_path, Mapping)
        else load_manifest(manifest_or_path)
    )
    validate_manifest(payload)
    return [dict(record) for record in payload["splits"][split]]


# A descriptive alias retained for scripts and notebooks.
prepare_3d_splits = prepare_manifest
