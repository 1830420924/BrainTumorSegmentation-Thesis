from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import ensure_directories, load_config  # noqa: E402
from brats3d.dataset import BraTS3DPatchDataset  # noqa: E402
from brats3d.manifest import (  # noqa: E402
    VALID_SPLITS,
    load_manifest,
    prepare_manifest,
    validate_manifest,
)
from brats3d.utils import (  # noqa: E402
    list_patient_dirs,
    load_nifti_array,
    load_volume_pair,
    require_nibabel,
    save_json,
)


def _deep_header_check(manifest: Mapping[str, Any]) -> dict[str, Any]:
    nib = require_nibabel()
    shapes: dict[str, int] = {}
    checked = 0
    for split in VALID_SPLITS:
        for record in manifest["splits"][split]:
            image = nib.load(str(record["image_path"]), mmap=True)
            mask = nib.load(str(record["mask_path"]), mmap=True)
            if len(image.shape) != 3 or image.shape != mask.shape:
                raise ValueError(
                    f"Header shape mismatch for {record['patient_id']}: "
                    f"FLAIR={image.shape}, seg={mask.shape}"
                )
            if not np.allclose(image.affine, mask.affine, atol=1.0e-4):
                raise ValueError(f"Header affine mismatch for {record['patient_id']}.")
            shape_key = "x".join(str(value) for value in image.shape)
            shapes[shape_key] = shapes.get(shape_key, 0) + 1
            checked += 1
    return {"patients_checked": checked, "shape_counts": shapes}


def check_dataset(config: Mapping[str, Any], deep: bool = False) -> dict[str, Any]:
    """Validate raw NIfTI references, split isolation, and one real 3D sample."""

    ensure_directories(config)
    training_root = Path(config["paths"]["training_root"])
    patient_dirs = list_patient_dirs(training_root)
    expected = int(config["dataset"].get("expected_training_patients", 0) or 0)
    if expected and len(patient_dirs) != expected:
        raise RuntimeError(
            f"Expected {expected} BraTS training patients, found {len(patient_dirs)} in {training_root}."
        )

    manifest_path = Path(config["paths"]["manifest_file"])
    manifest = load_manifest(manifest_path, check_files=True) if manifest_path.exists() else prepare_manifest(config)
    validate_manifest(manifest, check_files=True)

    all_ids = [
        str(record["patient_id"])
        for split in VALID_SPLITS
        for record in manifest["splits"][split]
    ]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Patient overlap detected between manifest splits.")
    if len(all_ids) != len(patient_dirs):
        raise RuntimeError(
            f"Manifest indexes {len(all_ids)} patients but raw root contains {len(patient_dirs)}."
        )

    preview_id = str(config["dataset"].get("preview_patient", ""))
    preview_record = next(
        (
            record
            for split in VALID_SPLITS
            for record in manifest["splits"][split]
            if record["patient_id"] == preview_id
        ),
        manifest["splits"]["train"][0],
    )
    pair = load_volume_pair(
        preview_record["image_path"],
        preview_record["mask_path"],
        normalization_epsilon=float(config["dataset"].get("normalization_epsilon", 1.0e-6)),
    )
    raw_segmentation, _ = load_nifti_array(preview_record["mask_path"])
    raw_labels = [int(value) for value in np.unique(raw_segmentation)]
    foreground = pair.image != 0
    normalized_values = pair.image[foreground]

    patch_dataset = BraTS3DPatchDataset(
        manifest,
        split="train",
        config=config,
        patches_per_patient=1,
        augment=False,
        cache_size=0,
    )
    patch = patch_dataset[0]
    expected_patch_shape = (1, *tuple(int(value) for value in config["dataset"]["patch_size"]))
    if tuple(patch["image"].shape) != expected_patch_shape:
        raise RuntimeError(
            f"Patch shape mismatch: expected {expected_patch_shape}, got {tuple(patch['image'].shape)}."
        )
    if not set(np.unique(patch["mask"].numpy())).issubset({0.0, 1.0}):
        raise RuntimeError("Patch mask is not binary.")

    summary: dict[str, Any] = {
        "status": "ok",
        "training_root": str(training_root.resolve()),
        "manifest_file": str(manifest_path.resolve()),
        "raw_patient_count": len(patient_dirs),
        "split_counts": {split: len(manifest["splits"][split]) for split in VALID_SPLITS},
        "split_source": manifest.get("split_source", {}),
        "raw_data_copied": bool(manifest.get("storage", {}).get("raw_data_copied", True)),
        "patient_overlap": False,
        "preview_patient": str(preview_record["patient_id"]),
        "nifti_shape_xyz": list(pair.original_shape),
        "tensor_shape_cdhw": [1, *list(pair.image.shape)],
        "raw_segmentation_labels": raw_labels,
        "binary_mask_labels": [int(value) for value in np.unique(pair.mask)],
        "tumor_voxels": int(pair.mask.sum()),
        "normalization": {
            **pair.normalization,
            "observed_normalized_mean": float(normalized_values.mean()) if normalized_values.size else 0.0,
            "observed_normalized_std": float(normalized_values.std()) if normalized_values.size else 0.0,
            "background_remains_zero": bool(np.all(pair.image[~foreground] == 0)),
        },
        "sample_patch_shape": list(patch["image"].shape),
        "sample_patch_sampling": patch["sampling"],
        "deep_header_check": _deep_header_check(manifest) if deep else {"performed": False},
    }
    output_path = Path(config["paths"]["artifacts_root"]) / "dataset_checks" / "dataset_check_summary.json"
    save_json(output_path, summary)
    summary["summary_file"] = str(output_path.resolve())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the independent BraTS 3D data pipeline.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default_config.yaml"),
        help="Path to the 3D YAML configuration.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Read every NIfTI header in addition to the default one-patient smoke check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = check_dataset(config, deep=args.deep)
    print("=" * 64)
    print("BraTS 2020 independent 3D dataset check: OK")
    print("=" * 64)
    print(f"Raw patients       : {summary['raw_patient_count']}")
    print(f"Split counts       : {summary['split_counts']}")
    print(f"Split source       : {summary['split_source'].get('kind')}")
    print(f"Preview volume XYZ : {tuple(summary['nifti_shape_xyz'])}")
    print(f"Tensor shape CDHW  : {tuple(summary['tensor_shape_cdhw'])}")
    print(f"Patch shape CDHW   : {tuple(summary['sample_patch_shape'])}")
    print(f"Raw seg labels     : {summary['raw_segmentation_labels']}")
    print(f"Summary            : {summary['summary_file']}")


if __name__ == "__main__":
    main()

