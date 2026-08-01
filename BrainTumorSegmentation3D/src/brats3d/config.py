from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "project_root": None,
        "training_root": None,
        "artifacts_root": None,
        "manifest_file": None,
        "legacy_splits_file": None,
    },
    "dataset": {
        "modality": "flair",
        "label_suffix": "seg",
        "patch_size": [64, 96, 96],  # Tensor order: depth, height, width.
        "patches_per_patient": 4,
        "positive_patch_ratio": 0.75,
        "cache_size": 1,
        "normalization_epsilon": 1.0e-6,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
        "reuse_legacy_splits": True,
        "preview_patient": "BraTS20_Training_001",
        "expected_training_patients": 369,
        "augmentation": {
            "enabled": True,
            "flip_probability": 0.5,
            "intensity_scale": 0.10,
            "intensity_shift": 0.10,
            "noise_std": 0.01,
        },
    },
    "training": {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
    },
    "inference": {
        "batch_size": 1,
        "roi_size": [64, 96, 96],
    },
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _default_project_root(config_path: Path) -> Path:
    if config_path.parent.name.lower() == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _resolve_path(value: str | Path, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _path_alias(paths: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if paths.get(canonical):
        return
    for alias in aliases:
        if paths.get(alias):
            paths[canonical] = paths[alias]
            return


def _validate_spatial_size(name: str, value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers in D,H,W order.")
    result = [int(item) for item in value]
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must all be positive, received {result}.")
    return result


def load_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the independent 3D project configuration.

    Relative entries in ``paths`` are resolved from the YAML file directory. The
    defaults deliberately point to the original NIfTI dataset and the old 2D
    patient split file; no image volume is copied into the 3D project.
    """

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration file must contain a top-level mapping.")

    config = _deep_merge(DEFAULT_CONFIG, loaded)
    paths = dict(config.get("paths", {}))
    _path_alias(paths, "training_root", ("data_root", "brats_root", "raw_training_root"))
    _path_alias(paths, "manifest_file", ("manifest_path", "splits_3d_file"))
    _path_alias(
        paths,
        "legacy_splits_file",
        ("old_splits_file", "old_2d_splits_file", "source_splits_file", "splits_file"),
    )

    configured_project_root = project_root or paths.get("project_root")
    if configured_project_root:
        configured_root_path = Path(configured_project_root).expanduser()
        if not configured_root_path.is_absolute():
            configured_root_path = config_path.parent / configured_root_path
        root = configured_root_path.resolve()
    else:
        root = _default_project_root(config_path)
    paths["project_root"] = str(root)

    root_parent = root.parent
    paths["training_root"] = paths.get("training_root") or str(
        root_parent / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    )
    paths["artifacts_root"] = paths.get("artifacts_root") or str(root / "artifacts")
    paths["manifest_file"] = paths.get("manifest_file") or str(root / "data" / "manifest_3d.json")
    paths["legacy_splits_file"] = paths.get("legacy_splits_file") or str(
        root_parent / "BrainTumorSegmentation" / "data_processed" / "splits.json"
    )

    # Explicit relative paths are interpreted relative to the configuration file.
    for key, value in list(paths.items()):
        if value is None:
            continue
        if key == "project_root":
            paths[key] = str(root)
        else:
            paths[key] = _resolve_path(value, config_path.parent)
    config["paths"] = paths

    dataset = config["dataset"]
    dataset["patch_size"] = _validate_spatial_size("dataset.patch_size", dataset["patch_size"])
    config["inference"]["roi_size"] = _validate_spatial_size(
        "inference.roi_size", config["inference"].get("roi_size", dataset["patch_size"])
    )
    ratio_keys = ("train_ratio", "val_ratio", "test_ratio")
    ratios = [float(dataset[key]) for key in ratio_keys]
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1.0e-6:
        raise ValueError(f"dataset split ratios must be non-negative and sum to 1, received {ratios}.")
    dataset["positive_patch_ratio"] = float(dataset["positive_patch_ratio"])
    if not 0.0 <= dataset["positive_patch_ratio"] <= 1.0:
        raise ValueError("dataset.positive_patch_ratio must be between 0 and 1.")
    if int(dataset["patches_per_patient"]) <= 0:
        raise ValueError("dataset.patches_per_patient must be positive.")
    if int(dataset.get("cache_size", 0)) < 0:
        raise ValueError("dataset.cache_size cannot be negative.")

    config["config_path"] = str(config_path)
    return config


def ensure_directories(config: Mapping[str, Any]) -> None:
    """Create only lightweight project output directories."""

    paths = config["paths"]
    Path(paths["project_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["artifacts_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["manifest_file"]).parent.mkdir(parents=True, exist_ok=True)


def get_manifest_path(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["manifest_file"])
