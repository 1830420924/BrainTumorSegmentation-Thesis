from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BraTS3DVolumeDataset
from .inference import sliding_window_inference
from .losses import build_loss
from .metrics import volume_segmentation_metrics
from .models import build_model
from .trainer import _torch_load, _write_csv, resolve_device
from .utils import dhw_to_xyz, ensure_dir, require_nibabel, save_json, seed_worker


def _single_patient_id(batch: Mapping[str, Any]) -> str:
    value = batch["patient_id"]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("3D evaluation requires one complete patient per batch.")
        return str(value[0])
    return str(value)


def _single_array(value: Any, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        raise ValueError(f"Expected array shape {expected_shape}, received {array.shape}.")
    return array


def _save_nifti_outputs(
    *,
    prediction_dhw: np.ndarray,
    probability_dhw: np.ndarray | None,
    affine: np.ndarray,
    original_shape_xyz: tuple[int, int, int],
    patient_id: str,
    output_dir: Path,
) -> dict[str, str]:
    nib = require_nibabel()
    prediction_xyz = dhw_to_xyz(prediction_dhw)
    if tuple(prediction_xyz.shape) != tuple(original_shape_xyz):
        raise ValueError(
            f"Cannot save {patient_id}: restored prediction shape {prediction_xyz.shape} "
            f"does not match original NIfTI shape {original_shape_xyz}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{patient_id}_prediction.nii.gz"
    prediction_image = nib.Nifti1Image(prediction_xyz.astype(np.uint8), affine)
    prediction_image.set_data_dtype(np.uint8)
    nib.save(prediction_image, str(prediction_path))
    paths = {"prediction_path": str(prediction_path)}

    if probability_dhw is not None:
        probability_xyz = dhw_to_xyz(probability_dhw)
        probability_path = output_dir / f"{patient_id}_probability.nii.gz"
        probability_image = nib.Nifti1Image(probability_xyz.astype(np.float32), affine)
        probability_image.set_data_dtype(np.float32)
        nib.save(probability_image, str(probability_path))
        paths["probability_path"] = str(probability_path)
    return paths


def _summarize_patient_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Evaluation DataLoader produced zero patients.")
    summary: dict[str, Any] = {"patients": len(rows)}
    for key in ("loss", "dice", "iou", "precision", "recall"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else None
        summary[f"{key}_std"] = float(finite.std(ddof=0)) if finite.size else None
    return summary


@torch.no_grad()
def evaluate_experiment(
    config: dict[str, Any],
    experiment_name: str,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
    save_nifti: bool | None = None,
) -> dict[str, Any]:
    training_cfg = config["training"]
    inference_cfg = config["inference"]
    evaluation_cfg = config.get("evaluation", {})
    split = str(split or evaluation_cfg.get("split", "test"))
    experiment_root = Path(config["paths"]["artifacts_root"]) / "experiments" / experiment_name
    checkpoint_path = Path(checkpoint_path or experiment_root / "best_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. Train {experiment_name!r} first."
        )

    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    experiment = dict(checkpoint.get("experiment", {}))
    model_name = str(checkpoint.get("model_name") or experiment.get("model"))
    loss_name = str(checkpoint.get("loss_name") or experiment.get("loss"))
    if model_name in ("", "None") or loss_name in ("", "None"):
        raise ValueError(f"Checkpoint lacks model/loss metadata: {checkpoint_path}")

    expected_modalities = tuple(str(value) for value in config["dataset"]["modalities"])
    checkpoint_modalities = tuple(str(value) for value in checkpoint.get("modalities", ()))
    if checkpoint_modalities != expected_modalities:
        raise ValueError(
            "Checkpoint modality order does not match this four-modality configuration: "
            f"checkpoint={checkpoint_modalities}, config={expected_modalities}."
        )
    if int(checkpoint.get("in_channels", -1)) != len(expected_modalities):
        raise ValueError(
            "Checkpoint input-channel count does not match the configured four modalities."
        )

    device = resolve_device(training_cfg.get("device", "auto"))
    model = build_model(
        model_name,
        in_channels=int(checkpoint.get("in_channels", training_cfg.get("in_channels", 1))),
        out_channels=int(checkpoint.get("out_channels", training_cfg.get("out_channels", 1))),
        base_channels=int(checkpoint.get("base_channels", training_cfg.get("base_channels", 16))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    criterion = build_loss(
        loss_name,
        bce_weight=float(experiment.get("bce_weight", 1.0)),
        dice_weight=float(experiment.get("dice_weight", 1.0)),
    )

    dataset = BraTS3DVolumeDataset(
        config["paths"]["manifest_file"],
        split=split,
        config=config,
    )
    workers = int(training_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(training_cfg.get("pin_memory", True) and device.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )

    save_nifti = bool(evaluation_cfg.get("save_nifti", False)) if save_nifti is None else save_nifti
    save_probabilities = bool(evaluation_cfg.get("save_probabilities", False))
    prediction_dir = experiment_root / f"{split}_nifti"
    threshold = float(training_cfg.get("threshold", 0.5))
    use_amp = bool(training_cfg.get("mixed_precision", True) and device.type == "cuda")
    patient_rows: list[dict[str, Any]] = []

    for index, batch in enumerate(loader, start=1):
        images = batch["image"]
        masks = batch["mask"].float().cpu()
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(
                f"Complete-volume evaluation expects [1,C,D,H,W], got {tuple(images.shape)}."
            )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = sliding_window_inference(
                model=model,
                volume=images,
                patch_size=tuple(int(value) for value in inference_cfg["roi_size"]),
                overlap=float(inference_cfg.get("overlap", 0.5)),
                sw_batch_size=int(inference_cfg.get("sw_batch_size", 1)),
                device=device,
                padding_mode=str(inference_cfg.get("padding_mode", "constant")),
                padding_value=float(inference_cfg.get("padding_value", 0.0)),
            )
        loss = criterion(logits, masks)
        metrics = volume_segmentation_metrics(
            logits,
            masks,
            threshold=threshold,
            from_logits=True,
        )
        patient_id = _single_patient_id(batch)
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "loss": float(loss.item()),
            **{key: float(value) for key, value in metrics.items()},
        }

        if save_nifti:
            probability_dhw = torch.sigmoid(logits)[0, 0].numpy().astype(np.float32)
            prediction_dhw = (probability_dhw >= threshold).astype(np.uint8)
            affine = _single_array(batch["affine"], expected_shape=(4, 4)).astype(np.float32)
            original_shape = tuple(
                int(value) for value in _single_array(batch["original_shape"]).reshape(-1)
            )
            if len(original_shape) != 3:
                raise ValueError(f"Invalid original_shape for {patient_id}: {original_shape}")
            row.update(
                _save_nifti_outputs(
                    prediction_dhw=prediction_dhw,
                    probability_dhw=probability_dhw if save_probabilities else None,
                    affine=affine,
                    original_shape_xyz=original_shape,
                    patient_id=patient_id,
                    output_dir=prediction_dir,
                )
            )

        patient_rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(loader):
            print(
                f"[{experiment_name}] {split} {index:03d}/{len(loader):03d} "
                f"patient={patient_id} dice={row['dice']:.4f}"
            )

    result_dir = ensure_dir(experiment_root / "evaluation")
    patient_csv = _write_csv(result_dir / f"{split}_patient_metrics.csv", patient_rows)
    summary = {
        "experiment_name": experiment_name,
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "evaluation_mode": "complete 3D patient volumes via sliding-window inference",
        "modalities": list(expected_modalities),
        "threshold": threshold,
        "roi_size_dhw": [int(value) for value in inference_cfg["roi_size"]],
        "overlap": float(inference_cfg.get("overlap", 0.5)),
        "patient_metrics_path": str(patient_csv),
        "nifti_predictions_saved": save_nifti,
        **_summarize_patient_rows(patient_rows),
    }
    if save_nifti:
        summary["prediction_dir"] = str(prediction_dir)
    save_json(result_dir / f"{split}_metrics.json", summary)
    return summary


def evaluate_all(
    config: dict[str, Any],
    experiment_names: list[str],
    split: str | None = None,
    save_nifti: bool | None = None,
) -> list[dict[str, Any]]:
    summaries = [
        evaluate_experiment(
            config=config,
            experiment_name=name,
            split=split,
            save_nifti=save_nifti,
        )
        for name in experiment_names
    ]
    output = Path(config["paths"]["artifacts_root"]) / "evaluation_summary.csv"
    _write_csv(output, summaries)
    save_json(
        Path(config["paths"]["artifacts_root"]) / "evaluation_summary.json",
        {"experiments": summaries},
    )
    return summaries


__all__ = ["evaluate_all", "evaluate_experiment"]
