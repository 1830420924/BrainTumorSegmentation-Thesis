from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from brats2d.dataset import BrainTumorSliceDataset
from brats2d.losses import build_loss
from brats2d.metrics import batch_segmentation_metrics
from brats2d.utils import ensure_dir, load_json, resolve_device, save_json, write_csv
from brats2d.visualization import save_prediction_panel

from .trainer import build_runtime_model, load_torch_checkpoint, unwrap_model


def evaluate_experiment_remote(config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    paths = config["paths"]
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]

    experiment_root = Path(paths["artifacts_root"]) / "experiments" / experiment_name
    metadata_path = experiment_root / "metadata.json"
    checkpoint_path = experiment_root / "best_model.pth"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for experiment: {metadata_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint for experiment: {checkpoint_path}")

    metadata = load_json(metadata_path)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    device = resolve_device(training_cfg["device"])

    model, runtime_device, gpu_ids, data_parallel_enabled = build_runtime_model(
        model_name=checkpoint["model_name"],
        base_channels=int(checkpoint["base_channels"]),
        device=device,
        training_cfg=training_cfg,
    )
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BrainTumorSliceDataset(paths["processed_root"], split="test")
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg["pin_memory"] and device.type == "cuda"),
    )

    criterion = build_loss(metadata["experiment"]["loss"])
    threshold = float(training_cfg["threshold"])
    total_loss = 0.0
    metrics_accumulator = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    num_batches = 0
    saved_previews = 0
    preview_limit = int(evaluation_cfg["preview_samples"])
    preview_dir = ensure_dir(experiment_root / "test_visuals")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(runtime_device)
            masks = batch["mask"].to(runtime_device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item())

            batch_metrics = batch_segmentation_metrics(logits, masks, threshold=threshold)
            for key, value in batch_metrics.items():
                metrics_accumulator[key] += value
            num_batches += 1

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float().cpu().numpy()
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            sample_ids = batch["sample_id"]

            for sample_index, sample_id in enumerate(sample_ids):
                if saved_previews >= preview_limit:
                    break
                image = (images_np[sample_index, 0] * 255.0).astype(np.uint8)
                mask = (masks_np[sample_index, 0] > 0.5).astype(np.uint8) * 255
                prediction = (predictions[sample_index, 0] > 0.5).astype(np.uint8) * 255
                save_prediction_panel(
                    image=image,
                    mask=mask,
                    prediction=prediction,
                    sample_id=sample_id,
                    output_path=preview_dir / f"{sample_id}.png",
                )
                saved_previews += 1

    if num_batches == 0:
        raise RuntimeError("Test loader produced zero batches.")

    summary = {
        "experiment_name": experiment_name,
        "test_loss": total_loss / num_batches,
        "dice": metrics_accumulator["dice"] / num_batches,
        "iou": metrics_accumulator["iou"] / num_batches,
        "precision": metrics_accumulator["precision"] / num_batches,
        "recall": metrics_accumulator["recall"] / num_batches,
        "preview_dir": str(preview_dir),
        "gpu_ids": gpu_ids,
        "gpu_count": len(gpu_ids),
        "data_parallel_enabled": data_parallel_enabled,
    }
    save_json(experiment_root / "test_metrics.json", summary)
    return summary


def evaluate_all_remote(config: dict[str, Any], experiment_names: list[str]) -> list[dict[str, Any]]:
    summaries = [evaluate_experiment_remote(config, experiment_name) for experiment_name in experiment_names]
    write_csv(Path(config["paths"]["artifacts_root"]) / "evaluation_summary.csv", summaries)
    return summaries
