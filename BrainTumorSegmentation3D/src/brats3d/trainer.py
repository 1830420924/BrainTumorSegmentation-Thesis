from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

from .dataset import BraTS3DPatchDataset, BraTS3DVolumeDataset
from .inference import sliding_window_inference
from .losses import build_loss
from .metrics import volume_segmentation_metrics
from .models import build_model
from .utils import ensure_dir, save_json, seed_worker


def resolve_device(requested: str | None = "auto") -> torch.device:
    name = str(requested or "auto").strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"training.device={requested!r}, but CUDA is not available. "
            "Use device: auto/cpu for a CPU smoke test or install a CUDA-enabled PyTorch build."
        )
    return torch.device(name)


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return payload


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
            writer.writeheader()
            writer.writerows(materialized)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value in (None, ""):
                converted[key] = None
                continue
            try:
                number = float(value)
                converted[key] = int(number) if key == "epoch" and number.is_integer() else number
            except (TypeError, ValueError):
                converted[key] = value
        parsed.append(converted)
    return parsed


def _save_training_curves(history: list[dict[str, Any]], output_path: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific.
        raise RuntimeError("matplotlib is required to save training curves.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    def values(key: str) -> list[float]:
        result: list[float] = []
        for row in history:
            value = row.get(key)
            result.append(float("nan") if value in (None, "") else float(value))
        return result

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, values("train_loss"), label="train")
    axes[0].plot(epochs, values("val_loss"), marker="o", label="full-volume val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, values("train_patch_dice"), label="train patch Dice")
    axes[1].plot(epochs, values("val_dice"), marker="o", label="patient 3D val Dice")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _patient_id(batch: Mapping[str, Any]) -> str:
    value = batch["patient_id"]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Full-volume validation requires batch_size=1.")
        return str(value[0])
    return str(value)


def _mean_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("Full-volume validation produced no patients.")
    summary: dict[str, float] = {}
    for key in ("loss", "dice", "iou", "precision", "recall"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        finite = [value for value in values if math.isfinite(value)]
        summary[key] = float(sum(finite) / len(finite)) if finite else float("nan")
    return summary


def build_dataloaders(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    training_cfg = config["training"]
    manifest_path = config["paths"]["manifest_file"]
    train_dataset = BraTS3DPatchDataset(manifest_path, split="train", config=config)
    val_dataset = BraTS3DVolumeDataset(manifest_path, split="val", config=config)

    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", True) and device.type == "cuda")
    generator = torch.Generator()
    generator.manual_seed(int(config["dataset"].get("random_seed", 42)))
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    # A complete BraTS case has a different spatial shape from a random patch;
    # keeping this at one also makes the per-patient metric definition explicit.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    threshold: float,
    accumulation_steps: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = bool(scaler.is_enabled())
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    optimizer_steps = 0
    log_every = max(1, len(loader) // 10)

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
            scaled_loss = loss / accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}: {loss.item()}")
        scaler.scale(scaled_loss).backward()

        should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        patch_metrics = volume_segmentation_metrics(
            logits.detach().float().cpu(),
            masks.detach().float().cpu(),
            threshold=threshold,
            from_logits=True,
        )
        total_loss += float(loss.detach().item())
        total_dice += float(patch_metrics["dice"])
        batches += 1
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(loader):
            print(
                f"  train batch {batch_index:04d}/{len(loader):04d} "
                f"loss={float(loss.detach().item()):.4f} "
                f"patch_dice={float(patch_metrics['dice']):.4f}"
            )

    if batches == 0:
        raise RuntimeError("Training DataLoader produced zero batches.")
    return {
        "loss": total_loss / batches,
        "patch_dice": total_dice / batches,
        "optimizer_steps": float(optimizer_steps),
    }


@torch.no_grad()
def validate_full_volumes(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    inference_cfg = config["inference"]
    threshold = float(config["training"].get("threshold", 0.5))
    use_amp = bool(config["training"].get("mixed_precision", True) and device.type == "cuda")
    patient_rows: list[dict[str, Any]] = []

    for index, batch in enumerate(loader, start=1):
        images = batch["image"]
        masks = batch["mask"].float().cpu()
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(
                f"Validation expects [1,C,D,H,W], received {tuple(images.shape)}."
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
        row: dict[str, Any] = {
            "patient_id": _patient_id(batch),
            "loss": float(loss.item()),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        patient_rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(loader):
            print(
                f"  full-volume val {index:03d}/{len(loader):03d} "
                f"patient={row['patient_id']} dice={row['dice']:.4f}"
            )

    return _mean_metric_rows(patient_rows), patient_rows


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    experiment: dict[str, Any],
    epoch: int,
    best_val_dice: float,
) -> dict[str, Any]:
    training_cfg = config["training"]
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "experiment": dict(experiment),
        "model_name": experiment["model"],
        "loss_name": experiment["loss"],
        "in_channels": int(training_cfg.get("in_channels", 1)),
        "out_channels": int(training_cfg.get("out_channels", 1)),
        "base_channels": int(training_cfg.get("base_channels", 16)),
        "best_val_dice": float(best_val_dice),
        "config": config,
        "config_path": config.get("config_path"),
    }


def train_experiment(
    config: dict[str, Any],
    experiment: dict[str, Any],
    epochs_override: int | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    training_cfg = config["training"]
    experiment_name = str(experiment["name"])
    experiment_root = ensure_dir(
        Path(config["paths"]["artifacts_root"]) / "experiments" / experiment_name
    )
    history_path = experiment_root / "training_history.csv"
    device = resolve_device(training_cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    epochs = int(epochs_override or training_cfg.get("epochs", 50))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    accumulation_steps = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    validate_every = max(1, int(training_cfg.get("validate_every", 1)))
    use_amp = bool(training_cfg.get("mixed_precision", True) and device.type == "cuda")

    train_loader, val_loader = build_dataloaders(config, device)
    model = build_model(
        str(experiment["model"]),
        in_channels=int(training_cfg.get("in_channels", 1)),
        out_channels=int(training_cfg.get("out_channels", 1)),
        base_channels=int(training_cfg.get("base_channels", 16)),
    ).to(device)
    criterion = build_loss(
        str(experiment["loss"]),
        bce_weight=float(experiment.get("bce_weight", 1.0)),
        dice_weight=float(experiment.get("dice_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("scheduler_factor", 0.5)),
        patience=int(training_cfg.get("scheduler_patience", 5)),
        min_lr=float(training_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_checkpoint_path = experiment_root / "best_model.pth"
    last_checkpoint_path = experiment_root / "last_model.pth"
    best_val_dice = -1.0
    start_epoch = 1
    history: list[dict[str, Any]] = []

    if resume_path is not None:
        checkpoint = _torch_load(resume_path, map_location="cpu")
        saved_experiment = checkpoint.get("experiment", {})
        if saved_experiment and saved_experiment.get("name") != experiment_name:
            raise ValueError(
                f"Checkpoint experiment {saved_experiment.get('name')!r} does not match "
                f"requested experiment {experiment_name!r}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        history = [
            row for row in _read_csv(history_path) if int(row.get("epoch", 0)) < start_epoch
        ]
        print(f"Resuming {experiment_name} from epoch {start_epoch}: {resume_path}")

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch - 1}, but requested epochs={epochs}."
        )

    save_json(experiment_root / "config_snapshot.json", config)
    print(
        f"[{experiment_name}] device={device} amp={use_amp} batch={training_cfg.get('batch_size', 1)} "
        f"accumulation={accumulation_steps} effective_batch="
        f"{int(training_cfg.get('batch_size', 1)) * accumulation_steps}"
    )

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_metrics = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            threshold=float(training_cfg.get("threshold", 0.5)),
            accumulation_steps=accumulation_steps,
            gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
        )
        should_validate = epoch % validate_every == 0 or epoch == epochs
        val_metrics: dict[str, float] | None = None
        patient_rows: list[dict[str, Any]] = []
        if should_validate:
            val_metrics, patient_rows = validate_full_volumes(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                config=config,
            )
            scheduler.step(val_metrics["dice"])
            _write_csv(experiment_root / "validation_patient_metrics.csv", patient_rows)

        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_patch_dice": train_metrics["patch_dice"],
            "optimizer_steps": int(train_metrics["optimizer_steps"]),
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_dice": None if val_metrics is None else val_metrics["dice"],
            "val_iou": None if val_metrics is None else val_metrics["iou"],
            "val_precision": None if val_metrics is None else val_metrics["precision"],
            "val_recall": None if val_metrics is None else val_metrics["recall"],
        }
        history.append(row)

        if val_metrics is not None and val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    experiment=experiment,
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                ),
                best_checkpoint_path,
            )

        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                experiment=experiment,
                epoch=epoch,
                best_val_dice=best_val_dice,
            ),
            last_checkpoint_path,
        )
        _write_csv(history_path, history)
        _save_training_curves(history, experiment_root / "training_curves.png")

        val_text = "not scheduled"
        if val_metrics is not None:
            val_text = f"{val_metrics['dice']:.4f}"
        print(
            f"[{experiment_name}] epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_patch_dice={train_metrics['patch_dice']:.4f} "
            f"val_patient_dice={val_text}"
        )

    metadata = {
        "experiment": dict(experiment),
        "device": str(device),
        "epochs": epochs,
        "batch_size": int(training_cfg.get("batch_size", 1)),
        "gradient_accumulation_steps": accumulation_steps,
        "mixed_precision": use_amp,
        "validation_mode": "complete 3D patient volumes via sliding-window inference",
        "validate_every": validate_every,
        "best_val_dice": float(best_val_dice),
        "best_model_path": str(best_checkpoint_path),
        "last_model_path": str(last_checkpoint_path),
        "history_path": str(history_path),
        "curves_path": str(experiment_root / "training_curves.png"),
    }
    save_json(experiment_root / "metadata.json", metadata)
    return metadata


__all__ = [
    "build_dataloaders",
    "resolve_device",
    "train_experiment",
    "validate_full_volumes",
]
