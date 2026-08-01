"""Memory-conscious sliding-window inference for complete 3-D volumes."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _integer_triplet(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value, value)
    else:
        try:
            result = tuple(int(item) for item in value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer or a length-three sequence") from error
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers, got {result}")
    return result


def _overlap_triplet(value: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = (float(value),) * 3
    else:
        try:
            result = tuple(float(item) for item in value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError("overlap must be a number or a length-three sequence") from error
    if len(result) != 3 or any(item < 0.0 or item >= 1.0 for item in result):
        raise ValueError(f"overlap values must be in [0, 1), got {result}")
    return result


def _scan_starts(image_size: int, patch_size: int, overlap: float) -> list[int]:
    if image_size <= patch_size:
        return [0]
    stride = max(int(patch_size * (1.0 - overlap)), 1)
    starts = list(range(0, image_size - patch_size + 1, stride))
    final_start = image_size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: int | Sequence[int],
    overlap: float | Sequence[float] = 0.25,
    sw_batch_size: int = 1,
    device: str | torch.device | None = None,
    padding_mode: str = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Infer one patient volume using overlapping patches and CPU accumulation.

    Args:
        model: A model returning logits shaped ``[B, C_out, D, H, W]``.
        volume: ``[C, D, H, W]`` or singleton-batch ``[1, C, D, H, W]``.
        patch_size: Spatial patch size in ``(D, H, W)`` order.
        overlap: Fractional overlap for each spatial axis, in ``[0, 1)``.
        sw_batch_size: Number of patches evaluated in one model call.
        device: Inference device; defaults to the model parameter device.
        padding_mode: Mode accepted by :func:`torch.nn.functional.pad`.
        padding_value: Constant padding value when ``padding_mode='constant'``.

    Returns:
        CPU ``float32`` logits cropped to the original spatial shape. The input
        rank is preserved (4-D input returns 4-D output; 5-D returns 5-D).
    """

    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"volume must be a torch.Tensor, got {type(volume).__name__}")
    if volume.ndim not in {4, 5}:
        raise ValueError(
            f"volume must have shape [C,D,H,W] or [1,C,D,H,W], got {tuple(volume.shape)}"
        )
    input_had_batch = volume.ndim == 5
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.shape[0] != 1:
        raise ValueError("Sliding-window inference accepts exactly one patient volume at a time")
    if not isinstance(sw_batch_size, int) or isinstance(sw_batch_size, bool) or sw_batch_size <= 0:
        raise ValueError(f"sw_batch_size must be a positive integer, got {sw_batch_size!r}")

    patch_depth, patch_height, patch_width = _integer_triplet(patch_size, "patch_size")
    overlap_depth, overlap_height, overlap_width = _overlap_triplet(overlap)
    original_depth, original_height, original_width = volume.shape[2:]
    if min(original_depth, original_height, original_width) <= 0:
        raise ValueError("volume spatial dimensions must be positive")

    pad_depth = max(patch_depth - original_depth, 0)
    pad_height = max(patch_height - original_height, 0)
    pad_width = max(patch_width - original_width, 0)
    cpu_volume = volume.detach().to(device="cpu", dtype=torch.float32)
    padding = (0, pad_width, 0, pad_height, 0, pad_depth)
    if any(padding):
        if padding_mode == "constant":
            cpu_volume = F.pad(cpu_volume, padding, mode=padding_mode, value=padding_value)
        else:
            cpu_volume = F.pad(cpu_volume, padding, mode=padding_mode)

    padded_depth, padded_height, padded_width = cpu_volume.shape[2:]
    depth_starts = _scan_starts(padded_depth, patch_depth, overlap_depth)
    height_starts = _scan_starts(padded_height, patch_height, overlap_height)
    width_starts = _scan_starts(padded_width, patch_width, overlap_width)
    coordinates = list(product(depth_starts, height_starts, width_starts))

    inference_device = torch.device(device) if device is not None else _model_device(model)
    output_sum: torch.Tensor | None = None
    output_count: torch.Tensor | None = None
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(coordinates), sw_batch_size):
                batch_coordinates = coordinates[batch_start : batch_start + sw_batch_size]
                patches = [
                    cpu_volume[
                        :,
                        :,
                        depth : depth + patch_depth,
                        height : height + patch_height,
                        width : width + patch_width,
                    ]
                    for depth, height, width in batch_coordinates
                ]
                patch_batch = torch.cat(patches, dim=0).to(inference_device)
                patch_logits = model(patch_batch)
                if not isinstance(patch_logits, torch.Tensor) or patch_logits.ndim != 5:
                    raise ValueError("model must return a tensor shaped [B,C,D,H,W]")
                if patch_logits.shape[0] != len(batch_coordinates):
                    raise ValueError("model output batch size does not match the patch batch size")
                if tuple(patch_logits.shape[2:]) != (
                    patch_depth,
                    patch_height,
                    patch_width,
                ):
                    raise ValueError(
                        "model must preserve patch spatial shape; got "
                        f"{tuple(patch_logits.shape[2:])} for patch "
                        f"{(patch_depth, patch_height, patch_width)}"
                    )

                patch_logits_cpu = patch_logits.detach().to(device="cpu", dtype=torch.float32)
                if output_sum is None:
                    output_sum = torch.zeros(
                        (1, patch_logits_cpu.shape[1], padded_depth, padded_height, padded_width),
                        dtype=torch.float32,
                    )
                    output_count = torch.zeros(
                        (1, 1, padded_depth, padded_height, padded_width),
                        dtype=torch.float32,
                    )

                assert output_count is not None
                for item_index, (depth, height, width) in enumerate(batch_coordinates):
                    output_sum[
                        :,
                        :,
                        depth : depth + patch_depth,
                        height : height + patch_height,
                        width : width + patch_width,
                    ] += patch_logits_cpu[item_index : item_index + 1]
                    output_count[
                        :,
                        :,
                        depth : depth + patch_depth,
                        height : height + patch_height,
                        width : width + patch_width,
                    ] += 1.0
    finally:
        model.train(was_training)

    if output_sum is None or output_count is None:
        raise RuntimeError("No sliding-window patches were generated")
    if torch.any(output_count == 0):
        raise RuntimeError("Sliding-window configuration left uncovered voxels")

    output = output_sum / output_count
    output = output[:, :, :original_depth, :original_height, :original_width]
    return output if input_had_batch else output.squeeze(0)

