"""Three-dimensional U-Net model definitions.

The models in this module consume tensors shaped ``[B, C, D, H, W]`` and
return raw logits with the same spatial shape.  Instance normalization is used
throughout because 3-D medical-image training commonly uses very small batch
sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Two 3-D convolutions, each followed by instance norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """Downsample by two and apply a double-convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _resize_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Resize a decoder tensor exactly to a skip tensor's spatial shape."""

    return F.interpolate(
        x,
        size=skip.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


class Up3D(nn.Module):
    """Upsample, concatenate an encoder skip, and fuse both tensors."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _resize_to_skip(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionGate3D(nn.Module):
    """Additive attention gate for filtering an encoder skip connection."""

    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gating_projection = nn.Sequential(
            nn.Conv3d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels, affine=True),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gating_signal: torch.Tensor, skip_tensor: torch.Tensor) -> torch.Tensor:
        if gating_signal.shape[2:] != skip_tensor.shape[2:]:
            gating_signal = _resize_to_skip(gating_signal, skip_tensor)
        attention = self.attention(
            self.gating_projection(gating_signal) + self.skip_projection(skip_tensor)
        )
        return skip_tensor * attention


class AttentionUp3D(nn.Module):
    """Upsampling block whose skip connection is filtered by attention."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = AttentionGate3D(
            gating_channels=in_channels,
            skip_channels=skip_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _resize_to_skip(x, skip)
        gated_skip = self.attention(x, skip)
        return self.conv(torch.cat([gated_skip, x], dim=1))


def _validate_model_arguments(in_channels: int, out_channels: int, base_channels: int) -> None:
    for name, value in (
        ("in_channels", in_channels),
        ("out_channels", out_channels),
        ("base_channels", base_channels),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 5:
        raise ValueError(f"Expected input shaped [B, C, D, H, W], got {tuple(x.shape)}")
    if min(x.shape[2:]) < 32:
        raise ValueError(
            "Each spatial dimension must be at least 32 voxels. Four downsampling stages "
            "followed by InstanceNorm require more than one bottleneck voxel; "
            f"got {tuple(x.shape[2:])}"
        )


class UNet3D(nn.Module):
    """Four-level 3-D U-Net returning unactivated segmentation logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        _validate_model_arguments(in_channels, out_channels, base_channels)

        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.enc2 = Down3D(base_channels, base_channels * 2)
        self.enc3 = Down3D(base_channels * 2, base_channels * 4)
        self.enc4 = Down3D(base_channels * 4, base_channels * 8)
        self.bottleneck = Down3D(base_channels * 8, base_channels * 16)

        self.up1 = Up3D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x)
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        x = self.bottleneck(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.head(x)


class AttentionUNet3D(nn.Module):
    """3-D U-Net with additive attention gates on all encoder skips."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        _validate_model_arguments(in_channels, out_channels, base_channels)

        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.enc2 = Down3D(base_channels, base_channels * 2)
        self.enc3 = Down3D(base_channels * 2, base_channels * 4)
        self.enc4 = Down3D(base_channels * 4, base_channels * 8)
        self.bottleneck = Down3D(base_channels * 8, base_channels * 16)

        self.up1 = AttentionUp3D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp3D(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x)
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        x = self.bottleneck(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.head(x)


def build_model(
    model_name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 16,
) -> nn.Module:
    """Build a supported 3-D segmentation model by name."""

    normalized_name = model_name.strip().lower().replace("-", "_")
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_channels": base_channels,
    }
    if normalized_name in {"unet", "unet3d", "unet_3d"}:
        return UNet3D(**kwargs)
    if normalized_name in {"attention_unet", "attention_unet3d", "attention_unet_3d"}:
        return AttentionUNet3D(**kwargs)
    raise ValueError(
        f"Unsupported model name {model_name!r}; expected 'unet3d' or 'attention_unet3d'"
    )


# Short aliases ease migration from the existing 2-D project while the explicit
# 3-D class names keep experiment reports unambiguous.
UNet = UNet3D
AttentionUNet = AttentionUNet3D
