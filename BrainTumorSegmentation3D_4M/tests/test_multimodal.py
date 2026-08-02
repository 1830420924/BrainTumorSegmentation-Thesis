"""CPU tests for the four-modality data contract and model input shape."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.config import FOUR_MODALITIES, load_config
from brats3d.dataset import BraTS3DPatchDataset, _extract_patch
from brats3d.inference import sliding_window_inference
from brats3d.manifest import validate_manifest
from brats3d.models import UNet3D
from brats3d.utils import LoadedMultiModalVolume, load_multimodal_volume, normalize_nonzero_volume


class ScaleModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def _record(patient_id: str = "BraTS20_Training_001") -> dict[str, object]:
    # Deliberately not insertion-ordered by channel, so validation/loading must
    # rely on the declared modality order rather than dict insertion order.
    return {
        "patient_id": patient_id,
        "image_paths": {
            "flair": "flair.nii",
            "t2": "t2.nii",
            "t1ce": "t1ce.nii",
            "t1": "t1.nii",
        },
        "mask_path": "seg.nii",
    }


def _manifest() -> dict[str, object]:
    return {
        "dataset": {"modalities": list(FOUR_MODALITIES)},
        "counts": {"train": 1, "val": 0, "test": 0},
        "splits": {"train": [_record()], "val": [], "test": []},
    }


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_declares_four_ordered_channels(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default_config.yaml")
        self.assertEqual(config["dataset"]["modalities"], list(FOUR_MODALITIES))
        self.assertEqual(config["training"]["in_channels"], 4)

    def test_configuration_rejects_channel_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.yaml"
            config_path.write_text(
                "dataset:\n"
                "  modalities: [t1, t1ce, t2, flair]\n"
                "training:\n"
                "  in_channels: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "in_channels"):
                load_config(config_path)


class ManifestTests(unittest.TestCase):
    def test_manifest_requires_all_four_named_modalities(self) -> None:
        validate_manifest(_manifest())
        incomplete = _manifest()
        del incomplete["splits"]["train"][0]["image_paths"]["flair"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest(incomplete)


class MultiModalLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        affine = np.eye(4, dtype=np.float32)
        mask = np.zeros((2, 3, 4), dtype=np.float32)
        mask[1, 2, 3] = 4.0
        self.arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {"seg.nii": (mask, affine)}
        for index, modality in enumerate(FOUR_MODALITIES, start=1):
            image = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + index
            image[0, 0, 0] = 0.0
            self.arrays[f"{modality}.nii"] = (image, affine)

    def _fake_load(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        return self.arrays[str(path)]

    def test_loader_stacks_explicit_channel_order_and_normalizes_each_channel(self) -> None:
        record = _record()
        with mock.patch("brats3d.utils.load_nifti_array", side_effect=self._fake_load):
            loaded = load_multimodal_volume(
                record["image_paths"],  # type: ignore[arg-type]
                "seg.nii",
                modalities=FOUR_MODALITIES,
            )

        self.assertEqual(loaded.modalities, FOUR_MODALITIES)
        self.assertEqual(loaded.image.shape, (4, 4, 2, 3))
        self.assertEqual(loaded.mask.shape, (4, 2, 3))
        self.assertEqual(int(loaded.mask.sum()), 1)
        for channel, modality in enumerate(FOUR_MODALITIES):
            expected_xyz, _ = normalize_nonzero_volume(self.arrays[f"{modality}.nii"][0])
            expected_dhw = np.transpose(expected_xyz, (2, 0, 1))
            np.testing.assert_allclose(loaded.image[channel], expected_dhw)
            self.assertIn(modality, loaded.normalization)

    def test_loader_rejects_any_misaligned_modality(self) -> None:
        self.arrays["t2.nii"] = (np.ones((3, 3, 4), dtype=np.float32), np.eye(4, dtype=np.float32))
        record = _record()
        with mock.patch("brats3d.utils.load_nifti_array", side_effect=self._fake_load):
            with self.assertRaisesRegex(ValueError, "t2/seg shape mismatch"):
                load_multimodal_volume(
                    record["image_paths"],  # type: ignore[arg-type]
                    "seg.nii",
                    modalities=FOUR_MODALITIES,
                )


class PatchTests(unittest.TestCase):
    def test_crop_uses_one_spatial_origin_for_all_channels(self) -> None:
        image = np.stack(
            [np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) + 1_000 * channel for channel in range(4)]
        )
        patch, origin = _extract_patch(image, center=(1, 2, 3), patch_size=(3, 4, 4))
        self.assertEqual(origin, (0, 0, 1))
        np.testing.assert_array_equal(patch, image[:, 0:3, 0:4, 1:5])

    def test_augmentation_flips_only_spatial_axes_and_keeps_channel_order(self) -> None:
        dataset = BraTS3DPatchDataset.__new__(BraTS3DPatchDataset)
        dataset.flip_probability = 1.0
        dataset.intensity_scale = 0.0
        dataset.intensity_shift = 0.0
        dataset.noise_std = 0.0
        image = np.stack(
            [np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100 * channel for channel in range(4)]
        )
        mask = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        augmented_image, augmented_mask = dataset._augment_patch(
            image.copy(), mask.copy(), np.random.default_rng(7)
        )
        np.testing.assert_array_equal(augmented_image, np.flip(image, axis=(1, 2, 3)))
        np.testing.assert_array_equal(augmented_mask, np.flip(mask, axis=(0, 1, 2)))
        self.assertTrue(augmented_image.flags["C_CONTIGUOUS"])

    def test_patch_dataset_returns_cdhw_without_an_extra_channel_axis(self) -> None:
        image = np.ones((4, 4, 5, 6), dtype=np.float32)
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[2, 3, 4] = 1
        loaded = LoadedMultiModalVolume(
            image=image,
            mask=mask,
            affine=np.eye(4, dtype=np.float32),
            original_shape=(5, 6, 4),
            modalities=FOUR_MODALITIES,
            normalization={
                modality: {"mean": 0.0, "std": 1.0, "nonzero_voxels": 1.0}
                for modality in FOUR_MODALITIES
            },
        )
        config = {
            "dataset": {
                "modalities": list(FOUR_MODALITIES),
                "patch_size": [2, 3, 4],
                "patches_per_patient": 1,
                "positive_patch_ratio": 1.0,
                "augmentation": {"enabled": False},
            }
        }
        with mock.patch("brats3d.dataset.load_multimodal_volume", return_value=loaded):
            sample = BraTS3DPatchDataset(_manifest(), "train", config=config, cache_size=0)[0]
        self.assertEqual(tuple(sample["image"].shape), (4, 2, 3, 4))
        self.assertEqual(tuple(sample["mask"].shape), (1, 2, 3, 4))


class FourChannelModelTests(unittest.TestCase):
    def test_unet_and_sliding_window_accept_four_channels(self) -> None:
        model = UNet3D(in_channels=4, out_channels=1, base_channels=2)
        inputs = torch.randn(1, 4, 32, 32, 32)
        self.assertEqual(model(inputs).shape, (1, 1, 32, 32, 32))

        volume = torch.randn(1, 4, 3, 5, 2)
        output = sliding_window_inference(ScaleModel(), volume, patch_size=(4, 4, 4), overlap=0.25)
        self.assertEqual(output.shape, volume.shape)
        torch.testing.assert_close(output, volume * 2.0)


if __name__ == "__main__":
    unittest.main()
