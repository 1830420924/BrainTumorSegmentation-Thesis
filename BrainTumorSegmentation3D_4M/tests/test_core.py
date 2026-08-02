"""CPU-only unit tests for the 3-D model core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats3d.inference import sliding_window_inference
from brats3d.losses import BCEDiceLoss, DiceLoss, build_loss
from brats3d.metrics import volume_segmentation_metrics
from brats3d.models import AttentionUNet3D, UNet3D, build_model


class ScaleModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class ModelTests(unittest.TestCase):
    def test_unet_models_preserve_spatial_shape(self) -> None:
        torch.manual_seed(7)
        inputs = torch.randn(1, 1, 32, 32, 32)
        for model_type in (UNet3D, AttentionUNet3D):
            with self.subTest(model=model_type.__name__):
                model = model_type(in_channels=1, out_channels=1, base_channels=2)
                outputs = model(inputs)
                self.assertEqual(outputs.shape, inputs.shape)

    def test_model_factory_and_invalid_input(self) -> None:
        self.assertIsInstance(build_model("unet3d", base_channels=2), UNet3D)
        self.assertIsInstance(
            build_model("attention-unet-3d", base_channels=2), AttentionUNet3D
        )
        with self.assertRaises(ValueError):
            build_model("yolo")
        with self.assertRaises(ValueError):
            UNet3D(base_channels=2)(torch.zeros(1, 1, 8, 16, 16))


class LossTests(unittest.TestCase):
    def test_dice_and_combined_loss_support_backpropagation(self) -> None:
        logits = torch.tensor(
            [[[[[-8.0, 8.0], [8.0, -8.0]]]]],
            requires_grad=True,
        )
        targets = torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]])
        dice_loss = DiceLoss()(logits, targets)
        combined_loss = BCEDiceLoss()(logits, targets)
        self.assertLess(dice_loss.item(), 0.001)
        self.assertLess(combined_loss.item(), 0.002)
        combined_loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_loss_factory(self) -> None:
        self.assertIsInstance(build_loss("bce"), nn.BCEWithLogitsLoss)
        self.assertIsInstance(build_loss("dice"), DiceLoss)
        self.assertIsInstance(build_loss("bce+dice"), BCEDiceLoss)
        with self.assertRaises(ValueError):
            build_loss("unknown")


class MetricTests(unittest.TestCase):
    def test_known_patient_volume_metrics(self) -> None:
        logits = torch.tensor([[[[[10.0, -10.0], [10.0, -10.0]]]]])
        target = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
        metrics = volume_segmentation_metrics(logits, target)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_correctly_empty_volume_scores_one(self) -> None:
        prediction = torch.zeros(1, 3, 4, 5)
        target = torch.zeros_like(prediction)
        metrics = volume_segmentation_metrics(prediction, target, from_logits=False)
        self.assertTrue(all(value == 1.0 for value in metrics.values()))


class SlidingWindowTests(unittest.TestCase):
    def test_overlap_and_padding_reconstruct_exact_output(self) -> None:
        volume = torch.arange(1 * 3 * 5 * 2, dtype=torch.float32).reshape(1, 3, 5, 2)
        model = ScaleModel()
        model.train()
        output = sliding_window_inference(
            model,
            volume,
            patch_size=(4, 4, 4),
            overlap=(0.5, 0.25, 0.5),
            sw_batch_size=2,
        )
        self.assertEqual(output.shape, volume.shape)
        torch.testing.assert_close(output, volume * 2.0)
        self.assertTrue(model.training, "The helper must restore the model's training state")
        self.assertEqual(output.device.type, "cpu")

    def test_singleton_batch_rank_is_preserved(self) -> None:
        volume = torch.randn(1, 1, 4, 4, 4)
        output = sliding_window_inference(
            ScaleModel(), volume, patch_size=4, overlap=0.0
        )
        self.assertEqual(output.shape, volume.shape)
        torch.testing.assert_close(output, volume * 2.0)

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sliding_window_inference(
                ScaleModel(), torch.zeros(1, 4, 4, 4), patch_size=4, overlap=1.0
            )


if __name__ == "__main__":
    unittest.main()
