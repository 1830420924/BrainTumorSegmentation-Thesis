# BraTS 2020 Brain Tumor Segmentation Thesis

This repository contains the source code, configurations, tests, and curated baseline
metrics for a brain-tumour segmentation thesis project based on BraTS 2020 MRI data.
It deliberately excludes patient data, generated images, model weights, and full
experiment artifacts.

## Current status

The completed baseline consists of two independent implementations:

- **2D baseline** (`BrainTumorSegmentation/`): FLAIR slices, whole-tumour binary
  segmentation, and three controlled U-Net experiments.
- **3D main baseline** (`BrainTumorSegmentation3D/`): a lightweight, single-modality
  FLAIR 3D segmentation pipeline designed for an RTX 3060 Laptop GPU with 6 GB VRAM.
  It trains on 3D patches and evaluates complete patient volumes with sliding-window
  inference.

The current 3D study uses the same patient-level split as the earlier 2D study:
258 training, 55 validation, and 56 test patients. The label is the binary whole-tumour
mask (`seg > 0`).

The next planned extension is a separate four-modality lightweight 3D experiment
(`T1`, `T1ce`, `T2`, `FLAIR`). It is not included in this baseline yet.

## Baseline results

The following are the completed, single-run results retained for the thesis baseline.

| Pipeline | Experiment | Test Dice | Test IoU | Notes |
| --- | --- | ---: | ---: | --- |
| 2D | U-Net + BCE | 0.7459 | 0.6607 | Slice-level test evaluation |
| 2D | U-Net + BCE-Dice | 0.7536 | 0.6666 | Slice-level test evaluation |
| 2D | Attention U-Net + BCE-Dice | 0.7578 | 0.6692 | Best 2D result |
| 3D | U-Net + BCE | 0.8470 | 0.7491 | Patient-level complete-volume evaluation |
| 3D | U-Net + BCE-Dice | **0.8508** | **0.7562** | Current best 3D result |
| 3D | Attention U-Net + BCE-Dice | 0.8420 | 0.7415 | Highest recall, but lower Dice/precision |

The 2D and 3D scores must not be treated as directly comparable: the 2D pipeline
evaluates selected tumour slices, while the 3D pipeline evaluates complete patient
volumes. More details are in [results/baseline_metrics.md](results/baseline_metrics.md).

## Repository layout

```text
BrainTumorSegmentation/       # Original 2D FLAIR segmentation project
  src/brats2d/                # 2D data, model, training and evaluation code
  remote_support/             # Archived V100-oriented helper path; not the main baseline

BrainTumorSegmentation3D/     # Independent 3D lightweight baseline
  src/brats3d/                # 3D data, model, training and evaluation code
  tests/                      # Core unit tests

results/                      # Small, curated result summaries safe to version
```

## Reproduction outline

1. Obtain the BraTS 2020 training data separately; it is not distributed in this
   repository.
2. Create a Python environment with the dependencies listed in the relevant project's
   `requirements.txt`.
3. Update the data paths in the configuration file for your local machine.
4. For the 3D baseline, run these commands from `BrainTumorSegmentation3D/`:

   ```powershell
   python prepare_3d_splits.py
   python check_dataset.py
   python -m unittest discover -s tests -v
   python smoke_test_gpu.py --model unet
   python train_3d.py --experiment unet_bce_dice
   python evaluate_3d.py --experiment unet_bce_dice
   ```

See the README in each subproject for its complete workflow and configuration details.

## What is intentionally excluded

The Git repository does not contain raw BraTS volumes, processed PNG data, generated
manifests with local absolute paths, checkpoints, NIfTI predictions, experiment
artifacts, local IDE settings, or virtual environments. This keeps the repository
portable and prevents patient-data and large-file uploads.
