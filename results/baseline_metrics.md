# Curated baseline metrics

This file records the compact, path-free summary of the completed baseline experiments.
Full logs, checkpoints, visualisations, NIfTI predictions, and per-patient CSV files
remain local and are intentionally excluded from Git.

## 2D FLAIR whole-tumour binary segmentation

| Experiment | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| U-Net + BCE | 0.7459 | 0.6607 | 0.8515 | 0.7465 |
| U-Net + BCE-Dice | 0.7536 | 0.6666 | 0.7932 | 0.7981 |
| Attention U-Net + BCE-Dice | **0.7578** | **0.6692** | 0.7704 | **0.8229** |

The 2D pipeline uses tumour-containing FLAIR slices and reports slice-level test metrics.

## 3D FLAIR whole-tumour binary segmentation

All three experiments use the same 258/55/56 patient split, 3D patch training, and
complete-volume sliding-window test inference.

| Experiment | Dice (mean ± SD) | IoU (mean ± SD) | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| U-Net + BCE | 0.8470 ± 0.1102 | 0.7491 ± 0.1511 | 0.8297 | 0.8846 |
| U-Net + BCE-Dice | **0.8508 ± 0.1181** | **0.7562 ± 0.1546** | **0.8519** | 0.8711 |
| Attention U-Net + BCE-Dice | 0.8420 ± 0.1119 | 0.7415 ± 0.1501 | 0.7986 | **0.9091** |

The current best overall 3D Dice is achieved by **U-Net + BCE-Dice**. The attention model
has the highest recall but does not improve Dice in this single-seed baseline.

## Interpretation limits

- Do not directly compare 2D and 3D Dice values because their evaluation units differ.
- These are single-run baseline results, not evidence of statistically significant
  superiority.
- Any four-modality extension should retain the same patient split and evaluation
  protocol, then be reported as a controlled supplementary experiment.
