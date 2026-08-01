# BraTS 2020 2D Attention U-Net

PyCharm project for BraTS 2020 2D brain tumor segmentation using `FLAIR` slices, binary masks, `U-Net`, and `Attention U-Net`.

## Project Root

Open this directory in PyCharm:

```text
D:\KMCU\Thesis\BrainTumorSegmentation
```

Use this interpreter:

```text
D:\Miniconda3\envs\pytorch_gpu\python.exe
```

## Install Missing Dependency

The current `pytorch_gpu` environment is missing `nibabel`, which is required to read `.nii` files.

```powershell
conda run -n pytorch_gpu pip install nibabel
```

## Directory Layout

```text
BrainTumorSegmentation
├── artifacts
├── configs
├── data_processed
├── scripts
├── src
├── check_dataset.py
├── prepare_2d_dataset.py
├── train.py
└── evaluate.py
```

## Recommended PyCharm Run Configurations

Create 4 run configurations in this order:

1. `check_dataset`
   - Script path: `D:\KMCU\Thesis\BrainTumorSegmentation\check_dataset.py`
   - Working directory: `D:\KMCU\Thesis\BrainTumorSegmentation`

2. `prepare_2d_dataset`
   - Script path: `D:\KMCU\Thesis\BrainTumorSegmentation\prepare_2d_dataset.py`
   - Working directory: `D:\KMCU\Thesis\BrainTumorSegmentation`

3. `train`
   - Script path: `D:\KMCU\Thesis\BrainTumorSegmentation\train.py`
   - Working directory: `D:\KMCU\Thesis\BrainTumorSegmentation`

4. `evaluate`
   - Script path: `D:\KMCU\Thesis\BrainTumorSegmentation\evaluate.py`
   - Working directory: `D:\KMCU\Thesis\BrainTumorSegmentation`

## Workflow

1. Run `check_dataset.py`
2. Run `prepare_2d_dataset.py`
3. Run `train.py`
4. Run `evaluate.py`

## Default Experiments

`train.py` is configured to run:

1. `U-Net + BCE`
2. `U-Net + BCE + Dice`
3. `Attention U-Net + BCE + Dice`

Use `--experiment <name>` to run a single experiment.
