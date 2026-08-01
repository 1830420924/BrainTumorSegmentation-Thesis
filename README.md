# BraTS 2020 脑肿瘤 MRI 分割毕业论文项目

本仓库保存基于 BraTS 2020 MRI 数据集的脑肿瘤分割毕业论文项目源码、配置、测试和精简实验结果。
为保护数据隐私并避免大文件上传，仓库不包含原始病例数据、生成图像、模型权重和完整实验产物。

## 当前进度

当前已完成两套相互独立的基线实现：

- **2D 基线**（`BrainTumorSegmentation/`）：使用 FLAIR 二维切片进行全肿瘤二分类分割，
  已完成三组 U-Net 对照实验。
- **3D 主基线**（`BrainTumorSegmentation3D/`）：面向 RTX 3060 Laptop GPU（6 GB 显存）设计的
  轻量化单模态 FLAIR 三维分割流程。训练阶段使用 3D Patch，评估阶段通过滑动窗口完成整例体数据推理。

当前 3D 研究沿用 2D 项目的患者级划分：258 例训练、55 例验证、56 例测试。标签为全肿瘤二值掩膜
（`seg > 0`）。

下一步计划是在不影响当前基线的前提下，单独开展四模态轻量化 3D 扩展实验（`T1`、`T1ce`、`T2`、`FLAIR`）。
该扩展尚未纳入当前基线。

## 已完成的基线结果

下表为当前毕业论文保留的单次训练基线结果。

| 流程 | 实验 | 测试集 Dice | 测试集 IoU | 说明 |
| --- | --- | ---: | ---: | --- |
| 2D | U-Net + BCE | 0.7459 | 0.6607 | 切片级测试评估 |
| 2D | U-Net + BCE-Dice | 0.7536 | 0.6666 | 切片级测试评估 |
| 2D | Attention U-Net + BCE-Dice | 0.7578 | 0.6692 | 当前最佳 2D 结果 |
| 3D | U-Net + BCE | 0.8470 | 0.7491 | 完整患者体积级评估 |
| 3D | U-Net + BCE-Dice | **0.8508** | **0.7562** | 当前最佳 3D 结果 |
| 3D | Attention U-Net + BCE-Dice | 0.8420 | 0.7415 | 召回率最高，但 Dice 与 Precision 较低 |

2D 与 3D 的指标不能直接横向比较：2D 流程评估的是选取的含肿瘤切片，3D 流程评估的是完整患者体积。
更多指标见 [results/baseline_metrics.md](results/baseline_metrics.md)。

## 仓库结构

```text
BrainTumorSegmentation/       # 原始 2D FLAIR 分割项目
  src/brats2d/                # 2D 数据、模型、训练和评估代码
  remote_support/             # 面向 V100 的历史辅助代码，不属于当前主基线

BrainTumorSegmentation3D/     # 独立的轻量化 3D 主基线
  src/brats3d/                # 3D 数据、模型、训练和评估代码
  tests/                      # 核心单元测试

results/                      # 可安全纳入 Git 的精简实验结果
```

## 复现实验

1. 请自行获取 BraTS 2020 训练数据；本仓库不分发该数据集。
2. 根据对应项目的 `requirements.txt` 创建 Python 环境。
3. 将配置文件中的数据路径改为本机实际路径。
4. 在 `BrainTumorSegmentation3D/` 目录中，可按以下顺序运行 3D 基线：

   ```powershell
   python prepare_3d_splits.py
   python check_dataset.py
   python -m unittest discover -s tests -v
   python smoke_test_gpu.py --model unet
   python train_3d.py --experiment unet_bce_dice
   python evaluate_3d.py --experiment unet_bce_dice
   ```

各子项目目录中的 README 提供了更完整的运行流程与配置说明。

## 有意不纳入 Git 的内容

仓库不会包含原始 BraTS 体数据、处理后的 PNG 数据、含本机绝对路径的生成清单、权重文件、NIfTI 预测、
完整实验产物、本地 IDE 设置或虚拟环境。这样既保持仓库可移植，也避免上传病例数据和大文件。
