# BraTS 2020 3D Attention U-Net（本机 RTX 3060 6GB）

这是原二维项目的独立三维版本。原项目及其训练结果保留在：

```text
D:\KMCU\Thesis\BrainTumorSegmentation
```

三维代码、配置、模型权重和评估结果全部写入：

```text
D:\KMCU\Thesis\BrainTumorSegmentation3D
```

本项目只面向本机 NVIDIA GeForce RTX 3060 Laptop GPU（6GB），不包含远程训练配置。

## 研究范围

- 数据：BraTS 2020 的 369 例有标注训练病例
- 输入：单模态 FLAIR 三维 MRI
- 标签：将原标签 `1/2/4` 合并为全肿瘤二值掩膜
- 模型：3D U-Net 与 3D Attention U-Net
- 划分：复用二维项目的患者级 `258/55/56` 划分，便于公平比较
- 训练：随机正负三维 Patch，不再删除无肿瘤层面
- 评价：完整患者体积滑窗推理，计算患者级 Dice、IoU、Precision、Recall

数据清单只保存原始 NIfTI 路径，不复制 MRI 大文件。

## 3060 6GB 默认配置

默认参数位于 `configs/default_config.yaml`：

```text
Patch / ROI       64 × 96 × 96 (D,H,W)
base_channels     16
batch_size        1
AMP               开启
梯度累积          4
num_workers       0（Windows稳定配置）
滑窗重叠          0.25
每5轮完整验证一次
```

3D模型显存需求远高于2D。不要直接将完整的 `155×240×240` 体积送入网络；训练使用Patch，验证和测试由滑窗拼回完整体积。

## Python环境

推荐继续使用现有解释器：

```text
D:\Miniconda3\envs\pytorch_gpu\python.exe
```

安装缺失依赖：

```powershell
pip install -r requirements.txt
```

## 运行顺序

在 PyCharm 中打开 `BrainTumorSegmentation3D`，工作目录设为项目根目录。
项目已在 `.run` 目录提供按 `01` 到 `05` 排序的本机运行配置，也可以直接按下面的命令执行。

1. 生成或刷新三维患者清单（不会复制NIfTI）：

```powershell
python prepare_3d_splits.py
```

2. 检查数据、标签、归一化和Patch：

```powershell
python check_dataset.py
```

若要核对全部369例NIfTI头信息：

```powershell
python check_dataset.py --deep
```

3. 运行CPU核心测试：

```powershell
python -m unittest discover -s tests -v
```

4. 在本机3060上运行一次真实Patch前向/反向显存测试：

```powershell
python smoke_test_gpu.py --model attention_unet
```

5. 建议一次只训练一个实验：

```powershell
python train_3d.py --experiment unet_bce
python train_3d.py --experiment unet_bce_dice
python train_3d.py --experiment attention_unet_bce_dice
```

中断后可从对应实验的 `last_model.pth` 恢复：

```powershell
python train_3d.py --experiment attention_unet_bce_dice --resume artifacts\experiments\attention_unet_bce_dice\last_model.pth
```

6. 使用最佳权重在完整测试病例上评价：

```powershell
python evaluate_3d.py --experiment attention_unet_bce_dice
```

如需保存三维预测NIfTI：

```powershell
python evaluate_3d.py --experiment attention_unet_bce_dice --save-nifti
```

## 三组对照实验

| 实验 | 目的 |
|---|---|
| `unet_bce` | 3D U-Net基线 |
| `unet_bce_dice` | 验证BCE-Dice复合损失 |
| `attention_unet_bce_dice` | 验证三维注意力门的增益 |

## 输出目录

```text
artifacts/
  dataset_checks/
  experiments/<experiment_name>/
    best_model.pth
    last_model.pth
    training_history.csv
    training_curves.png
    validation_patient_metrics.csv
    evaluation/
```

## 显存不足时

当前默认值已经在本机3060 6GB上完成真实前向和反向测试。如果其他程序占用显存导致 OOM，先关闭占用GPU的软件；仍不足时，将配置中的 `patch_size` 和 `roi_size` 同步改为 `[48, 80, 80]`，并把 `base_channels` 从 `16` 降为 `8`。最后才考虑 `[32, 64, 64]` 与 `base_channels=4`。

完整三维验证会比二维慢很多，这是滑窗覆盖完整患者体积的正常成本。训练前先运行 `smoke_test_gpu.py`，确认当前显存状态可用。
