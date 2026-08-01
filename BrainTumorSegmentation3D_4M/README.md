# BraTS 2020 轻量四模态 3D 分割

这是论文工程中的独立四模态试验目录。它在同一 Git 仓库的
`feature/four-modal` 分支中开发，不会改动已完成的单模态 3D 基线
`BrainTumorSegmentation3D/`。

## 目标与对照

- 任务：BraTS 2020 全肿瘤（WT）二分类分割，标签规则为 `seg > 0`。
- 首个受控试验：3D U-Net + BCE-Dice，和单模态最佳基线保持相同的切分、
  patch/ROI、训练轮数和优化器设置；唯一主变量是输入从 FLAIR 改为四模态。
- 四个输入通道固定为 `T1, T1ce, T2, FLAIR`，即张量 `[C,D,H,W]` 中
  `C=4`。这个顺序写入配置、清单和检查点，不能随意交换。
- 不加载或续训旧的单模态权重；四通道模型必须从头训练。

## 数据与预处理约定

- 原始 BraTS NIfTI 数据不复制，仍从仓库外层的
  `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/` 读取。
- 复用已完成 2D 工程的病人级切分：训练/验证/测试为 `258 / 55 / 56`，避免
  病人泄漏，也使四模态与单模态的比较公平。
- 每例的 T1、T1ce、T2、FLAIR 和分割标签均检查 3D shape 与 affine 对齐。
  特殊标签文件名也通过通用解析器处理。
- 每个模态在自身有限、非零脑区体素上独立 Z-score 标准化，背景保持为 0。
- 训练时所有模态和标签使用同一空间裁剪/翻转；通道轴不会被翻转。强度缩放
  与平移在四个通道之间共享，噪声仅加到图像的非零体素。

## 推荐执行顺序

在本目录中使用配置的 Python 环境执行：

```powershell
python prepare_3d_splits.py
python check_dataset.py --deep
python smoke_test_gpu.py --model unet --full-volume
python train_3d.py --experiment unet_bce_dice_4m --epochs 1
python train_3d.py --experiment unet_bce_dice_4m
python evaluate_3d.py --experiment unet_bce_dice_4m --split test
```

前三步分别生成轻量路径清单、检查全部 369 例的头信息、以及在真实病人数据上
执行四通道 patch 的前向/反向/优化器一步和完整体积滑窗推理。通过后才建议运行
1 个 epoch 的端到端试验，最后再开始正式 50 epoch 训练。

RTX 3060 6 GB 的默认设置是 `patch_size=roi_size=[64,96,96]`、
`base_channels=16`、`batch_size=1`、AMP 和梯度累积 4 步。如果显存检查超过
90%，先把 patch 与 ROI 同时降到 `[48,80,80]`；若因此更改配置，也应以同配置
重新训练 FLAIR 基线后再比较模态收益。

## 不提交的本地内容

`data/` 中的绝对路径清单、`artifacts/`、检查点、预测 NIfTI、原始数据和虚拟
环境均被 Git 忽略。仓库只保存代码、配置、测试和可复现实验说明。
