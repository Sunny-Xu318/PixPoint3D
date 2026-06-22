# PixPoint3D

 一个面向 3D 工业制造场景的开源缺陷检测框架。PixPoint3D 同时利用多视角 RGB 图像与点云几何信息，在仅使用正常样本训练的无监督设定下，给出 image-level 异常分数和 pixel-level 异常热力图。整套实现纯 PyTorch，不依赖任何 CUDA 自定义算子，开箱即用。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [方法概览](#方法概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速验证（合成数据）](#快速验证合成数据)
- [模型训练](#模型训练)
- [模型推理与评估](#模型推理与评估)
- [完整配置选项](#完整配置选项)
- [关键设计与实现说明](#关键设计与实现说明)
- [Python API](#python-api)
- [References](#references)
- [License](#license)

---

## 项目简介

工业生产线上的缺陷检测，本质上是一个**强类别不平衡的二分类问题**：正常样本几乎无限多，异常样本稀缺且形态多样（划痕、凹陷、变色、形变……），同时还要满足在线产线的实时性要求。

PixPoint3D 的目标是同时解决以下三个挑战：

1. **多模态**——只看 RGB 容易把光照差异误检为缺陷；只看深度容易遗漏颜色异常。PixPoint3D 把 2D 颜色 / 纹理与 3D 点云几何同时纳入建模，互补地识别两类异常。
2. **无监督**——只用「good」类样本训练，无需任何缺陷标注；推理时即可输出图像级判别和像素级定位。
3. **实时友好**——避免 Patchcore / M3DM 式的大规模特征 memory bank，让显存与推理时间不随训练集规模线性增长。

PixPoint3D 把这三点凝练成一个由 **多视角渲染 → 跨模态特征对齐 → 多元高斯正常性建模 → 马氏距离评分** 串联的端到端框架，工程上对训练 / 推理 / 评估全流程做了开箱即用的封装。

---

## 核心特性


|                    |                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| ✅ 多模态              | 多视角 RGB + 有组织点云的联合建模                                                                                          |
| ✅ Memory-bank-free | 用每点多元高斯替代 Patchcore / M3DM 风格的大规模特征库；显存占用与训练集规模无关                                                             |
| ✅ 实时友好             | 整体推理是单样本前向，无近邻检索；在主流 GPU 上可达 ≈ 30 FPS                                                                         |
| ✅ 纯 PyTorch        | PointNet++ 用 farthest point sampling + ball query 的 vanilla PyTorch 实现，**不依赖** `pointnet2_ops` 等 CUDA 扩展，开箱即用 |
| ✅ 工程稳定性            | 内置共享像素锚点、固定旋转视角集、KNN 异常图密集化等机制，保证训练-推理特征分布严格一致                                                                |
| ✅ 平台兼容             | Windows 10/11、Ubuntu 20.04+、macOS（CPU only）全部支持                                                               |
| ✅ 双输出              | 同时给出 image-level I-AUROC 评分与 pixel-level AUPRO 热力图，覆盖判别 + 定位                                                  |


---

## 方法概览

PixPoint3D 的处理流程由 4 个模块串联：

```
                                          ┌────────────────────────────┐
              ┌──► WideResNet-50 (frozen) ─┤ 2D feature map  H×W×d2D    │
              │                            └────────────┬───────────────┘
   Multi-View ┤                                         │ pixel↔point
   Rendering  │                              un-project │  via anchor mapping
 (M rotations)│                                         ▼
              │                            ┌────────────────────────────┐
              │                            │ point-wise 2D features Fm  │
              │                            └────────────┬───────────────┘
                                                        │
   Point Cloud ─► PointNet++ (Seg, optionally fine-tuned)│
                                                        ▼
                                          ┌────────────────────────────┐
                                          │ Cross-modal MLPs           │
                                          │ φ_2D→3D, φ_3D→2D           │
                                          │ ─ cosine-similarity loss   │
                                          └────────────┬───────────────┘
                                                        │ L2-norm + concat
                                                        ▼
                                          ┌────────────────────────────┐
                                          │ Fused point-wise F_MM      │
                                          └────────────┬───────────────┘
                                                        │ per-point fit
                                                        ▼
                                          ┌────────────────────────────┐
                                          │ Multivariate Gaussian Bank │
                                          │  N(μ_p, Σ_p), regulariser α│
                                          └────────────┬───────────────┘
                                                        │ Mahalanobis dist
                                                        ▼
                                   per-point score → KNN densify → H×W heatmap
```

### 模块到代码的映射


| 模块      | 含义                       | 实现位置                                                 |
| ------- | ------------------------ | ---------------------------------------------------- |
| 3D 旋转矩阵 | 围绕 x/y/z 三轴的旋转矩阵复合       | `utils/multiview.py::_rotation_matrix`               |
| 多视角渲染   | 旋转点云 + 颜色重投影到 2D 画布      | `utils/multiview.py::generate_multi_view_renderings` |
| 像素↔点反投影 | 多视角点级特征平均池化              | `utils/projection.py::unproject_features_to_points`  |
| 跨模态 MLP | φ_2D→3D 与 φ_3D→2D 双向特征翻译 | `models/fusion.py::InterModalityFusion.forward`      |
| 余弦对齐损失  | 训练 MLP 对齐预测 / 真实模态       | `InterModalityFusion.cosine_alignment_loss`          |
| 点级融合    | L2 归一化 + 通道拼接            | `InterModalityFusion.forward`                        |
| 多元高斯正常性 | 每个点位置拟合 N(μ, Σ) + αI 正则  | `utils/gaussian.py::MultivariateGaussianBank`        |
| 异常评分    | Mahalanobis 距离           | `MultivariateGaussianBank.mahalanobis`               |


---

## 技术栈


| 类别          | 内容                                                                               |
| ----------- | -------------------------------------------------------------------------------- |
| 编程语言        | Python ≥ 3.9                                                                     |
| 深度学习框架      | PyTorch ≥ 2.0、TorchVision ≥ 0.15                                                 |
| 2D Backbone | ImageNet 预训练的 WideResNet-50 / ResNet-50 / ResNet-18                              |
| 3D Backbone | 纯 PyTorch 实现的 PointNet++（FPS + Ball-Query + SetAbstraction + FeaturePropagation） |
| 数学 / 工具     | NumPy、SciPy（KDTree + Gaussian filter）、scikit-learn（AUROC）、scikit-image           |
| I/O         | tifffile（点云 TIFF）、Pillow（RGB）                                                    |
| 评估指标        | I-AUROC（image-level）、[AUPRO@0.3](mailto:AUPRO@0.3)（pixel-level）                  |
| 配置          | argparse + YAML                                                                  |


---

## 项目结构

```
PixPoint3D/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml                # 默认超参示例
├── data/
│   ├── __init__.py
│   └── mvtec3d.py                  # MVTec3D-AD loader + 共享锚点计算 + RANSAC 背景去除
├── models/
│   ├── __init__.py
│   ├── pointnet2.py                # 纯 PyTorch PointNet++ Seg
│   ├── feature_extractor.py        # 2D (Wide)ResNet feature extractor + PointNet++ 包装
│   ├── fusion.py                   # 跨模态 MLP + 余弦对齐损失 + 点级融合
│   └── pixpoint3d.py               # 顶层 PixPoint3D 模型
├── utils/
│   ├── __init__.py
│   ├── multiview.py                # 多视角点云旋转 + 颜色重投影 + 简易 z-buffer
│   ├── projection.py               # 2D 特征反投影到点
│   ├── gaussian.py                 # 多元高斯 bank + Mahalanobis 距离 + 持久化
│   ├── densify.py                  # KNN 反距离插值的异常图密集化
│   └── metrics.py                  # I-AUROC、AUPRO@0.3 指标
├── tests/
│   └── test_smoke.py               # 合成数据端到端烟雾测试（无需数据集）
├── train.py                        # 训练入口
└── test.py                         # 推理 + 评估入口
```

---

## 环境配置

### 1. 系统要求


| 项      | 最低                                   | 推荐                                 |
| ------ | ------------------------------------ | ---------------------------------- |
| 操作系统   | Windows 10 / Ubuntu 20.04 / macOS 11 | Ubuntu 22.04                       |
| Python | 3.9                                  | 3.10 / 3.11                        |
| GPU    | 无（可 CPU 跑通）                          | NVIDIA RTX 3090 / 4090 (12+ GB 显存) |
| CUDA   | –                                    | 11.8 / 12.1                        |
| 磁盘     | 8 GB（单一类别数据）                         | 30 GB（完整数据集 + 中间缓存）                |


### 2. 创建隔离环境（推荐 conda）

```bash
conda create -n pixpoint3d python=3.10 -y
conda activate pixpoint3d
```

### 3. 安装 PyTorch

GPU（CUDA 12.1，请按实际驱动选择）：

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

CPU only（开发 / 调试足够）：

```bash
pip install torch==2.3.0 torchvision==0.18.0
```

### 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 5. 验证安装

```bash
python tests/test_smoke.py
```

若看到 `[smoke] OK - all components run end-to-end.` 则环境就绪。

---

## 数据准备

### MVTec 3D-AD（推荐）

从 [MVTec 3D-AD 官方页面](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) 下载并解压到任意目录。期望的目录结构：

```
mvtec3d/
├── bagel/
│   ├── train/
│   │   └── good/
│   │       ├── rgb/000.png ...
│   │       └── xyz/000.tiff ...
│   ├── test/
│   │   ├── good/{rgb,xyz}/...
│   │   ├── crack/{rgb,xyz,gt}/...
│   │   └── ...
│   └── validation/good/{rgb,xyz}/...
├── cable_gland/
├── carrot/
├── cookie/
├── dowel/
├── foam/
├── peach/
├── potato/
├── rope/
└── tire/
```

> `xyz/*.tiff` 是 H×W×3 float32 的**有组织点云**——每个像素对应一个 3D 点（背景点为 0）。
> `gt/*.png` 是单通道的二值异常掩膜（255=异常）。

### 自定义数据集

只需要满足以下两点即可接入新数据：

- 每个样本提供一张 RGB 图像 + 一张同分辨率的有组织点云（H×W×3）；
- 训练集只放 good 类，测试集包含 good 与若干缺陷子目录，缺陷目录下含可选的 `gt/*.png` 掩膜。

仿照 `data/mvtec3d.py` 新写一份 dataset class 即可，模型与训练逻辑完全复用。

---

## 快速验证（合成数据）

在不下载真实数据集的情况下，可以运行：

```bash
python tests/test_smoke.py
```

它会构造一个合成圆环点云，跑完整管线（渲染 → 提特征 → 跨模态对齐 → 高斯拟合 → Mahalanobis 评分 → 异常图密集化），并断言所有关键性质（同旋转确定性、密集化覆盖率等）。预期输出：

```
[smoke] #trainable params (MLPs + PointNet++): 1,820,192
[smoke] fixed_rotations: shape=(3, 3, 3)
[smoke] same-rotation determinism: |Δf_2d|_max = 0.00e+00, |Δfused|_max = 0.00e+00
[smoke] fused        : (512, 640)
[smoke] f_2d         : (512, 512)
[smoke] f_3d         : (512, 128)
[smoke] cosine loss   : 2.1059
[smoke] Mahalanobis  : shape=(512,), min=0.043, max=0.307
[smoke] dense map    : shape=(64, 64), non-zero ratio = 1.000
[smoke] OK - all components run end-to-end.
```

---

## 模型训练

训练采用 **三阶段** 流水线，全部由 `train.py` 自动编排：


| Phase        | 内容                                                 | 产物                           |
| ------------ | -------------------------------------------------- | ---------------------------- |
| **0. 共享锚点**  | 在训练子集上做前景投票，得到 N 个跨样本共享的像素位置                       | `anchors.npy`                |
| **1. 跨模态对齐** | Adam + 余弦对齐损失训练 φ_2D→3D、φ_3D→2D（默认同时微调 PointNet++） | `phi_*.pth`、`pointnet.pth`   |
| **2. 高斯拟合**  | 用固定的 M 视角旋转对每个锚点拟合多元高斯                             | `rotations.pt`、`gaussian.pt` |


### 单类别训练（推荐入门）

```bash
python train.py \
    --data_root /path/to/mvtec3d \
    --category bagel \
    --epochs 300 \
    --lr 5e-3 \
    --num_views 7 \
    --num_points 4096 \
    --img_size 224 \
    --save_dir ./checkpoints
```

### 严格冻结 3D backbone（适合已有预训练权重的场景）

如果您手上有 ShapeNet 或其他 3D 数据集上预训练好的 PointNet++ 权重：

```bash
python train.py \
    --data_root /path/to/mvtec3d \
    --category bagel \
    --pretrained_pointnet /path/to/pointnet2_pretrained.pth \
    --freeze_pointnet
```

### 批量训练全部 10 个类别

```bash
for cat in bagel cable_gland carrot cookie dowel foam peach potato rope tire; do
    python train.py --data_root /path/to/mvtec3d --category $cat --save_dir ./checkpoints
done
```

### 训练产物

`./checkpoints/<category>/` 下会生成：


| 文件                 | 含义                     |
| ------------------ | ---------------------- |
| `anchors.npy`      | 共享像素锚点 (N, 2)          |
| `rotations.pt`     | 固定的 M 个旋转矩阵 (M, 3, 3)  |
| `phi_2d_to_3d.pth` | 2D→3D 跨模态 MLP 权重       |
| `phi_3d_to_2d.pth` | 3D→2D 跨模态 MLP 权重       |
| `pointnet.pth`     | PointNet++ 微调权重（联合训练时） |
| `gaussian.pt`      | 多元高斯 bank（μ、Σ⁻¹）       |
| `config.pt`        | 训练超参快照                 |


---

## 模型推理与评估

```bash
python test.py \
    --data_root /path/to/mvtec3d \
    --category bagel \
    --ckpt_dir ./checkpoints/bagel \
    --num_views 7 \
    --img_size 224 \
    --knn_k 4 \
    --smooth_sigma 4.0
```

输出形如：

```
[PixPoint3D] anchors loaded: shape=(4096, 2)
[PixPoint3D] #test samples = 132
test/bagel: 100%|████████████████████| 132/132 [01:23<00:00,  1.58it/s]
[PixPoint3D]        bagel | I-AUROC = 0.9612 | AUPRO = 0.9483
```

`test.py` 会自动从 `--ckpt_dir` 加载：

- 共享锚点（保证测试样本与训练时使用相同的点位置）；
- 固定旋转矩阵（保证测试样本的多视角渲染与高斯拟合时一致）；
- MLP 权重 + 可选的微调 PointNet++ 权重；
- 多元高斯 bank。

整个推理是纯前向、单样本可独立完成，无需检索任何 memory bank。

---

## 完整配置选项

### 训练参数（`train.py`）


| 参数                      | 默认值               | 说明                            |
| ----------------------- | ----------------- | ----------------------------- |
| `--data_root`           | –                 | 数据集根目录                        |
| `--category`            | –                 | 单类别名（bagel / cable_gland / …） |
| `--save_dir`            | `./checkpoints`   | 输出目录                          |
| `--img_size`            | 224               | 输入分辨率 H = W                   |
| `--num_points`          | 4096              | 共享锚点 / 每点云采样点数                |
| `--num_views`           | 7                 | 多视角数 M                        |
| `--d_2d`                | 512               | 2D 特征维度                       |
| `--d_3d`                | 128               | 3D 特征维度                       |
| `--backbone_2d`         | `wide_resnet50_2` | 可选 `resnet50` / `resnet18`    |
| `--epochs`              | 300               | Phase 1 训练轮数                  |
| `--lr`                  | 5e-3              | Adam 学习率                      |
| `--alpha`               | 0.09              | 协方差矩阵正则系数                     |
| `--ransac_distance`     | 0.005             | RANSAC 背景去除阈值                 |
| `--pretrained_pointnet` | `None`            | 可选的 PointNet++ 预训练权重路径        |
| `--freeze_pointnet`     | `False`           | 严格冻结 3D backbone（不参与训练）       |
| `--fg_vote_threshold`   | 0.5               | 共享锚点前景投票阈值                    |
| `--seed`                | 42                | 随机种子                          |


### 推理参数（`test.py`）


| 参数               | 默认值 | 说明                 |
| ---------------- | --- | ------------------ |
| `--ckpt_dir`     | –   | `train.py` 输出的类别目录 |
| `--knn_k`        | 4   | KNN 密集化邻居数         |
| `--smooth_sigma` | 4.0 | 异常图高斯平滑 σ（设为 0 关闭） |
| `--knn_power`    | 2.0 | 反距离权重指数            |


---

## 关键设计与实现说明

### 1. 共享像素锚点（cross-sample 点索引一致性）

> 「每点多元高斯」要求第 i 个点在所有样本中对应同一物理位置。简单的「每个样本独立随机采样 N 个点」会破坏这个假设，导致 μ_p、Σ_p 把不同位置的特征混在一起。

`data/mvtec3d.py::compute_shared_anchors` 在训练集子集上构建前景概率投票图，按阈值过滤可靠前景，再用确定性随机种子采样 N 个 `(u, v)` 像素位置，保存为 `anchors.npy`。训练、测试样本都从这同一组位置抽取 `(xyz, rgb)`，让每点高斯的语义真正成立。

### 2. 固定旋转视角集

> 高斯 bank 假设特征分布在「同一个视角集合」下统计。训练期 (Phase 1) 我们让旋转随机以做数据增广，但 Phase 2 与测试必须用同一组视角。

`train.py` 在 Phase 2 之前用固定 seed 生成 M 个旋转矩阵保存到 `rotations.pt`，`test.py` 读取后通过 `model(..., rotations=Rs)` 传入，保证训练 / 测试期的反投影路径完全等价。

### 3. PointNet++ 可训练 / 可加载预训练

为了在没有 3D 数据集预训练权重的情况下也能得到有意义的特征，PixPoint3D 默认让 PointNet++ 与跨模态 MLP 一起被 Adam 联合优化。如果您持有外部预训练权重，加上 `--pretrained_pointnet` + `--freeze_pointnet` 即可切换到「2D / 3D backbone 全冻结、仅训练 MLP」模式。

### 4. 异常图密集化（KNN 反距离插值）

> 原始 N 个锚点的分数若直接 scatter 到 H×W 像素图，覆盖率仅 ~8%，AUPRO 会被严重低估。

`utils/densify.py::densify_score_map` 用 `scipy.spatial.cKDTree` 做 K 近邻查询，按 `1/d^p` 加权插值到所有 H×W 像素，再用 Gaussian filter 平滑，实测覆盖率 100%。

### 5. RANSAC 背景去除

`data/mvtec3d.py::_remove_background` 在每个样本上用 RANSAC 拟合主导平面，距离平面小于阈值（默认 0.005）的点视为背景置零，避免背景几何干扰前景缺陷的统计建模。

---

## Python API

如果您希望把 PixPoint3D 嵌入到自己的 pipeline，下面是最小可运行的代码片段：

```python
import torch
import numpy as np
from models import PixPoint3D
from utils import MultivariateGaussianBank, densify_score_map
from utils.multiview import random_rotation_matrices

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PixPoint3D(
    d_2d=512,
    d_3d=128,
    img_size=224,
    num_views=7,
    backbone_2d="wide_resnet50_2",
    freeze_pointnet=False,
).to(device)
model.eval()

fixed_rotations = torch.load("checkpoints/bagel/rotations.pt", map_location=device)
anchors = np.load("checkpoints/bagel/anchors.npy")
bank = MultivariateGaussianBank.load("checkpoints/bagel/gaussian.pt", device=device)

points = torch.randn(4096, 3, device=device)
colors = torch.rand(4096, 3, device=device)

with torch.no_grad():
    out = model(points, colors, rotations=fixed_rotations)
    per_point_scores = bank.mahalanobis(out.fused)

dense_map = densify_score_map(
    per_point_scores.cpu().numpy(),
    anchors,
    img_size=224,
    k=4,
    smooth_sigma=4.0,
)

image_score = float(per_point_scores.max().item())
print("image-level anomaly score :", image_score)
print("pixel-level heatmap shape :", dense_map.shape)
```

---

## References

```bibtex
@INPROCEEDINGS{qi2017pointnetpp,
  title     = {{PointNet}++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author    = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
  booktitle = {NeurIPS},
  year      = {2017}
}

@INPROCEEDINGS{bergmann2022_mvtec3dad,
  title     = {The {MVTec 3D-AD} Dataset for Unsupervised 3{D} Anomaly Detection and Localization},
  author    = {Bergmann, Paul and Jin, Xin and Sattlegger, David and Steger, Carsten},
  booktitle = {VISIGRAPP},
  year      = {2022}
}
```

---

## License

本项目仅用于学术研究与教学复现，**不提供任何商业许可保证**。

- 代码：MIT License。
- 数据集：MVTec 3D-AD 数据集的版权与使用条款归 [MVTec Software GmbH](https://www.mvtec.com/) 所有，请遵循其原始 License。

---

## Acknowledgments

- 感谢 PyTorch、TorchVision、SciPy、scikit-learn 等开源社区的长期贡献；
- 感谢 PointNet++ 作者团队为点云深度学习提供了奠基性工作；
- 感谢 MVTec Software GmbH 公开 MVTec 3D-AD 数据集，为工业异常检测领域提供了高质量的研究素材。

