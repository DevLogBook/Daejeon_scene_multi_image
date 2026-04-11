# AgriTPSStitcher

面向农业无人机场景的密集匹配与几何拼接研究原型系统。

本项目围绕“低纹理、重复纹理、视角扰动和局部非刚性形变条件下的图像拼接”展开，构建了一套以教师蒸馏为核心、以稠密几何场预测为目标的端到端训练与推理流程。系统以成对图像为输入，首先通过轻量级特征主干与匹配模块估计粗到细的对应关系，再结合全局单应与局部 TPS 形变场，完成重叠区域的高精度对齐与全景生成。

该仓库更接近科研代码仓而非工程化 SDK，重点在于方法验证、训练策略迭代和可视化诊断。

## 1. 研究背景

农业航拍与地面巡视数据通常存在以下难点：

- 作物行列与土壤纹理高度重复，局部区域容易产生多峰匹配歧义。
- 采集高度、俯仰角和平台姿态变化较大，单一全局模型难以充分解释局部变形。
- 光照、阴影、云层遮挡与曝光变化会削弱基于光度一致性的约束。
- 高分辨率图像直接进行 dense matching 代价较高，且训练稳定性差。

针对这些问题，本项目采用“教师模型监督 + 学生模型蒸馏 + 分阶段优化”的技术路线，在保证可训练性的同时提升局部几何细化能力。

## 2. 方法概述

项目核心网络为 `AgriTPSStitcher`，其主要组成如下：

1. `MobileViT + BiFPN` 轻量级特征主干  
   从多尺度特征中提取 coarse/fine 表征，为后续匹配与局部几何建模提供上下文。

2. 多峰感知的密集匹配模块  
   在 coarse 网格上构建相似度矩阵，通过最优传输、Top-K 候选筛选和局部几何一致性验证，缓解重复纹理下的错误峰值问题。

3. 全局几何与局部形变联合建模  
   系统同时显式估计全局单应矩阵和局部 TPS / dense warp，从而兼顾大尺度视角变化与重叠区域的细粒度对齐。

4. 教师蒸馏与缓存式训练  
   使用 `RoMaV2` 作为 teacher，预计算 `warp / confidence / feature` 缓存，以降低训练阶段的显存压力和在线教师推理开销。

5. 三阶段训练策略  
   按照 warm-up、aggregator 激活、end-to-end finetune 的顺序逐步放开参数与损失项，提升训练稳定性。

## 3. 主要特点

- 面向重复纹理场景设计的多峰匹配与几何验证机制。
- 支持 pair file 直读与 teacher cache 两种训练模式。
- 支持单卡训练与 DDP 分布式训练。
- 支持离线裁剪、成对样本生成、teacher 缓存预计算、诊断可视化和最终拼接推理。
- 推理阶段在全分辨率下融合全局单应和局部 TPS 变形，并结合 Voronoi seam 与 multi-band blending 生成全景图。

## 4. 仓库结构

```text
.
├── dataset/
│   ├── dataset.py                 # 数据集定义：pair file、cache dataset、HDF5 dataset
│   ├── generate_pairs.py          # 从 input1/input2 目录生成配对列表
│   ├── generate_pairs_in_file.py  # 从单目录顺序图像生成相邻配对
│   ├── offline_random_crop.py     # 基于 SIFT 的严格无 padding 离线裁剪
│   └── clean_and_mv.py            # 清理、抽样并迁移配对数据
├── dense_match/
│   ├── backbone.py                # MobileViT + BiFPN 主干
│   ├── network.py                 # 核心网络：匹配、几何估计、TPS 聚合
│   ├── flow_to_tps.py             # 稠密流到 TPS 控制点的映射与估计
│   ├── refine.py                  # warp refinement 与几何/光度损失
│   ├── utils.py                   # 训练阶段定义、优化器与教师辅助函数
│   ├── precompute_teacher.py      # 教师缓存预计算
│   ├── train.py                   # 单卡训练入口
│   ├── train_ddp.py               # 多卡 DDP 训练入口
│   ├── validate_matcher.py        # 定量诊断与可视化验证
│   └── inference.py               # 全景拼接推理入口
├── warp/
│   ├── filter.py                  # 传统几何过滤相关脚本
│   └── solution.py                # 传统模块/工具脚本
├── backbones/                     # 预训练 backbone 权重目录
├── checkpoints/                   # 训练得到的权重
├── data/                          # 原始数据目录
└── README.md
```

## 5. 环境配置

建议环境：

- Python 3.10 或 3.11
- CUDA 11.8+（推荐）
- PyTorch 2.x

建议使用 `conda` 或 `venv` 创建独立环境，然后安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy matplotlib pillow tqdm
pip install albumentations timm safetensors tensorboard h5py
```

### 外部依赖

1. `RoMaV2`  
   当前代码默认从 `dense_match/RoMaV2/src` 导入 `romav2`。请将对应代码或子模块放置到：

```text
dense_match/RoMaV2/
```

2. `MobileViT` 本地主干权重  
   `dense_match/backbone.py` 默认尝试读取：

```text
backbones/model.safetensors
```

若该文件不存在，程序会给出 warning，并可能以未加载本地权重的方式继续运行。

## 6. 数据组织与配对文件

### 6.1 数据目录

推荐采用如下组织方式：

```text
your_dataset/
├── input1/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── input2/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

### 6.2 配对文件格式

训练脚本支持文本形式的图像对列表，每行两个路径，中间以空格或逗号分隔。例如：

```text
/abs/path/to/input1/0001.jpg /abs/path/to/input2/0001.jpg
/abs/path/to/input1/0002.jpg /abs/path/to/input2/0002.jpg
```

也支持使用相对路径。以 `#` 开头的行为注释行。

### 6.3 生成配对列表

如果数据已经按 `input1 / input2` 对齐，可使用：

```bash
python dataset/generate_pairs.py --pairs-folder /path/to/your_dataset
```

如果单目录内图像按时间顺序排列，希望构造相邻帧配对，可使用：

```bash
python dataset/generate_pairs_in_file.py --pairs-folder /path/to/sequence_folder
```

## 7. 数据预处理

### 7.1 离线裁剪增强

`dataset/offline_random_crop.py` 实现了基于 SIFT 对应点的严格无 padding 裁剪策略，适合在高分辨率场景下构造具有重叠区域且带轻微位姿扰动的训练样本。

当前脚本底部带有本地路径示例，正式使用前请先根据自己的数据路径修改入口参数。

### 7.2 数据清理与抽样

`dataset/clean_and_mv.py` 用于：

- 清理 `input1` 与 `input2` 中未对齐的孤立图像
- 从有效配对中随机抽样
- 将抽中的样本迁移到新的数据目录
- 同步输出新的 pair list

这对于构建中小规模实验集或消融实验子集很有用。

## 8. 教师缓存预计算

项目默认推荐先用 teacher 预计算缓存，再训练学生网络。这样做有三个直接收益：

- 避免训练阶段重复运行 teacher 网络
- 降低总训练时间与显存波动
- 使训练过程更稳定、更易复现

### 8.1 单卡预计算

```bash
python dense_match/precompute_teacher.py \
  --pairs-file /path/to/pairs.txt \
  --save-dir teacher_cache \
  --teacher-setting precise \
  --teacher-grid-size 32 \
  --variations 2
```

### 8.2 多卡预计算

```bash
torchrun --nproc_per_node=4 dense_match/precompute_teacher.py \
  --pairs-file /path/to/pairs.txt \
  --save-dir teacher_cache \
  --teacher-setting precise \
  --teacher-grid-size 32 \
  --variations 2
```

缓存目录下通常会生成若干 `.pt` 文件，内部包含：

- `img_a`, `img_b`
- `warp_AB`
- `confidence_AB`
- `feat_A`, `feat_B`

## 9. 训练

### 9.1 单卡训练

```bash
python dense_match/train.py \
  --cache-dir teacher_cache \
  --save-dir checkpoints/agrimatch_v2 \
  --log-dir runs/agrimatch_v2 \
  --batch-size 4 \
  --epochs 65 \
  --num-workers 4 \
  --amp
```

如果尚未生成缓存，也可以传入 `--pairs-file` 直接读取图像对；但当前训练脚本已经显式关闭在线 teacher 推理，因此实际推荐始终使用 `--cache-dir`。

### 9.2 多卡 DDP 训练

```bash
torchrun --nproc_per_node=4 dense_match/train_ddp.py \
  --cache-dir teacher_cache \
  --save-dir checkpoints/agrimatch_ddp \
  --log-dir runs/agrimatch_ddp \
  --batch-size 4 \
  --epochs 65 \
  --num-workers 4 \
  --amp
```

### 9.3 训练阶段设计

训练流程在 `dense_match/utils.py` 中定义为三阶段：

1. `SUPERVISED_WARMUP`  
   冻结 aggregator，先让 matcher 与 backbone 在蒸馏监督下收敛。

2. `AGGREGATOR_ACTIVATION`  
   激活 TPS aggregator，逐步引入 fold、photo 和 smoothness 等损失。

3. `END_TO_END_FINETUNE`  
   端到端联合微调，使用更保守的学习率与余弦退火策略完成收敛。

### 9.4 主要训练参数

常用参数包括：

- `--cache-dir`：teacher 缓存目录或 HDF5 缓存文件
- `--save-dir`：checkpoint 输出目录
- `--log-dir`：TensorBoard 日志目录
- `--teacher-setting`：teacher 模式，默认 `precise`
- `--teacher-grid-size`：teacher 特征网格大小，默认 `32`
- `--batch-size`：每步 batch 大小
- `--accum-steps`：梯度累积步数
- `--amp`：启用混合精度
- `--resume`：从已有 checkpoint 恢复训练

## 10. 推理与全景生成

推理脚本会读取两张图像和训练好的学生模型权重，输出最终全景图。其流程包括：

- 低分辨率输入下的网络前向预测
- 全局单应矩阵估计
- 局部 TPS dense grid 注入
- 全分辨率 remap
- Voronoi seam mask 生成
- multi-band blending 融合

示例命令：

```bash
python dense_match/inference.py \
  --img-a /path/to/image_a.jpg \
  --img-b /path/to/image_b.jpg \
  --ckpt checkpoints/agrimatch_v2/best.pt \
  --out output/panorama.jpg
```

默认会输出：

- `output/panorama.jpg`：最终拼接结果
- `output/panorama_overlap.jpg`：用于定性观察的重叠可视化

## 11. 验证与可视化诊断

### 11.1 定量诊断

`dense_match/validate_matcher.py` 可用于：

- 计算重叠区域上的 PSNR / SSIM / L1
- 分析 TPS 对 coarse warp 的修正幅度
- 统计 overlap ratio、confidence、gate 等中间量
- 输出可视化诊断图

### 11.2 匹配歧义可视化

`vis_feng.py` 主要用于研究多峰匹配现象，例如：

- 单点查询的相似度热图
- 多峰位置分布
- 匹配熵与 peak ratio 分析

这类工具适合做方法分析、错误案例整理和论文图表生成。

## 12. 结果文件

训练过程中常见输出包括：

- `checkpoints/.../last.pt`
- `checkpoints/.../best.pt`
- `runs/...` TensorBoard 日志
- `output/...` 推理图像

checkpoint 默认保存内容包括：

- `student` 模型参数
- `optimizer` / `scheduler` 状态
- `epoch` / `step`
- `train_loss_ema`
- `val_loss` / `best_val_loss`

## 13. 已知注意事项

1. 本仓库部分脚本底部仍保留本地路径示例，运行前建议逐项检查。
2. 若在 Windows 上恢复 Linux 环境下保存的路径对象，部分脚本已通过 `pathlib.PosixPath = pathlib.WindowsPath` 做兼容处理。
3. 训练主流程当前推荐完全依赖 teacher cache；若不提供缓存，`train.py` 会在进入在线 teacher 分支前直接报错。
4. `RoMaV2` 和 backbone 本地权重不随仓库自动分发，需要用户自行准备。
5. 推理脚本假设 checkpoint 内存在 `student` 键，并与当前网络结构兼容。

## 14. 推荐实验流程

对于首次复现实验，建议遵循以下顺序：

1. 整理原始数据并生成 pair file
2. 根据需要执行离线裁剪或抽样
3. 使用 `precompute_teacher.py` 生成 teacher cache
4. 使用 `train.py` 或 `train_ddp.py` 训练学生模型
5. 使用 `validate_matcher.py` 做定量诊断与可视化
6. 使用 `inference.py` 生成最终拼接结果

## 15. 引用与致谢

如果该代码对你的研究有帮助，建议在论文或报告中说明以下技术来源：

- RoMaV2 教师模型
- MobileViT 轻量级视觉主干
- TPS 形变建模与传统图像融合技术
