# GRM 农业数据评估

在本项目根目录运行 `evaluate_agri_synthetic.py`（合成数据）和
`evaluate_agri_real.py`（真实农业数据）。将这两个文件及
`agri_eval_common.py`、`agri_eval_adapter.py` 一起上传到 HPC 的 GRM 项目根目录，
使用原 GRM 环境及 `grm_matcher/` 源码。

## 运行

下面在已分配 GPU 的计算节点运行。`/path/to/...` 必须替换为实际 HPC 数据路径；
合成根目录应包含 test 清单及图像，真实根目录应包含 pair_list.json、annotations.json 和图像。
权重路径按实际训练输出调整，支持包含 config 和 model 的 best_agri.pt / best_imc.pt。

```bash
python -u evaluate_agri_synthetic.py --checkpoint training_runs/grm_scratch_agri/best_agri.pt --data-root /path/to/homography_dataset --split test --output evaluation_outputs/grm_synthetic_test
python -u evaluate_agri_real.py --checkpoint training_runs/grm_scratch_agri/best_agri.pt --data-root /path/to/test --output evaluation_outputs/grm_real_test
```

本地数据路径默认分别为 `dataset/homography_dataset` 和
`dataset/test`，本地运行可省略 --data-root。默认设备 cuda，可指定 --device cpu。
先加 `--limit 2` 并使用独立输出目录试跑。每次输出目录必须为空或不存在；不支持断点续跑。
无需权重即可用 `--check-data` 检查数据，同样要指定新的输出目录。

## 与四个基线一致的协议

- 公共评估实现与 EDM、EfficientLoFTR、LoFTR、RoMaV2 的当前版本相同。
- 默认输入 512×512、FP32，RGB 归一化遵循 checkpoint 配置。
- 保留 checkpoint 的候选采样及重排配置；不使用训练时强制采样覆盖或真值锚点。
- GRM 输出先取 valid_mask 有效点，置信度为 p_match，不额外按 Global/Local 角色筛选。
  按置信度排序最多保留 --max-matches 2048 个，实际数量受模型 max_source_queries 限制。
  如需固定 256 点预算对比，应为所有模型都设置 --max-matches 256；这只是上限，不保证每对产生256点。
- 预测点使用 USAC_MAGSAC 估计 H，阈值为3个输入像素。半像素缩放变换恢复原图坐标，所有误差按原图像素计算。
- 合成 H、有效区域掩码、真实人工点仅用于打分，不传给模型或 H 估计。
- 合成集统计匹配精度、覆盖率、角点误差及 AUC；真实集统计人工点重投影误差、PCK 和 AUC。
  估计失败的有标注图像对仍计入失败率及准确率分母。
- 真实 pair_list 中所有图像对均推理并保存匹配结果，未标注对仅不参与几何准确率。
  默认尝试保存全部拼接图；没有有效 H 的图像对保留失败记录，不能生成有效拼接图。
  合成默认仅保存前10对拼接预览，可用 --save-visuals -1 全量保存。

## 输出和计时

`settings.json` 记录配置、名单、权重哈希及 GRM 配置；`summary.json` 记录汇总。
逐对 CSV/JSONL 记录成功失败状态、匹配数、误差与耗时。
`matches/` 保存每对 NPZ，`stitches/` 保存能成功生成的拼接预览。

计时包括 matching_seconds（输入张量转换、推理、结果回传）、geometry_seconds（筛选及H估计）、
stitch_save_seconds（拼接及保存）、pair_total_seconds（包含读图、评分和保存的逐对总时间）。
GPU 计时前后同步；首次预热单独记录、不计入逐对耗时。
pipeline_seconds 已包含 matching_seconds 和 geometry_seconds，不能重复相加。
公平测速应使用同一 GPU、环境、输入尺寸和输出设置，避免多个任务竞争 GPU。

## 已验证范围

公共协议9项测试通过；使用本地短训练测试权重，合成和真实数据各2对完成CUDA推理、评分及结果落盘。
该测试权重产生零有效匹配，脚本正确记录H估计失败并跳过拼接；它不是HPC最终训练权重，
测试结果不代表最终模型精度。正式结论需要使用HPC训练完成的权重进行完整评估。

## 低重叠率采样实验（新增，可选）

默认 `--grm-sampling-mode checkpoint`，遵循权重配置；旧权重仍使用原采样策略。
新策略不改参数形状，可直接读取现有 best_agri.pt，无需为了试验重新训练。

```bash
python -u evaluate_agri_synthetic.py --checkpoint training_runs/grm_scratch_agri/best_agri.pt --data-root /path/to/homography_dataset --split val --grm-sampling-mode overlap_aware --output evaluation_outputs/grm_overlap_val
python -u evaluate_agri_real.py --checkpoint training_runs/grm_scratch_agri/best_agri.pt --data-root /path/to/test --grm-sampling-mode overlap_aware --output evaluation_outputs/grm_overlap_real
```

先在验证集与同权重 `--grm-sampling-mode adaptive_density` 对比并确定阈值，再运行真实测试集。
每个策略使用新的输出目录。所有覆盖选项和生效配置记录在 settings.json 的 arguments、runtime.config 和 sampling_overrides。

新策略先构造“预测的可靠支持区域”，它不是已知真值重叠掩码：

1. 源点需通过 matchability 下限，以及至少一个有足够概率质量的双向 Top-K 候选一致性检查。没有把多候选退化为严格双向 Top-1。
2. 保留旧 fixed_score_threshold 绝对下限，另要求可靠性分数达到合格点最高分的一定比例。不再用全图分位数保留固定比例的点。
3. 仅在支持区域内选点，以可靠性分数为主、归一化空间距离为有界加分。距离尺度来自支持区域，而不是整幅图。
4. 不提供网格最低配额、不强制达到 minimum_query_count，不因原区域上限阻止可靠窄带内选点。点数可低于几何求解要求，甚至为零；空槽仍为 valid_mask=False。

默认试验值为 matchability >= 0.2、相对分数 >= 0.1，可分别通过
`--grm-support-min-matchability` 和 `--grm-support-relative-score` 调整，范围均为[0,1]。
相对分数也用于排除仅有极小概率的“碰巧双向一致”候选。以上只是初始值，不是已校准最优值。
新模式忽略旧网格配额、分位数回退及局部NMS配置，保留最大点数、动态质量预算、绝对阈值和 coverage_weight。

模块级输出 diagnostics 新增 reciprocal_query_count、supported_query_count、supported_query_fraction，
表示预测支持，不可当成真值重叠率；逐对评估仍记录 matches、ransac_inliers、H误差及失败状态。

当前模型仍使用固定长度张量并对无效位置填充，因此实际点数减少不意味着推理时间会同比下降。
这个版本也不增加H质量拒绝器。少量/共线内点造成的退化H仍需另做质量诊断。

训练时不建议直接从随机初始化启用这些硬门控，否则初期预测不可靠可能导致无训练对应点。
本次保留现有训练策略，先做同权重推理消融。后续训练需设计监督采样到预测采样的过渡，并用部署采样进行验证。

建议按真值重叠率0–10%、10–30%、30–50%、50%以上分组报告匹配精度、匹配数量、重叠区内采样比例、
H-AUC和失败率；真值掩码仅用于分组/评分。固定随机种子和图像名单，不能通过删除低重叠失败对提高指标。

机制测试验证窄带支持、少点/零点、不可靠一对多竞争、掩码及非有限值、批次独立性和旧配置兼容。
本地短训练权重的CUDA端到端试跑成功，但仍无有效匹配，因此没有证明新策略对最终训练模型有精度收益。
