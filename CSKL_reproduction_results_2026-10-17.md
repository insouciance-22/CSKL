# CSKL 复现实验记录

核查日期：2026-10-17。新增核查范围：2026-10-15、2026-10-16，保留此前已核实结果。

本记录只整理现有实验，不修改训练代码、旧配置、旧指标或 checkpoint，不启动训练。它是结果索引，不是完整代码备份。

## 统计口径

- 插补：METR-LA block，最终窗口预测按 `mean` 聚合后的 MAE / RMSE。
- 预测：PEMS-BAY，验证集选出的最佳 checkpoint 的 `test_metrics.json` 中 `overall` MAE / RMSE。
- Kriging：PEMS-BAY，日志中反归一化后、指定目标节点有效位置上的 MAE / RMSE。
- 论文来源：`F:/py_project/CSKL/14101_Compositional_Spatial_Ke.pdf`，第 7 页表 1、表 2，第 8 页表 3。
- 论文数值是多 seed 的均值和波动；本地除特别说明外为单次结果。不能把与论文均值接近等同于复现了统计增益。
- 本地汇总中的标准差采用样本标准差（ddof=1）；不假定与论文的标准差定义完全相同。
- Kriging 项目本次实际位置为 `F:/py_project/Kriging-Library`，已不在 CSKL 根目录内。

## 论文与本地实验对应表

数值顺序均为 MAE / RMSE。论文列保留原始均值；其波动见原论文。

| 任务与数据集 | 模型/版本 | 论文原始模型 | 论文对应 CSKL | 本地结果 | 次数/状态 | 判断 |
|---|---|---|---|---|---|---|
| 插补 METR-LA block | GRIN + 加法 CSKL | 2.008 / 3.539 | 1.844 / 3.249 | 1.8412 / 3.2484 | 2020、2021、2022 三 seeds 均值 | 接近，参数已更新 |
| 插补 METR-LA block | GRIN + CSKL(Aimp) | 2.008 / 3.539 | 1.837 / 3.240 | 1.8361 / 3.2433 | 2020、2021 两 seeds 均值 | 接近；2022 旧运行缺最终结果 |
| 插补 METR-LA block | BiMPGRUNet + 加法 CSKL | 论文 MPGRU 2.247 / 3.948 | 2.118 / 3.717 | 2.1051 / 3.6986 | 10-16 22:54，seed 2020 | 数值接近；BiMPGRU 与论文 MPGRU 的入口对应仍需确认 |
| 插补 METR-LA block | BiMPGRUNet + CSKL(Aimp) | 论文 MPGRU 2.247 / 3.948 | 2.089 / 3.652 | 2.1071 / 3.6636 | 10-16 22:53，seed 2020 | 数值接近；同样存在 backbone 命名映射问题 |
| 预测 PEMS-BAY | DCRNN + 三核加法 | 1.591 / 3.690 | 1.570 / 3.634 | 1.5703 / 3.6225 | 10-13，单次 | 接近，保留 |
| 预测 PEMS-BAY | STGCN + 三核加法 | 1.711 / 3.829 | 1.735 / 3.896 | 1.7325 / 3.9027 | 10-16，单次 | 接近论文 CSKL；论文在此设置本就弱于原始 STGCN |
| 预测 PEMS-BAY | GWNet + 三核加法 | 1.578 / 3.610 | 1.567 / 3.606 | 1.5968 / 3.6537 | 10-16，单次 | MAE 比论文 CSKL 高约 1.90%，优先查配置与核退化 |
| 预测 PEMS-BAY | DGCRN + 三核加法 | 1.578 / 3.672 | 1.559 / 3.580 | 1.5711 / 3.6645 | 10-15 启动、10-16 完成，单次 | 接近论文原始模型；MAE/RMSE 比 CSKL 高约 0.78%/2.36% |
| Kriging PEMS-BAY | IGNNK 原始模型 | 4.042 / 6.553 | 不适用 | 4.0211 / 6.4949 | 10-16，seed 2020 | 已补齐本地对照 |
| Kriging PEMS-BAY | IGNNK + CSKL(Akrig) | 4.042 / 6.553 | 3.992 / 6.477 | 4.0344 / 6.4441 | 10-13，seed 2020 | 对本地原始模型：MAE 略差，RMSE 略好 |
| Kriging PEMS-BAY | KITS 原始模型 | 3.968 / 6.711 | 不适用 | 4.1699 / 6.8535 | 10-16，seed 2020 | 原始模型本身也比论文均值差 |
| Kriging PEMS-BAY | KITS + CSKL(Akrig) | 3.968 / 6.711 | 3.914 / 6.686 | 3.9660 / 6.5602 | 10-14，seed 2020 | 对本地原始模型：MAE 降 4.89%，RMSE 降 4.28% |

### GRIN 分 seed 明细

| 版本 | 运行目录（位于 grin-main/logs/la_block/grin） | seed | MAE | RMSE |
|---|---|---|---|---|
| 加法 | 2026-10-15_11-09-14_2020 | 2020 | 1.8327737765 | 3.2438255803 |
| 加法 | 2026-10-15_14-33-29_2021 | 2021 | 1.8516302575 | 3.2497640446 |
| 加法 | 2026-10-15_16-45-08_2022 | 2022 | 1.8391867062 | 3.2516712129 |
| Aimp | 2026-10-13_11-01-47_2020 | 2020 | 1.8327388649 | 3.2498000148 |
| Aimp | 2026-10-13_13-04-03_2021 | 2021 | 1.8395445364 | 3.2368445221 |

加法三 seeds：MAE 1.8412 ± 0.0096，RMSE 3.2484 ± 0.0041。

Aimp 两 seeds：MAE 1.8361 ± 0.0048，RMSE 3.2433 ± 0.0092。两版本 seed 数不同，不据此判断优劣；不需要为了补表盲目重跑已完成实验。

## BiMPGRUNet 导出异常与补录

应用没有连接到当时的终端，无法读取 IDE 控制台历史。已读取四次运行的 TensorBoard 事件和 checkpoint，但 TensorBoard 的 `test_mae` / `test_mse` 使用窗口级口径，不等于控制台最终 mean 聚合指标。

代码先输出最终聚合指标，再访问 `filler.model.kernel_type`，再写 `metrics.json`。因此该 AttributeError 发生在训练和最终指标计算之后，不代表训练失败。依据：`grin-main/scripts/run_imputation.py:339`、`:347`、`:362`。

以下两个最终结果按用户提供的控制台输出原样补录，不伪造原运行的 `metrics.json`：

| 来源 | MAE | RMSE | MSE | MRE | MAPE | 归属状态 |
|---|---|---|---|---|---|---|
| 用户提供的第一组控制台输出 | 2.1263 | 3.7284 | 13.9009 | 0.0368 | 0.0508 | 按输出顺序推测是加法版；原控制台缺运行目录，待确认 |
| 用户提供的第二组控制台输出 | 2.0687 | 3.6396 | 13.2464 | 0.0358 | 0.0491 | 按输出顺序推测是 Aimp；原控制台缺运行目录，待确认 |

两次无指标文件的目录，核版本由 checkpoint 参数结构直接确认：

| 运行目录（位于 grin-main/logs/la_block/bimpgru） | seed | checkpoint 结构 | 最终聚合指标来源 |
|---|---|---|---|
| 2026-10-16_14-33-59_2020 | 2020 | 三组 alpha + logits，加法版；核参数已更新 | 上述第一组为暂定对应，未自动恢复 |
| 2026-10-16_14-36-23_2020 | 2020 | 三组 alpha + w_1/w_2/w_3，Aimp；核参数已更新 | 上述第二组为暂定对应，未自动恢复 |

后来还有两次完整导出，主表使用它们，不与早期同 seed 运行平均，也不挑选更低值替换：

| 运行目录（同上） | 版本 | MAE | RMSE | MSE | MRE | MAPE | kernel theta |
|---|---|---|---|---|---|---|---|
| 2026-10-16_22-53-10_2020 | Aimp | 2.1071482290 | 3.6635794820 | 13.4218146211 | 0.0364874910 | 0.0496715918 | 0.1 |
| 2026-10-16_22-54-00_2020 | 加法 | 2.1051372376 | 3.6986420147 | 13.6799527531 | 0.0364526686 | 0.0500722547 | 0.1 |

同 seed 多次结果不同，需要比较实际核 threshold、运行时源码和随机性；当前记录不足以确定原因。早期 checkpoint/hparams 未完整保存核超参数。

## 新确认的问题与结论

1. Kriging 两对实验的 test_node / val_node 文件内容完全一致，均为 seed 2020，日志中的训练主要参数也一致。它们是比跨论文均值更直接的本地对照，但模型切换还可能影响随机数消耗，且不是多 seed 显著性证据。
2. KITS 的本地 MAE 实际改善 4.89%。此前仅比较论文原始 KITS 的 3.968 容易误判；本地原始 KITS 是 4.1699。无需继续把 KITS 当作“CSKL 完全没有收益”。
3. IGNNK MAE 从 4.0211 变为 4.0344，约恶化 0.33%；RMSE 从 6.4949 降到 6.4441，约改善 0.78%。此设置下 MAE 收益尚未复现。
4. Kriging 两训练器仍使用测试集损失选 checkpoint：`F:/py_project/Kriging-Library/exp/exp_normal.py:90`、`exp/exp_kernel.py:102`。这些是历史协议结果，不可直接作为独立测试集上的新论文证据。
5. GWNet、DGCRN、STGCN 新 checkpoint 均确有三组核尺度和 mixing logits，参数发生更新，不能解释为“忘记加入三核”。
6. DGCRN 最佳 checkpoint 的 Gaussian 全部尺度绝对值小于 1e-6；按当前公式与 PEMS-BAY 距离在内存中重建，Gaussian 非对角元素全部为 0。GWNet Gaussian 非对角非零仅 35/105300。此为已观测现象；是否由 weight decay、ReLU、尺度学习及数据共同造成，尚未确定。不能把总图也说成全零，其他核及模型自身图仍可能起作用。
7. BiMPGRUNet 两个完整结果的 threshold 为 0.1，而 GRIN 为 0.0。各自与师兄历史设置是否相同需要确认，不能为统一形式直接修改。
8. 论文 MPGRU 与实际 BiMPGRUNet 的映射仍需师兄确认；数值接近不单独证明 backbone 一致。当前单向 MPGRUNet 仍引用未导入的 Kernel_Layer_th 且缺 num_nodes，不宜启动。
9. 不足以宣称完整统计复现或所有 backbone 均获益；已足以停止大面积复现、保存当前阶段，并围绕 GRIN 开展后续方法验证。

## 下一步（尚未执行）

1. 保存当前源码、有效配置、核版本、checkpoint 路径、依赖和结果清单的备份；Kriging 必须包含工作区外的 `F:/py_project/Kriging-Library`。仅保留数据集配置副本不足以还原当时模型源码。
2. 确认用户两段输出与早期 BiMPGRUNet 目录的归属，随后以 checkpoint 仅评估恢复最终聚合指标，无需训练。未确认前保留“用户提供、暂定对应”标签。
3. 下一轮仅在获得修改授权后修复核参数选择与指标导出的兼容性，并把实际 kernel_type/theta 等写入运行配置，避免再次依赖手改类名。
4. 优先静态比较 GWNet、DGCRN 与师兄对应运行的配置及源码；检查 Gaussian 退化和 optimizer 的参数分组/正则化。不要先增加 epoch 或扫学习率，也不要为追论文数值反复选择测试结果。
5. 面向新工作先明确独立验证协议，再在需要的 backbone 上配对重跑 baseline/CSKL。Kriging 历史选模偏差无法通过重新计算旧 checkpoint 消除。
6. 不必补齐全部模型、数据集的五 seeds 才开始优化。GRIN 的加法和 Aimp 已可作起点；先验证新方法，再为关键对照补配对 seeds。GRIN 真正学习的单 Gaussian、固定图对照若无同协议有效记录，仍应在关键实验阶段补齐。

## 结果路径索引

- GRIN：`F:/py_project/CSKL/grin-main/logs/la_block/grin/`，运行名见上表。
- BiMPGRUNet：`F:/py_project/CSKL/grin-main/logs/la_block/bimpgru/`，运行名见上表。
- DCRNN：`F:/py_project/CSKL/BasicTS-master/checkpoints/DCRNN/PEMS-BAY_200_12_12/20261013_171025_216_pid2124/5f37b99ab4cabe8f3dc6851a37444679/test_metrics.json`。
- STGCN 三核：`F:/py_project/CSKL/BasicTS-master/checkpoints/STGCNChebGraphConv/PEMS-BAY_100_12_12/20261016_104115_239_pid41812/69d554107b83ab77b07587fe1ee98a2d/test_metrics.json`。
- GWNet：`F:/py_project/CSKL/BasicTS-master/checkpoints/GraphWaveNet/PEMS-BAY_200_12_12/20261016_120533_286_pid33456/6cf8f4402e89e8775d8fda5c5886482e/test_metrics.json`。
- DGCRN：`F:/py_project/CSKL/BasicTS-master/checkpoints/DGCRN/PEMS-BAY_100_12_12/20261015_091318_522_pid25104/6225182df0b8ef360fb80079d035bdf8/test_metrics.json`。
- 原始 KITS：`F:/py_project/Kriging-Library/logs/pems_bay/KITS/2026-10-16_10-42-32/log.log`。
- KITS Akrig：`F:/py_project/Kriging-Library/logs/pems_bay/KITS_kernel/2026-10-14_21-01-27/log.log`。
- 原始 IGNNK：`F:/py_project/Kriging-Library/logs/pems_bay/IGNNK/2026-10-16_12-11-58/log.log`。
- IGNNK Akrig：`F:/py_project/Kriging-Library/logs/pems_bay/IGNNK_kernel/2026-10-13_17-25-13/log.log`。

## 排除项

- 10-12 GRIN 单核/两核：checkpoint 核参数未更新，不计作成功学习的消融。
- 10-14 STGCN：单 Gaussian 结果 1.6816 / 3.7828，不混入三核 CSKL。
- GRIN Aimp 2026-10-13_14-32-40_2022：目前缺最终指标，不计入完成次数。
- MPGRUNet 2026-10-15_09-36-01_2020：构造失败，无有效测试结果。
