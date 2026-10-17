# CSKL 代码问题与待确认修改方案

日期：2026-10-17。状态：仅分析和方案，不包含训练代码修改。

本轮只新增本说明文件。未修改源码、配置、原指标、checkpoint；未运行训练、反向传播或 optimizer step。核矩阵诊断由已有 checkpoint 和距离在内存中重建。

## 1. 建议分成两批，不把实现修复与训练改进混在一起

第一批：修正模型/核选择、初始化接口、结果导出、有效配置和版本记录。保留原公式、各模型原有效 threshold、自环、模型结构和 optimizer 策略，用现有 checkpoint 验证兼容性。

第二批：针对 GWNet/DGCRN 做有明确标签的核正则化对照。先比较参数分组，后考虑正值参数化。第二批会改变训练策略，属于改进，不能宣称只是恢复原论文配置。

## 2. 当前确定的问题及修改位置

| 优先级 | 文件与位置 | 问题 | 建议修改 |
|---|---|---|---|
| P0 | `grin-main/scripts/run_imputation.py:128` | 读取 YAML 之前，按 CLI 默认模型 `brits` 注册模型参数；稍后 YAML 又可将模型改成 bimpgru/grin，导致注册模型与实际模型不同 | 先预读取 config 和显式 model-name，确定最终模型，再调用该模型参数注册；完整解析后再构建模型 |
| P0 | 同文件 `:134` | YAML 在 CLI 后无条件覆盖，命令行指定 kernel/seed 可能不生效 | 固定优先级：显式 CLI > YAML > 类/程序默认；只有用户真正传入的 CLI 选项覆盖 YAML，不能让 argparse 默认值误覆盖 YAML |
| P0 | `grin-main/lib/nn/models/grin.py:200` | GRIN 的 `add_model_specific_args` 实际缩进在 SAAdjacency 内；GRINet 没有该方法 | 将该方法放回 GRINet；不移动或修改 attention forward 的计算行为 |
| P0 | `grin-main/lib/nn/models/mpgru.py:31` | 当前单向 MPGRUNet 引用未导入的 Kernel_Layer_th，且未传 num_nodes；原 TypeError 修复尚不完整 | 按明确配置实例化实际核；从 adj.shape[0] 得节点数并校验与输入一致，杜绝默认 0 节点核 |
| P0 | 同文件 `:104` | BiMPGRUNet 的 kernel_type、核类、alpha 等硬编码，不能由配置可靠选择 | 增加与 GRIN 一致的核选项，配置值与实际实例相互校验；保留 BiMPGRUNet 现有效 theta=0.1，GRIN 保留 0.0 |
| P0 | `grin-main/scripts/run_imputation.py:345` | 检测有 kernel_layer 后，假定所有模型都有 model.kernel_type；元数据失败导致全部最终指标未写盘 | 独立保存已计算指标，再保存实际核元数据；核名称从真实模块获得，配置与实际冲突在训练前失败；可选元数据缺失不能丢弃指标 |
| P1 | 同文件 `:164`、`:236`；`lib/fillers/filler.py:42` | 保存的是 args 和过滤后的模型参数，未覆盖硬编码的核实际配置 | 保存 resolved_config 和实际模型元数据：类路径、核类、公式语义、theta、初始化、可学习范围、节点数、effective 模型参数等 |
| P1 | `BasicTS-master/baselines/{DCRNN,GWNet,STGCN,DGCRN}/arch/*.py` | 核类与 use_alpha 在模型源码里硬编码，数据集配置不完整描述实际运行 | 将核选项加入 MODEL.PARAM，并从配置构造现有类；保留当前有效图归一化和自环，不统一任务公式 |
| P1 | `BasicTS-master/basicts/utils/add_layers.py:86` | 同名/近似名类语义跨项目不同；theta 与 use_alpha 绑定；缺显式 kernel_alpha 元数据 | 增加实际元数据和显式选项；兼容旧配置的默认行为，不能悄悄改变旧单核 theta=0.1、三核学习版 theta=0.0 |
| P1 | `BasicTS-master/experiments/train.py:22` | seed 只用于种 RNG 和控制台打印，没有显式进入保存的 cfg；目录和 CFG._ 随机数不证明 seed | 将最终实际 seed 作为配置字段并写进每次运行和 checkpoint 元数据，保存命令行和依赖/源码版本 |
| P1 | `BasicTS-master/basicts/runners/base_tsf_runner.py:437` | 测试中有核数组导出，部分导出不受 save_results 控制；另有 runner 假定三核一定有 logits | 区分主要指标、可选核诊断、预测数组；仅评估恢复结果时写新输出目录，禁止默认覆盖旧运行；按真实核能力导出 |
| P2 | `BasicTS-master/basicts/runners/optim/builder.py:55` | kernel 参数与 backbone 一起应用 weight_decay | 仅第二批增加显式 kernel_weight_decay 参数分组；保留 legacy 策略，不能把训练变化混入第一批修复 |

## 3. kernel 选择的具体合同

沿用现有类，不引入跨项目统一大框架。Imputation 的 GRIN、MPGRUNet、BiMPGRUNet 使用同一个本地 registry，避免三份条件分支逐渐不一致；Forecasting 使用自己的本地 registry。

| 语义标记 | Imputation 类 | Forecasting 类 | 含义 |
|---|---|---|---|
| additive | Kernel_Layer | Kernel_Layer_ty | 三核加权相加 |
| imputation_path | Kernel_Layer_th | 不在本批新增 | 三个两两矩阵乘积，逐边 gating 后相加 |
| gaussian | Kernel_Layer_ada | Kernel_Layer（目前） | 单一 Gaussian |
| rq_matern_additive | Kernel_Layer_22 | Kernel_Layer_zuhe（目前） | RQ/Matérn 加法；有需求时再开放 |

必须明确并保存：kernel_type、actual_kernel_class、composition、kernel_alpha、kernel_theta、kernel_use_alpha、kernel_no_self_loop、num_nodes。kernel_type 与实例真实类不一致时，在第一步训练之前失败。

kernel_use_alpha=False 当前只表示不学习尺度，并不冻结 logits 或 gating。不能据此标成“原始固定图模型”。固定图 baseline 若要开放，应独立用 graph_mode=fixed，直接走已保存固定 adjacency，确认未参与的核参数不进 optimizer。该项应在实际需要 baseline 时单独实现，不通过冻结尺度冒充。

额外接口缺陷：`grin-main/lib/nn/layers/kernel_layers.py:166` 的 Kernel_Layer_22 在 use_alpha=False 时未创建 alpha2，但 forward 第 175 行仍读取它。若第一批开放该组合，必须先补齐静态分支并验证；否则明确不支持该设置，不能把它当成可运行选项。静态张量如需随模型迁移和进 checkpoint，应注册为 buffer。

constructor 需要显式接收这些配置，否则 filter_args（`grin-main/lib/utils/parser_utils.py:68`）会静默丢弃不在签名中的字段。对 kernel 相关未知项增加局部校验，不为本任务重写全项目参数系统。

检查正常选择至少涵盖三种模型、两种组合，以及有/无 CLI 覆盖；显式模型名与 YAML 不一致时应输出清晰的最终选择，或按预先规定的策略报错。每次运行开始时打印并保存实际模型类和核类，不依赖用户记忆手改了哪一行。

## 4. 指标导出和旧结果恢复

### 新运行

1. 模型创建后先收集/校验元数据，提前发现必需字段错误。
2. 最终 mean 聚合指标计算完成，立即保存五项主要指标：mae、rmse、mse、mre、mape。使用临时文件原子替换新运行的最终文件，避免半截 JSON。
3. 核诊断、训练时间等补充信息独立写出；可选导出失败时记录明确警告/状态，不把已经计算的指标丢掉，也不能静默吞掉结构错误。
4. 在结果中注明 evaluation_protocol=whole_series_mean，best_checkpoint、best_epoch、selection_metric=val_mae、有效 mask 数量。Lightning test_* 是窗口级指标，应标注为 window，不替代 whole-series 结果。
5. 保留既有 `mean` 指标结构，避免现有统计脚本失效。实际元数据应可在缺少 model.kernel_type 的旧模型上从真实 kernel_layer 类名恢复；其余缺失字段明确写 unknown/null，不猜默认值。

### 两次历史 BiMPGRUNet 导出失败

不重跑训练，不直接调用会创建训练目录并 fit 的 run_experiment。后续需单独的仅评估入口：严格加载指定 checkpoint；恢复原数据划分、mask seed、标准化、theta 和自环；只执行 eval/no_grad；输出到独立 recovery 目录，引用原 checkpoint。

早期运行没完整记录 theta，未确认前不能假设等于后来 0.1。用户提供的 2.1263/3.7284 与 2.0687/3.6396 保留为 user_console 来源，不伪造到旧 metrics.json，也不拿 TensorBoard 窗口级指标代替。

## 5. 配置保存的最小内容

- 用户请求配置：原 YAML / 数据集 Python 配置，以及实际命令行。
- 最终有效配置：合并优先级后的值、真实 class 全限定名、实际 kernel 参数、有效 embedding/u 设置、模型输入输出形状。
- 数据协议：train/val/test 时间边界、window/stride、mask seed、节点划分文件及哈希、距离文件/节点顺序哈希。只保存路径/哈希与必要的小 mask/索引，不复制大数据集。
- optimizer：真实 param groups，尤其 backbone/kernel 的 weight_decay、lr、参数名或清单摘要；冻结状态；scheduler 与 early-stop 设置。
- 版本：git revision + dirty 状态 + 关键源码副本/哈希；仅 revision 无法还原未提交修改。保存模型、核、入口、runner、metric、数据协议等关键文件。
- 实际 seed：包括 BasicTS CLI seed，而不是用目录随机字段推断。
- 环境：Python、torch、Lightning、CUDA/cuDNN 版本及确定性选项。没有启用确定性就如实记录，不在本轮顺手强制开启。
- checkpoint 标识与指标口径。加载旧 checkpoint 的 sigma 是普通 Python 属性而非 buffer，需要从原始有限距离准确重算并比对，不能对已经填大值的 dist 再算标准差。

## 6. MPGRU / BiMPGRU 映射：哪些确认了，哪些没有

已确认：

- `scripts/run_imputation.py:39` 将 mpgru 映射到 MPGRUNet，单向；`:41` 将 bimpgru 映射到 BiMPGRUNet，双向。不是两个可随意互换的入口。
- 师兄本地保留的 2025-01-31 METR-LA block baseline/adaptive 日志写的是 bimpgru。
- 旧 checkpoint 和 2026-10-16 四次新 checkpoint 都同时含 gcgru_fwd、gcgru_bwd；输出首层权重均为 (64,129,1,1)，未含 model.emb。这说明两批实际 backbone 一致为无节点 embedding 的双向网络。
- `config/bimpgru/la_block.yaml:18` 写 d_emb=8，但 constructor 使用 embedding_size；filter_args 丢掉 d_emb，实际 embedding_size=0。修参数名时若直接把 8 生效，会改变 backbone 和 checkpoint 形状。第一批应把 effective_embedding_size=0 明确保存/声明；开启 8 应另列新实验。
- 参考代码包 `F:/py_project/论文代码（仅代码）/project_CSKL/Imputation` 也同时保留两入口及不完整的单向 kernel 调用，不能把该包视为全部论文实验的完整可运行真值。

尚不能确认：论文表中“MPGRU”的每一行具体由哪个运行目录生成。论文命名本身、数值相近和旧 checkpoint 结构都不能替代作者的行到运行记录映射。

建议后续使用准确的实验名 BiMPGRU + CSKL（论文 MPGRU 对应待作者确认）；在得到作者确认前保留该说明。不要为了字面匹配论文，把已跑的双向实验改为单向重跑。

## 7. GWNet / DGCRN 核退化诊断

### 7.1 已确认的观测

以 PEMS-BAY 原始距离有限项标准差 s=3620.2991、当前三核公式、checkpoint 的尺度与 logits 重建。以下非对角数值指 kernel 原始分支，不是模型完整网络的所有图。

| 项目 | GWNet（10-16 最佳 checkpoint） | DGCRN（10-15 启动的最佳 checkpoint） |
|---|---|---|
| 最佳 epoch | 25 | 68 |
| Gaussian 非对角非零条目 | 35 / 105300 | 0 / 105300 |
| Matérn 非对角非零条目 | 98 / 105300 | 10 / 105300 |
| 三核合成图正向归一化后平均自环权重 | 约 66.95% | 约 82.79% |
| 初始 alpha=1、均匀 mixing 的同口径自环权重 | 约 17.87% | 约 17.87% |
| 合成图 NaN / 零行 | 无 / 无 | 无 / 无 |

RQ 分支仍提供主要的跨节点关系。RQ 的很小正数受公式 epsilon 下限影响，不能仅看“非零条目很多”就认为关系有效。整体图并未全零。

checkpoint 三组 alpha、logits 都有 optimizer 状态并发生改变；当前 forward -> PyTorch 图归一化 -> 图卷积 -> loss 的路径没有发现 detach/NumPy 断图。模型还保留自身 adaptive graph，因此核退化也不等于整个模型无法空间建模。

### 7.2 有证据支持的机制，尚非唯一因果结论

- GWNet/DGCRN 的 Adam weight_decay=1e-4；builder 把 model.parameters() 全部放进同一组，包含核尺度和 logits。
- 核尺度控制关系范围，惩罚 alpha 向 0 会把局部范围压窄，语义不同于惩罚普通线性层权重。
- 当前 a=ReLU(alpha)，负尺度的数据梯度为 0；很小的正尺度也会让 Gaussian/Matérn 的指数数值下溢，数据梯度难以恢复。正则化/Adam 状态仍可能使原参数发生变化，不能把“数据梯度为零”说成“参数绝对不再变化”。
- 多数远距离/不可达边本就数据梯度弱，尺度被正则化压缩可能合理；但 DGCRN Gaussian 连有限距离跨节点项也全部消失，值得独立诊断。
- 原始参考代码包的 optimizer 同样全参数 weight_decay，核同样 ReLU。不能说已确定是用户改错；也不能说师兄论文实际运行一定采用该包版本。

未证明：取消 kernel weight_decay 一定恢复论文结果；核退化是否一定有害；是否有更合适的核长度尺度；历史结果是否使用不同随机 seed/源码。需要单变量验证，不能通过多个超参数同时改来归因。

### 7.3 第二批建议的最小修改

1. 增加可配置 kernel_weight_decay。默认缺省/None 时继承原 decay，保留 legacy 模式；diagnostic 模式 backbone 保持 1e-4，仅 kernel_layer 参数设为 0，lr/epoch/scheduler/数据划分不变。
2. optimizer 通过实际子模块参数身份分组，保证不遗漏、不重复；没有 kernel 的模型仍走原行为；DDP 需先 unwrap module。checkpoint 保存分组策略。旧 optimizer state 的组结构不兼容时禁止静默恢复；旧模型权重仍可严格加载，但诊断对照应从同样初始化开始。
3. 首先只看短检查中的数据梯度、尺度分位数、分支非对角质量、正向/反向 support 自环比例与熵，再决定是否需要完整配对实验。使用验证指标决策，保留旧测试结果。
4. 如取消核 decay 后仍出现死区，再单独尝试 a=softplus(raw)+eps，初始化 raw=softplus_inverse(1-eps) 以保持实际初值 a=1。不能把旧 alpha 权重直接当 raw 加载；只能显式转换或作为新版本训练，并验证初始矩阵一致。
5. Softplus 也不能保证学不到很小尺度；不自动加入尺度硬下界或强行等权 mixing。它们会额外改变先验，需另设明确实验。
6. 不把此更改顺手施加到 GRIN/Kriging；当前 GRIN 的 l2_reg=0，没有证据要求同时改变三个任务。

### 7.4 其他配置差异

参考代码包 GWNet 的最大轮数是 100，本地是 200；本地最佳在 epoch 25，早停在后续，不能仅凭上限不同解释当前差距。DGCRN 参考包与本地均为 100，scheduler milestones 为 [100,150]，在此次最佳 epoch 68 前并未触发。应记录真实配置，不能猜测延长到 200 就是论文设置。

## 8. 需要用户确认的修改范围

建议先批准第一批：

- 修正 GRIN 参数注册位置、Imputation 两阶段参数解析和覆盖顺序。
- 给 MPGRUNet/BiMPGRUNet 与四个 forecasting backbone 增加显式 kernel 选择；默认保留各自原有效行为，并记录真实实例。
- 修复最终指标导出，补齐 resolved_config、实际 seed、版本和 checkpoint 选择元数据。
- 明确记录 BiMPGRUNet embedding_size=0，保留未启用 embedding 的结构。
- 对历史异常运行准备独立仅评估恢复方式，但未知 theta/运行归属先确认；不重训。

第二批需单独确认：GWNet/DGCRN 的核参数 weight_decay 分组对照；如有需要再做 Softplus。第一批不替换尺度参数化、不修改学习率和任务公式、不改变数据划分。

## 9. 修改后的验收（均待确认后执行）

| 检查 | 通过条件 |
|---|---|
| 参数解析 | config-only、CLI 覆盖、显式模型选择均得到预期类；不再按 brits 注册另一个模型的参数 |
| 核选择 | 相同 seed 只切 kernel_type 可切换加法/Aimp；配置、实例、导出一致；threshold 保持原有效值 |
| 构造 | 单向/双向节点数正确、无缺失导入；legacy checkpoint 无意外 missing/unexpected keys |
| 结构 | BiMPGRUNet 无新 embedding，输出层仍是 129 输入；不以接口修复改变论文对照模型 |
| 指标恢复 | 无 kernel_type 的旧模型不丢失已计算主要指标；mean 聚合与窗口级指标分别标注 |
| 配置记录 | 每次记录真实核、theta、seed、源码、数据协议、optimizer groups、best checkpoint |
| 参数分组（第二批） | backbone/kernel 参数覆盖完整且不重叠；legacy 设定保持原 optimizer 语义 |
| 梯度（第二批） | 只在批准后的单批次内存检查实际数据梯度与一次更新，不先启动完整训练 |

## 10. 关键证据路径

- 当前模型：`F:/py_project/CSKL/grin-main/lib/nn/models/grin.py`、`mpgru.py`。
- 旧双向 checkpoint：`F:/py_project/CSKL/grin-main/logs/la_block/bimpgru/2025-01-31_13-46-50_2020_st/`、`2025-01-31_13-46-33_2020_ada/`。
- 新双向 checkpoint：同一日志根下的 `2026-10-16_14-33-59_2020`、`14-36-23`、`22-53-10`、`22-54-00`。
- GWNet：`F:/py_project/CSKL/BasicTS-master/checkpoints/GraphWaveNet/PEMS-BAY_200_12_12/20261016_120533_286_pid33456/6cf8f4402e89e8775d8fda5c5886482e/GraphWaveNet_best_val_MAE.pt`。
- DGCRN：`F:/py_project/CSKL/BasicTS-master/checkpoints/DGCRN/PEMS-BAY_100_12_12/20261015_091318_522_pid25104/6225182df0b8ef360fb80079d035bdf8/DGCRN_best_val_MAE.pt`。
- 原始参考包：`F:/py_project/论文代码（仅代码）/project_CSKL/`。它提供对比证据，不自动视为论文表格的唯一版本。
