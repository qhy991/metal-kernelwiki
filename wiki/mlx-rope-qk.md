# RoPE 合并 Q/K：位置、布局与批量解码先过数值门

2026-10-07 的 M4/MLX 0.31.2 无模型验证中，360 个配置、两条路径共 **720 项检查，486 通过、234 失败**。Q/K 沿 head 合并再做一次 RoPE，在部分配置下会把原本通过的隔列输入变成失败路径；另有两条路径共同的长 offset 精度误差。本轮没有计时、profiler 或模型收益结论，保留 exit 2 与原容差。

## 合并成立的条件

比较 `rope(Q), rope(K)` 与 `split(rope(concat([Q,K], axis=1)), [Hq], axis=1)`。两者 B、T、D、dtype 和 RoPE 参数必须一致，只有 head 数可以不同；这里 Hq=8、Hkv=2，必须在 8 处切分，不能等分。

[API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html)将最后两维解释为 T/D。`dims` 旋转前若干通道，尾部保留；traditional 配对相邻通道，另一模式配对前后半段，分界是 dims/2。按 [v0.31.2 前端](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/fast.cpp)，dims 必须为正偶数且不超过 D，offset 为整数标量或长度 B 的向量。base/freqs 二选一，freqs 长度为 dims/2；实际使用其倒数，不能把 inverse frequencies 再直接当 freqs。当前网页版本为 0.32.3，本轮安装版本另行核对。

模型中的前处理和缓存时序也属于契约：[Llama v0.31.3](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/llama.py)先对新 Q/K 用相同 cache offset 做 RoPE，再追加缓存；[Qwen3](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3.py)在此之前分别执行 Q/K norm。不能合并已旋转的历史 K，或省去不同权重的 norm。此处源码研究不构成模型运行结果。

## 预设实验域

Apple M4/16GB、macOS 27.0（26A428）、Python 3.14.3、MLX 0.31.2、NumPy 2.4.3，默认 Metal GPU，`MLX_ENABLE_TF32` 未设置。未导入模型，MLX-LM 安装版本为 0.31.3。

- 五个工作负载：B1/T1 标量、B2/T1 标量、B2/T1 等值向量、B2/T17 向量、B1/T256 标量 offset。
- 每项交叉 float32/BF16、连续/head-sequence 转置/隔列切片三种输入、dims=64/128、两种 traditional、offset=0/8192/131072，共 360 个配置。每个配置都执行分开与合并两条路径。
- B2/T1 两种 offset 表达使用相同输入和位置；B2/T17 的实际位置向量为 `[offset, offset+7]`。T1 的 head-sequence 转置在内存上仍可连续，不能算作独立非连续控制。隔列输入的末维 stride=2，已用小型元数据 kernel 读取实际 shape/strides。
- PCG64 seed=67 顺序生成 Q `[2,8,256,128]` 和 K `[2,2,256,128]` 标准正态 float32，其他输入取前缀。BF16 在 CPU 按 RNE 舍入后与实际 GPU 输入逐元素比较。重复前缀不是独立随机样本。

CPU 以实际存储的输入和 float64 计算 `theta=(offset+token)*10000**(-2*j/dims)`，再计算正余弦及旋转；scale=1。这是理想位置公式的任务参考，不是 float32 Metal 实现的逐位模拟。只在前 dims 内计算最大绝对误差及最大逐行 relative L2，尾部另要求精确不变，Q/K 和每个 batch 单独核查。

预设门：float32 的 max_abs≤0.002 且 row-relative-L2≤0.0005；BF16 分别≤0.03、≤0.005，同时要求形状/dtype/有限性正确。门槛是本合成任务的目标，不是框架保证或模型质量容差；两路径彼此相同不能代替这套参考。

## 实际结果

下表每格含两种 dtype、三种输入布局、两种 dims、两种 traditional 和两条路径，共 48 项；数字为通过数。

| 工作负载 | offset=0 | 8192 | 131072 |
|---|---:|---:|---:|
| B1/T1，标量 | 48/48 | 48/48 | 24/48 |
| B2/T1，标量 | 48/48 | 8/48 | 4/48 |
| B2/T1，等值向量 | 48/48 | 48/48 | 24/48 |
| B2/T17，向量 | 48/48 | 48/48 | 12/48 |
| B1/T256，标量 | 48/48 | 30/48 | 0/48 |

所有输出有限，所有未旋转尾部精确保留。360 对路径中 344 对逐元素相同，但其中 **107 对共同未过 oracle 门**。237 个配置两条路径均过门，111 个均失败，12 个仅分开路径通过；没有仅合并路径通过的配置。因此“合并前后对齐”与“计算满足位置精度要求”必须分开判断。

### 批量单 token：布局改变触发不同结果

B2/T1、offset=8192 的隔列输入，八个 dtype/dims/traditional 组合的分开路径全部通过，合并路径全部失败。同位置的 `[8192,8192]` 向量 offset 控制在全部 48 项中通过。本轮 offset=0 控制也全过，不能用零位置检查替代非零位置验证。

例如 float32、dims=128、traditional=False、末维 stride=2：分开路径 Q 的 max_abs 为 0.000859；合并后第一个 batch 仍为这个量级，第二个 batch 的 Q/K 最大绝对误差分别约 **5.04059 / 4.00554**。数学上合并条件成立，也不能据此跳过具体布局与版本的正确性检查。

[v0.31.2 host 源码](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/rope.cpp)在连续、T=1、单一 offset 时使用 single 路径，其网格 head 维为 N；一般路径包含 B。concat 产生连续输入，能改变候选路径。这与当前结果一致，但本轮没有二进制 dispatch 或 profiler 证明，不将源码推断当作已捕获的执行轨迹。

[历史修复 b545a35](https://github.com/ml-explore/mlx/pull/3498/commits/b545a35b9baa8f471a134718a930d97ef46cf504)把 single 网格的 N 改为 B×N；[测试 bf6421b](https://github.com/ml-explore/mlx/pull/3498/commits/bf6421bfb05d9e39d182432611b99ee9a3dca933)覆盖 36 个 B/head/D/配对组合、T=1、标量 offset=5。[v0.32.0 发布说明](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)列出 #3498。本轮没有升级或执行修复版本，向量 offset 与隔列布局只是诊断对照，不能直接推荐为服务修复。

### 长位置：两路共同误差也可能越界

剔除 B2/T1 标量这一特定风险，其他工作负载仍会在较大 offset 超过预设门。例如 B1/T256、float32、dims=128、非 traditional 的分开路径：Q 最大绝对误差从 offset=0 的 **5.49e−5**，增至 8192 的 **0.002148** 和 131072 的 **0.033861**；三个布局的合并输出均与各自分开输出相同。

[Metal 实现](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/rope.metal)以 float32 构造位置/频率/角度，使用 fast sin/cos，最终转回输入 dtype。这提供数值误差机制的候选解释；没有逐阶段测出误差来源的占比，也没有把本次偏差全归因于某个函数。所测位置远低于 float32 的连续整数分辨率上限，不能用“位置整数本身已无法表示”解释这些结果。BF16 门与 float32 不同，BF16 某格通过不代表其精度更高。

## 为什么一次调用未必节省工作

[concat 实现](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/slicing.cpp)分配合并 buffer 并复制输入；[split](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/common.cpp)可返回共享 buffer 视图。拆分后 batch stride 可能仍跨越全部合并 heads，影响下游布局与存活内存。

本轮 B2/T17、float32、连续输入、dims=128、offset=0 的实际输出元数据显示：分开路径 Q/K 的 batch stride 分别为 **17408 / 4352 个元素**，合并再拆分后均为 **21760**；head/sequence/feature stride 都是 `[2176,128,1]`。这确认了输出布局差异，未测其下游复制成本或延迟。full/partial RoPE 还存在分配、donation 与复制条件，不能从 Python 调用次数推导 kernel 总数。

优化流程应先保留模型的参数与状态，再验证代表性 batch/位置/布局。数值满足目标后，才比较含 concat、拆分输出、cache/attention 消费者的主机完成时间、分配与 profiler；本轮没有达到整个候选域的正确性要求，因此没有性能结论。

## 原始记录与边界

独立 CPU 审计用复数旋转重算参考，并单独实现 BF16 RNE；720 项路径的失败集合保持一致，参考最大差约 2.99e−11。144 对 B2/T1 标量与等值向量控制的第一个 batch 全部逐元素相同，第二个 batch 仅 64 对相同；向量控制仍有长 offset 精度失败。审计只读取保存值，GPU dtype 与 shape/stride 来自原在线检查，未重跑 GPU。

原始输入、执行脚本、安装 docstring、实际布局元数据、360 个配置的输入/CPU oracle/两路输出与全部失败指标保存在仓库外。来源 `local-mlx-rope-qk-20261007`，逻辑引用 `2026-10-07-mlx-rope-qk/derived/summary.json`。原始记录未随公开库发布，外部读者不能仅凭此页独立复验原运行。

未测 FP16、其他 base/scale、自定义 freqs、负位置、任意 position IDs、不同 H/D、模型 RoPE scaling、量化或旋转缓存、梯度、端到端质量、内存峰值或修复版本。配合[融合](fusion.md)、[MLX 执行](mlx-execution.md)和[Attention](attention.md)使用。
