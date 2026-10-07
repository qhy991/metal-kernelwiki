# 量化 KV attention：把量化损失与多 token 执行错误分开

2026-10-07，Apple M4、MLX 0.31.2 / MLX-LM 0.31.3 的无模型探针中，**128 个配置有 48 个未通过数值门**。失败均发生在本次测试的 Tk={1024,2048,4096}、Lq={2,3}；QK、softmax 和同一份量化数据的 dense 对照全部通过，错误定位到量化 PV。执行以 exit 2 保留失败结束，没有性能验收、软件更新或容差修改。

## 实际调用路径

本轮调用安装版本的 `QuantizedKVCache.update_and_fetch` 和模型使用的 `scaled_dot_product_attention` 包装函数。[v0.31.3 helper](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py)根据 cache 的 bits 属性选择：Q 乘 scale → 转置量化矩阵乘得到 QK → mask → precise softmax → 非转置量化矩阵乘得到 PV。它不是把 packed cache 直接传给 `mx.fast` SDPA。

GQA 将 Q 从 `[B,Hq,L,D]` 改为 `[B,Hkv,R,L,D]`，R=Hq/Hkv；每个 query head 使用 `floor(head/R)` 对应的 KV head。KV 三元组增加广播轴。`"causal"` 使用右下对齐，独立参考条件为 `key <= Tk-Lq+query`。本轮没有全遮挡行、任意 per-head mask 或 attention sinks；不能由这些结果推断其行为。

每个 Lq 都从新 cache 开始：先追加前 Tk−Lq 个 K/V，按旧 offset 生成 mask，再追加最后 Lq 个，使用 fetch 返回的有效切片。内部容量与有效长度必须区分。[缓存实现](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) 本轮 48 组跨 Lq packed 比较精确一致，64 次 cache 解码与 CPU 单次舍入解码一致，因此阶段参考不依赖一个可能已经改变的旧 cache。所有 helper 与阶段对照重新构造 Q，避免源码中的 `queries *= scale` 导致重复缩放。

## 覆盖与预设数值门

M4/16GB，macOS 27.0（26A428），Python 3.14.3、NumPy 2.4.3，默认 Metal GPU，`MLX_ENABLE_TF32` 未设置。float32，B=1、Hq=4、Hkv={1,2}、D=128，Tk={512,1024,2048,4096}、Lq={1,2,3,4}，affine bits={4,8}、group=64。

CPU PCG64 seed=53 依次生成 Q `[1,4,4,128]`、K `[1,2,4096,128]` 的标准正态数，再生成 V=0.5×normal。Q 分为原值和放大 4 倍的 focused 组。其他 shape 使用前缀；重复条件不是独立随机样本。scale 固定为 `float32(1/sqrt(128))`，CPU 先按 float32 舍入 Q×scale。

CPU 从实际 uint32 packed 参数低位到高位解码，以 float64 计算 `code*scale+bias` 再舍入到 float32；所有 16 个格式与 native dequantize 精确一致。随后用 float64 QK、稳定 softmax、PV 构成完整 oracle。原始未量化 K/V 只用于单独报告量化失真，不替代此执行正确性参考。

门槛在执行前固定：QK、PV 与完整输出的最大绝对误差 ≤2e-5、最大逐行 relative L2 ≤1e-4；softmax 分别 ≤5e-7、≤1e-4，行和误差 ≤1e-6，概率非负且被遮挡位置精确为零。另检查形状、float32 dtype 和有限性；零参考行要求精确为零。这些是本合成任务的目标，不是通用 API 容差。

## 分阶段结果

| 检查 | 通过 / 总数 | 最大绝对误差 |
|---|---:|---:|
| QK 对 CPU 解码 K | 128/128 | 9.85e-6 |
| softmax 对实际 GPU QK 的 CPU softmax | 128/128 | 1.89e-7 |
| 量化 PV 对实际 GPU P × CPU 解码 V | 80/128 | 2.38424 |
| 实际 attention 包装函数对完整 CPU oracle | 80/128 | 2.38424 |
| dense PV 对相同 P/V | 128/128 | 2.03e-6 |
| 解码 K/V 后的 fast SDPA 对完整 oracle | 128/128 | 8.92e-7 |

实际包装函数与手工阶段输出逐元素一致。阶段 PV 使用已经计算出的实际 GPU probabilities 作 CPU 参考，故前面 QK 的舍入不会被误归因给 PV。两个 dense 对照同样使用这一份量化参数，不是换回原始 K/V 后比较。

下表每格覆盖两种 KV head 数、两种 bits 和两种 Q 分布，共 8 个配置；“通过”仅指本次门。

| 有效 Tk | Lq=1 | Lq=2 | Lq=3 | Lq=4 |
|---:|---|---|---|---|
| 512 | 8/8 通过 | 8/8 通过 | 8/8 通过 | 8/8 通过 |
| 1024 | 8/8 通过 | 8/8 失败 | 8/8 失败 | 8/8 通过 |
| 2048 | 8/8 通过 | 8/8 失败 | 8/8 失败 | 8/8 通过 |
| 4096 | 8/8 通过 | 8/8 失败 | 8/8 失败 | 8/8 通过 |

该现象扩展了先前[二维 QVM 失败](quantized-matmul-validation.md)的已测范围，现已包括指定 GQA、causal mask、softmax 和真实 helper。它仍然不是模型或服务复现，也不代表所有 Tk≥1024 的配置都已经执行。

## 源码解释与处置

[v0.31.2 前端](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/ops.cpp)广播 batch 轴而不把 GQA repeats 合入 M；结合[Metal dispatch](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp)，本输入域中 PV 的 M=Lq。源码将 M<4、K≥1024 分配给 split-K QVM，所测长上下文均使用 8 分区；Lq=4 使用另一矩阵路径。这是源码链推断，不是本机 profiler 或二进制 dispatch 证明。

对 48 个失败案例另做纯 CPU 事后诊断：在每个 query head 内展平实际 P，以错误起点 `row*(Tk/8)` 读取 Tk 个值，再乘对应 KV head 的解码 V。各案例最大拟合残差为 **4.80e−9–2.44e−7**。对正确 PV 参考，首行的各案例最大误差为 4.28e−9–1.96e−7，后续行的各案例最大误差为 **0.038415–2.384243**。这支持旧行跨度缺陷的解释；错误切片既不重新归一化，也不替代正确 oracle，原有 48 个失败保持失败。它仍不是二进制 dispatch 或修复通过的证明。

[历史修复](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68)将分区长度与完整行 stride 分开。[v0.32.0 发布说明](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)列明包含 #3497；该发布记录不证明本轮 0.31.2 安装包已修复，也不代替目标机器验证。本轮没有安装新版本复验。该失败域应阻止相应部署候选进入性能验收。Lq1/Lq4 对照通过不能自动授权逐 token 拆分、补齐到 4 或重分 batch 作为修复；仍需完整状态、mask、质量和端到端验证。

## 存储收益不等于正确性通过

本轮所有 cache 容量恰好等于有效 Tk，tensor payload 包括 packed K/V 以及 float32 scales/biases。每个值的存储为 `bits/8 + 8/group` 字节，实际 nbytes 相对相同形状 float32 K/V 的压缩比为 4-bit **6.4×**、8-bit **3.556×**。这是缓存张量本身的比值，不是进程峰值或任意容量下的部署节省。

用同一 CPU attention 流程比较解码 K/V 与原始 K/V，得到以下合成量化失真。每格是该 bits 的 64 个配置中，各配置最大指标的范围；没有模型质量门或业务验收含义。

| bits | 最大绝对差范围 | 最大逐行 relative L2 范围 |
|---:|---:|---:|
| 4 | 0.00586–0.32850 | 0.13645–0.63584 |
| 8 | 0.000307–0.02425 | 0.00740–0.04104 |

这份失真的参考分母是原始 K/V 的 CPU 输出，执行误差的参考则是解码 K/V 输出；两个 relative L2 不能相加。即使 8-bit 的量化失真较小，也不能掩盖这轮独立发现的 PV 执行错误。

## 保留范围

独立 CPU 审计逐列解包 16 份已保存的 packed，并以逐 head `NumPy @` 重算全部 128 个完整参考和实际 P 的 PV；相对原 worker 的参考最大差约 4.84e−15，896 项阶段指标与原记录在 1e−12 的运算顺序差内一致，失败集合不变。跨 Lq 的 48 次 packed 相等性有在线检查记录，但未另存重复数组供独立复查。query_after 未保存，因此不据此断言精确的物理 alias 或原地变更机制。

原始输入、16 个 packed 格式、128 个案例的 Q、mask、GPU 各阶段输出、dense 对照、CPU 输出、安装 helper 源码及全部失败指标保留在仓库外。来源 `local-mlx-quantized-attention-20261007`，逻辑引用 `2026-10-07-mlx-quantized-attention/derived/summary.json`。原始记录未随公开库发布，外部读者不能仅凭页面独立复验原运行。

未测 BF16/FP16、其他 group、B>1、D≠128、全遮挡行、sink/窗口/旋转缓存、转换触发点、语言模型质量、内存峰值、时间或 profiler。配合[KV cache 决策](mlx-kv-cache.md)、[QVM 数值定位](quantized-matmul-validation.md)与[验证流程](measurement.md)使用。
