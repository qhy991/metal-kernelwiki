# 量化矩阵乘：把性能路径与数值路径一起比较

证据状态：上游报告与待验证方法，核查于 2026-10-07。部分数值现象已有 [M4 本地记录](local-mlx-m4.md)，尚无整模型收益或质量结论。

## 先确定误差来自哪里

保存一次量化产生的 packed weights、scales、biases，所有候选复用它们。用同一份实际存储输入建立 CPU float64 参考。区分三种差异：原始权重与量化权重之差、量化权重解码时的 dtype 舍入、kernel 的乘加/归约误差。拿未量化权重作唯一 oracle 会把几类误差混在一起；以另一个 GPU matmul 作唯一参考也可能共享缺陷。

建议同时保留：CPU 解包后的理想 affine 权重参考、解码值再舍入到输入 dtype 的参考、实际 kernel 输出。单独核对 CPU 解码与框架 `dequantize` 的关系。BF16 的参考舍入若采用 float64→float32→BF16，明确写出两阶段过程，不声称是任意 float64 直接舍入的证明。

## Batch 改变可能改变数值路径

上游 [issue #4613](https://github.com/ml-explore/mlx/issues/4613) 在 MLX 0.32.3、M4 Max 报告 Split-K 的 partials 存储精度影响结果。检查时仍为 Open；作者对根因的判断不能代替本地 dispatch 证据。

可执行的小型诊断：固定 `X:[128,2560]` 与 `W:[1024,2560]`，affine 4-bit、group 64；一次量化，令 `M={16,32,33,64,65,128}`，每次使用相同 X 的前 M 行。比较每种 M 的前 16 行，记录最大/平均绝对差、相对 L2、有限值计数。最终 dtype 相同并不承诺 bitwise batch invariance；未满足任务质量门槛前，不据此批准 batch/dispatch 改动。

把 kernel MAE 除以“理想结果仅做最终 dtype 舍入”的 MAE，是描述性比值，不是 ULP 数、合法误差上限或缺陷判据。分母很小时，float32 也可能得到很大的比值。应同时看绝对误差与任务质量，而非只看倍数。M4 的已测结果与上游环境不同，只能支持相似现象，不能宣称复现同一个根因。

## 比性能需要三条路径

上游 [issue #4621](https://github.com/ml-explore/mlx/issues/4621) 在 MLX 0.32.3、M2 Max 64GB 上报告某些大 M 场景中反量化加 dense matmul 比 packed QMM 快。它是候选来源，不能把报告中的 token 阈值直接变为其他机器的分派规则。

| 路径 | 计时内的工作 | 内存核算 |
|---|---|---|
| A：packed QMM | 直接使用同一 packed 权重 | packed 常驻及算子临时量 |
| B：逐次解码+dense | 每次新建 `dequantize` 图，再 matmul 并求值 | packed 常驻、dense 临时量及其生存期 |
| C：常驻 dense | 已在计时前求值的 dense 权重上 matmul | dense 常驻必须纳入部署预算 |

这三路是本页提出的实验设计，尚未在本机计时。先用较小 `W:[2048,1024]`、BF16、bits={4,8}、group=64、`M={1,8,32,128,512}` 跑通数值门；它不保证命中上游大模型路径。各路径在独立新进程核算内存，防止 C 的常驻 dense 污染 A/B；保留相同种子与确定性输入构造。计时需构建新操作并求值，B 不得把解码移到计时外。

在运行前定义任务允许的数值/质量门槛；先通过 oracle 再比较重复分布。不要在结果出来后放宽容差。若仅部分 M 获益，补实际模型的 prompt chunk、prefill、decode、长上下文质量及内存，再决定是否分派。不同 shape、bits、设备上的阈值独立确定。

## 下一次结果应改变什么

- A/B 数值未过门：定位解包、dtype、累加或 mask，不隐藏在更宽容差后。
- B 大 M 获益、C 也获益：确认临时解码成本与 dense 常驻预算，再做端到端验证。
- 只 C 获益：收益依赖额外常驻内存；不能称为零成本替换。
- 时间分布重叠或主机等待占主导：先建立测量分辨率和 profiler 证据，再调 kernel。

测量合同见 [measurement](measurement.md)，工具覆盖限制见 [runtime-measurement](runtime-measurement.md)。
