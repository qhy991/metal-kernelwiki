# 量化矩阵乘：把性能路径与数值路径一起比较

证据状态：上游报告、M4 限定本地观测与待验证性能方法，核查于 2026-10-07。r1 数值现象见 [M4 本地记录](local-mlx-m4.md)；后续多行 QVM 失败见本页。尚无整模型收益或质量结论。

## 先确定误差来自哪里

保存一次量化产生的 packed weights、scales、biases，所有候选复用它们。用同一份实际存储输入建立 CPU float64 参考。区分三种差异：原始权重与量化权重之差、量化权重解码时的 dtype 舍入、kernel 的乘加/归约误差。拿未量化权重作唯一 oracle 会把几类误差混在一起；以另一个 GPU matmul 作唯一参考也可能共享缺陷。

建议同时保留：CPU 解包后的理想 affine 权重参考、解码值再舍入到输入 dtype 的参考、实际 kernel 输出。单独核对 CPU 解码与框架 `dequantize` 的关系。BF16 的参考舍入若采用 float64→float32→BF16，明确写出两阶段过程，不声称是任意 float64 直接舍入的证明。

## Batch 改变可能改变数值路径

上游 [issue #4613](https://github.com/ml-explore/mlx/issues/4613) 在 MLX 0.32.3、M4 Max 报告 Split-K 的 partials 存储精度影响结果。检查时仍为 Open；作者对根因的判断不能代替本地 dispatch 证据。

可执行的小型诊断：固定 `X:[128,2560]` 与 `W:[1024,2560]`，affine 4-bit、group 64；一次量化，令 `M={16,32,33,64,65,128}`，每次使用相同 X 的前 M 行。比较每种 M 的前 16 行，记录最大/平均绝对差、相对 L2、有限值计数。最终 dtype 相同并不承诺 bitwise batch invariance；未满足任务质量门槛前，不据此批准 batch/dispatch 改动。

把 kernel MAE 除以“理想结果仅做最终 dtype 舍入”的 MAE，是描述性比值，不是 ULP 数、合法误差上限或缺陷判据。分母很小时，float32 也可能得到很大的比值。应同时看绝对误差与任务质量，而非只看倍数。M4 的已测结果与上游环境不同，只能支持相似现象，不能宣称复现同一个根因。

## 多行 QVM：前缀一致仍可能一起算错

2026-10-07 在 Apple M4/16GB、macOS 27.0（26A428）、MLX 0.31.2、NumPy 2.4.3、Python 3.14.3 上执行一次无模型探针，默认 GPU，`MLX_ENABLE_TF32` 未设置。覆盖 [历史回归测试](https://github.com/ml-explore/mlx/pull/3497/commits/2ecf184f9150b85b0aa139f7d827751011874d35) 的形状范围，但使用自己的 CPU PCG64 输入，并非原测试输入重放：

- float32，`X:[M,K],W:[K,128]`，`M={2,3},K={2048,4096},bits={4,8},group={32,64},transpose=False`。
- 16 个形状/格式组合，seed=23/47，共 32 个用例。X 和 W 用独立 SeedSequence `[seed,K,0/1]` 生成 `0.1*normal` 后存为 float32；同 seed/K 的原始输入跨格式复用，M2/M3 共享 X 前缀和同一份 packed W。
- CPU 按 uint32 从低位到高位解包，以实际 float32 scales/biases 做 float64 affine 解码，再一次舍入到 float32，最后以 CPU float64 dot 作主 oracle。预设 `max_abs<2e-3`；该阈值来自上游 dense 对照，被此探针采用为准确性目标，不是通用误差承诺。
- 同时保存理想 affine、分开 float32 乘/加舍入两种参考，以及 `mx.dequantize` 后 GPU dense 对照。原始权重、packed 参数和输出均保留；不把量化损失混入 kernel 对照。

**32/32 用例均未通过主 oracle 和 GPU dense 对照门。** 首行的逐行最大误差至多 `2.836e-7`，后续行的逐行最大误差为 `1.587～2.847`。GPU dense 相对主 oracle 最大误差为 `6.804e-7`；16 份权重的单次舍入 CPU 解码与 native dequantize 精确数值相等，但这不证明 GPU 使用了某种 FMA 指令。

安装包附带头文件与 [v0.31.2 kernel](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h) 仍有旧行跨度写法；[同版本 host 路由](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp) 为这些形状选择 8 个 K 分区。[历史修复](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68) 将分区长度与完整行 stride 分开。源码观察不是当前二进制 dispatch 的证明。

仅用保存数组做的事后 CPU 诊断：若错误地按 `K/8` 跨到下一行，第 r 行会读取 `flatX[r*(K/8):r*(K/8)+K]`。这一错误切片模型与实际输出的最大残差为 `2.907e-7`，支持旧 stride 缺陷假设；它不是新 oracle，没有把失败改判为通过，也没有修复安装包。

16 组 M2/M3 前缀比较全部精确相等，尽管后续行均算错。因此 batch 一致性只能补充外部正确性检查。遇到这种失败，先阻止该候选进入性能验收，再对修复版本或其他实现做独立验证；不要放宽容差、只比较第一行，或直接采用未经验证的逐行调用作为部署修复。

本轮没有性能、profiler、模型质量或其他 shape/dtype 的结论。来源 `local-mlx-qvm-cache-20261007` 的外部逻辑引用为 `2026-10-07-mlx-qvm-cache/results-summary.json`；原始失败、脚本、输入/输出和独立诊断留在仓库外，未公开。这里与 r1 的 transpose=True、M>=16 观察属于不同输入域。


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
