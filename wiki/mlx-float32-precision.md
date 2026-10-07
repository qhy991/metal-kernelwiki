# Float32 输出不等于内部全精度：用恒等关系定位

证据：官方文档与限定本机观察，检查于 2026-10-07。先分开存储 dtype、内部计算精度、任务的数值门槛；不能仅凭输出为 float32 就断言每个乘加都采用完整 float32。

## 版本和实际路径决定精度

[MLX 0.32.3 Numerical Precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html) 说明部分矩阵运算可默认走降精度硬件路径，输入输出保持 float32；适用算子和误差依后端、硬件而变。该新版说明不能直接追认本机 0.31.2 的完整契约。文档提出的精度选项也要先核对安装版本支持；本轮没有改开关后重跑。

固定任务容差后再比较优化。数学恒等关系给出很强的诊断，但不自动成为框架 bitwise 承诺；精确关系未成立、任务容差未过门和违反框架契约，应分别记录。若另设精度选项实验，保留默认路径的结果，不把改配置后的通过回填到原运行。

## Singleton attention 的诊断构造

当只有一个未被 mask 的 key、没有 sinks 时，softmax 的结果为 1，输出应为 V。设 Q/K 全零，`B={1,2},Hq=16,Hkv=1,Tq=Tk=1,D=512,scale=1,mask="causal"`。B=2 时 Q/K/V 都显式有两个 batch，避免混入额外 batch broadcast。

对同一组实际存储 V 比较：compact SDPA、显式 repeat K/V 后的 SDPA、`ones @ V` 的 head broadcast、显式 repeat V 后的 matmul。记录有限值、最大误差、是否与 V 精确数值相等（`np.array_equal`）；再记录是否等于 V 经 fp16 或 bf16 舍入后的参考。这只能说明舍入模式吻合，不能证明具体 dispatch 或融合。

V 使用三类输入：不同 batch 的线性 ramp、seed=19 的带符号正态值、包含 `±(1+2^-10)` 和 `±(1+2^-12)` 的精度哨兵。前者在 fp16 中可表示，后者不可；fp16 输入本身已发生的舍入必须进入参考，不能与原始 float64 构造值混比。

## M4 本地结果：没有观察到该历史现象

Apple M4/16GB、macOS 27.0（26A428）、MLX 0.31.2，`MLX_ENABLE_TF32` 未设置。上述 4 种调用 × 2 个 batch × 3 类 V × float32/float16，共 **48 组配置**，在 3 个独立进程重复执行；每次均有限、最大绝对误差 0、精确等于实际存储 V。三个进程使用相同输入，并不等于 144 个不同测试条件。

上游 [issue #3953](https://github.com/ml-explore/mlx/issues/3953) 报告的是 M5 Max/MLX 0.32.0 的广播与单 batch SDPA 精度现象；检查时 Closed，但未核实修复 commit。本机结果只能说明这个输入域未观察到相应精度损失，不能证明该历史问题已修复，也不能保证所有 float32 路径。

本地来源 ID：`local-mlx-boundaries-20261007`；外部逻辑引用为 `2026-10-07-mlx-boundaries/summary.json`。原始脚本/结果留在仓库外，未随公开库提供。调用方是否 repeat 只是 API 输入差异，本轮未解析 kernel 路径；也没有为这个 singleton 构造采集性能结果。

## 优化时的取舍

显式物化 broadcast 是诊断对照，不是默认修复。它可能增加分配和访存；即使某版本数值不同，也须在任务质量门与整体成本下评价。singleton 通过不能覆盖多 key 的点积、softmax 和 PV 归约误差；下一步应改变 key 数、mask、head dimension、输入动态范围及实际 cache offset。

量化权重还需把量化损失、解码舍入和 kernel 误差分开，见 [量化 matmul 验证](quantized-matmul-validation.md)。求值边界对时间的影响见 [异步求值](mlx-async-evaluation.md)，勿让“数值相同”替代性能测量。
