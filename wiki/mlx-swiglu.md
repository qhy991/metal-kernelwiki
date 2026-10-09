# SwiGLU：已有编译函数、显式表达式与自定义 Metal

[English companion](en/mlx-swiglu.md)

证据：2026-10-07，Apple M4、MLX 0.31.2、MLX-LM 0.31.3 的模型外比较。**216 项预检查通过**，包括 F32/BF16、连续与隔列输入和非整齐尾维。优先确认调用链里是否已经使用 compiled helper；手写 kernel 还会改变数值计算方式，不能把差异全部解释成融合。

## 确认基线已有的优化

[MLX-LM v0.31.3 的 swiglu](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/activations.py) 已使用 `mx.compile(shapeless=True)`，参数顺序是 gate、x。[MLX v0.31.2 的 SiLU](https://github.com/ml-explore/mlx/blob/v0.31.2/python/mlx/nn/layers/activations.py) 本身也已编译。因此本轮的普通表达式基线直接用 `mx.sigmoid`，不经 `nn.silu`。

| 路径 | 返回的唯一输出 |
|---|---|
| expression | `(gate * mx.sigmoid(gate)) * x` |
| library_compiled | 安装版本 `mlx_lm.models.activations.swiglu(gate, x)` |
| custom_strided | 原创 MSL，以 float32 计算稳定 sigmoid 和两个乘法，最后转回输入 dtype |

所读 [融合资格列表](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp) 包含这些逐元素操作。[Metal 编译实现](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/compiled.cpp) 为图节点生成对应 dtype 的临时变量，融合不意味着所有中间值都提升到 float32、只在最终舍入。库内 [Sigmoid](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/unary_ops.h) 与 custom 的代数形式不同；[BF16 math overload](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16_math.h) 也有转换回 BF16 的步骤。

custom 使用 `e=exp(-abs(g))`；g≥0 时 sigmoid 为 `1/(1+e)`，否则为 `e/(1+e)`，指数调用 `precise::exp`。每线程处理 4 个带边界检查的逻辑元素，分别按两个输入自己的二维 strides 寻址；256 线程一组，grid 为 `ceil(M*D/4)` 个线程，输出逐元素写满。D4103 的尾部也参与计算。这里没有证明向量加载指令、最优线程组、寄存器压力或实际 dispatch 数。

## 布局处理属于完整路径

连续与隔列输入的逻辑值完全相同。隔列方式是先上传并求值完整 GPU backing，再取 `[:,::2]`；空隙分别填 3.25 和 −2.5。独立元数据 kernel 检查列 stride 为 1 或 2，并回读所有输入核对实际存储值。没有把 NumPy 非连续 view 直接上传后当作 GPU 非连续输入。

[compiled 布局处理](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/compiled.cpp) 配合上述 Metal 实现，可根据折叠后的 shape/strides 直接读取非连续输入。不能假设 compiled 会先复制成连续数组。custom 设 `ensure_row_contiguous=False` 并自行寻址；该版本的[默认连续化机制](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp)及[调用契约](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst)可解释候选，但不是本轮 binary 的 profiler 证明。本轮没有另测 custom-copy 路径。

## 数值合同与观察

M4/16GB、macOS 27.0（26A428）、Python 3.14.3、NumPy 2.4.3；默认 Metal GPU，精度环境变量未设置。未下载模型或更新软件。gate/x 同形状 `[M,D]`、同 dtype，M={1,32,512}，D={4096,4103}；两种布局、三个分布、三条路径，共 216 项。

CPU PCG64 seed=71，每个 D 顺序生成两份 `[512,D]` 正态数组。随机分布使用原值；零分布令偶数行 gate=0、奇数行 x=0，输出应全零；饱和分布令 gate 为按列线性展开的 [−80,80]，x 仍为随机值。输入先按 F32/BF16 实际存储值舍入，小 M 取前缀。布局、前缀和进程重复不是独立随机样本。

oracle 从实际输入开始，以 CPU float64 稳定 sigmoid 及乘法计算，不将参考预先舍入为输出 dtype。所有元素须满足 `abs(y-ref) ≤ atol + rtol*abs(ref)`，同时满足最大逐行 relative L2 门；shape/dtype/有限性也须通过。零参考元素和零参考行均要求精确为零。

| dtype | atol | rtol | 最大逐行 relative L2 门 |
|---|---:|---:|---:|
| F32 | 2e-6 | 3e-6 | 2e-6 |
| BF16 | 2e-4 | 0.015 | 0.01 |

所有 216 项通过，未改变 oracle 或门槛。下表 mixed error 为逐元素误差除以对应混合容差后的最大值，≤1 才通过。

| dtype / 路径 | 最大绝对误差 | 最大 mixed error | 最大逐行 relative L2 |
|---|---:|---:|---:|
| F32 / expression、library_compiled 各自 | 1.514e-5 | 0.04491 | 5.162e-8 |
| F32 / custom_strided | 1.514e-5 | 0.05510 | 6.202e-8 |
| BF16 / expression、library_compiled 各自 | 1.0 | 0.85831 | 0.003359 |
| BF16 / custom_strided | 1.0 | 0.25886 | 0.001909 |

绝对误差大于 atol 不等于违反这个混合门；108 项有此现象，仍满足逐元素混合门和逐行门。独立 CPU 审查改用 `1/(1+exp(-g))` 重算有限输入，216 项均通过，两套 float64 参考最大差 6.66e-16。expression 与 library_compiled 的 72 对保存输出精确相同，108 对连续/隔列输出精确相同。该相等关系只针对本轮输入。

custom 与两个基线都不是普遍逐位等价：两种 dtype 各 36 对比较，仅 12 个全零条件精确相同；BF16 最大路径差为 0.25。保存的“理想 sigmoid 后逐阶段舍入”和“理想结果仅最终舍入”只用于诊断，BF16 参考转换先经 float32 再做 BF16 RNE；前者不是库函数精确模拟，后者也不与 custom 全部相等。BF16 的较小 mixed error 不等于更好的模型质量，更不能把数值变化只归因于融合。

## 计时范围

仅在 216 项预检查和独立 CPU 复核通过后，顺序启动 9 个计时进程，每路径 3 个；逐轮轮换路径顺序，进程内轮换配置起点。只测 normal/D4096，两种 dtype、三个 M、两种布局。每条件 1 次 first、3 次 warmup、10 次预热后样本，共 108/324/1080 个；**1512 个在线数值检查全部通过**。

每次同步默认 stream 后，从新的函数调用计到 `mx.eval(y)` 完成。输入、compiled callable 和 custom 对象复用；输入已驻留并求值。CPU 回读、检查、保存和日志在区间外，但会影响后续状态。未控制其他应用、温度、功耗、allocator 或硬件缓存，也未刷新缓存。所读[编译文档](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/usage/compile.rst)描述的 shapeless 复用不代表每种 dtype 都不重编译；本轮 first 不是全局冷启动或纯编译耗时。

下表单位 **微秒**，测量的是主机调用至完成时间。每格为三个进程各自 10 个预热后样本中位数的 `中位数 [最小,最大]`；范围不是全部 30 个样本的极值。

| dtype | M | 输入 | expression | library_compiled | custom_strided |
|---|---:|---|---:|---:|---:|
| float32 | 1 | 连续 | 377.8 [346.4,379.9] | 388.9 [154.0,402.5] | 274.0 [165.6,280.3] |
| float32 | 1 | 隔列 | 340.5 [284.2,352.5] | 341.0 [328.6,352.0] | 165.5 [160.7,565.8] |
| float32 | 32 | 连续 | 423.5 [352.6,535.4] | 406.9 [375.0,476.2] | 525.4 [410.6,585.3] |
| float32 | 32 | 隔列 | 446.5 [353.2,591.9] | 536.5 [412.7,602.4] | 552.0 [431.5,607.4] |
| float32 | 512 | 连续 | 1634.7 [1234.4,1893.2] | 848.2 [726.7,1354.4] | 785.1 [770.9,1313.1] |
| float32 | 512 | 隔列 | 1459.1 [1409.1,1866.6] | 1045.9 [946.8,1578.2] | 1667.1 [963.0,1689.9] |
| bfloat16 | 1 | 连续 | 273.3 [161.0,334.9] | 334.1 [332.7,346.5] | 257.8 [254.0,339.8] |
| bfloat16 | 1 | 隔列 | 242.8 [238.4,261.6] | 257.5 [177.9,334.1] | 247.1 [146.5,346.2] |
| bfloat16 | 32 | 连续 | 395.6 [318.1,444.8] | 376.3 [186.1,441.0] | 403.5 [402.4,412.1] |
| bfloat16 | 32 | 隔列 | 511.9 [393.7,516.9] | 403.6 [295.6,424.6] | 412.8 [398.3,414.1] |
| bfloat16 | 512 | 连续 | 1229.3 [741.3,1341.2] | 1059.6 [604.0,1076.7] | 757.1 [740.5,884.7] |
| bfloat16 | 512 | 隔列 | 986.4 [922.0,1690.4] | 830.5 [716.8,1240.8] | 787.5 [757.9,1324.4] |

大尺寸的汇总中位数看起来倾向 compiled，但 **M512 的四组 dtype/layout 条件，每组都是 compiled 两轮快于 expression、一轮慢于 expression**。F32 连续输入的 expression/compiled 比依次为 0.911、1.927、2.605；不能只取后两轮或把汇总比值写成稳定加速。custom 对 compiled 在全部四组 M512 条件下也有轮次反转。

本次 F32/M512 连续、BF16/M512 隔列两组里，custom 三轮均比 expression 快；它没有在全部条件胜过已有 compiled helper，且 BF16 算术路线不同。M1/M32 同样没有跨配置的统一赢家。这些记录支持保留候选继续在真实热点比较，不支持固定选择阈值或模型加速倍率。没有 profiler，无法将时间差精确拆成融合、寻址、算术或主机开销。

独立 CPU 复核另检查 108 份 first 输出，全部通过同一数值门且与各自对应预检查输出精确相同。其余 1404 个成功计时输出只留在线指标，不能说后续已独立重算全部 1512 份输出。


## 使用与复验

若实际热点位于 SwiGLU，先确认模型是否调用已有 helper，再将显式表达式、compiled helper 和数值契约允许的 custom 一起比较。参数顺序、输入 dtype、输出存活期和真实布局要保持一致。只测激活不覆盖上游 gate/up GEMM、下游 down GEMM 或 MoE gather；不得直接宣称已验证 GEMM epilogue 融合、完整 FFN 或模型 tok/s。

本轮未测 FP16、混合 dtype、广播、转置、负 stride、两输入不同的步长、NaN/Inf、任意幅度或反向传播；也没有 GPU 时间戳、内存峰值、profiler、服务或模型质量证据。正确性覆盖 D4103，性能只覆盖 D4096。采用前仍需目标模型的质量和端到端性能验收。

原始记录保存脚本、原创 MSL、环境、12 份输入/CPU 参考、216 份预检查输出及全部事件；计时部分另存首次输出与所有在线指标。来源 `local-mlx-swiglu-20261007`，逻辑引用 `2026-10-07-mlx-swiglu/derived/summary.json`。原记录位于仓库外、未随公开库发布，外部读者不能仅凭本页独立复验原运行。文件留存不建立持续 custody 或正式实验资格。

相关：[融合](fusion.md)、[MLX 执行](mlx-execution.md)、[GEMM/MoE](gemm-moe.md)、[自定义双输出 RMSNorm](mlx-custom-rms.md)、[测量范围](measurement.md)。
