# MSL 优化：从源码写法到待验证的性能机制

[English companion](en/msl-optimization.md)

适用：已定位到逐元素 epilogue、归约或小 kernel 热点，需要修改 MSL。先读 [地址空间、对齐与同步](msl-programming.md)。以下以 **MLX v0.31.2 源码**和 Apple 官方资料解释候选；没有为这些参数组合测得通用最优值，也不把历史 Apple 演示数字当作 M4 LLM 收益。

## 每线程多做一点，先看线程数和访问布局

[`unary_v<T,U,Op,N>`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/unary.h)每线程处理 N 个元素，完整块走固定小循环，尾部单独检查。`unary_g` 则先根据 shape/strides 解出物理位置。配套 [`WorkPerThread<U>`](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/utils.h) 默认 `8/sizeof(U)`，是这个版本的实现策略。

机制推断：把索引开销摊到几个元素上，可给编译器更多固定工作；同时减少线程数、增加临时值，可能损失并行度或增加 spill。对于 SwiGLU/activation 可比较 N=1/2/4/8，但保持表达式、dtype 舍入点及总输入不变。源码里的标量循环不证明生成向量 load，教程里的 `packed_float4` 也不保证比该循环快。

连续输入与通用 stride 路径应各有明确合同；默认连续化可能先复制，关闭它则由 kernel 自己寻址。计入实际复制与下游消费，并测短输入、整块、N±1 尾部和真实 GPU view。Apple 的 [M1 Pro/Max compute 演讲](https://developer.apple.com/videos/play/tech-talks/10580/)还讨论有界 signed 循环索引对编译器的帮助；先证明不会溢出，不为“可能向量化”把所有地址盲改成 int。

## 少一次读取，可能换来更长的寄存器生命周期

[MLX RMSNorm](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/rms_norm.metal)使用线程内 float 累加、`simd_sum`、threadgroup partial 和第二级归约。forward 输出阶段重读输入，并在乘权重前显式转换回 T。某些 VJP 分支保留局部数组，另一些循环分支重读；它们属于不同算法/规模，不能当成同一合同下的 A/B。

| 候选 | 可能减少什么 | 必须检查的代价 |
|---|---|---|
| 在 thread 数组保留中间值 | 后续输入读取或重复计算 | live registers、动态索引、spill、并行度 |
| 在 threadgroup 缓存共享 tile | 多个线程的重复读取 | 初始化、barrier、共享容量、复用次数 |
| 合并 epilogue | 中间 buffer 写回/读取、提交 | 同时存活的值、输出合同、尾部和数值舍入 |
| 较小线程组或 tile | 单组资源与负载不均 | 更多组/重复工作、归约与同步开销 |

这是候选因果分析，不能从“少一个源码 load”直接推出更快。局部数组不保证完全放在寄存器；threadgroup staging 只有在实际复用足以支付开销时才可能有利，也不能照搬 CUDA 的软件流水线。tensor/MPP 路线另见 [Metal tensors](metal-tensors.md)。

[host 分派](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/normalization.cpp)先处理布局，再按行宽选择普通/looped kernel 和线程组。优化应比较相同输出存活期、相同 dtype 与相同布局；[已有双输出 RMSNorm 观察](mlx-custom-rms.md)说明是否连续化、是否保留 residual 都会影响公平比较。

## 分清四种“常量”

| 写法/接口 | 固定的时间与用途 |
|---|---|
| `constant float& eps` | 运行时绑定的只读数据；地址空间限定不消除动态输入。 |
| `constexpr` 或模板整数/类型 | 编译期已知，可用于小循环、局部数组、dtype 特化。 |
| `constant bool has_bias [[function_constant(0)]]` | 创建 specialized function 时设定，用于结构开关。 |
| MLX `template=[('T',dtype),('N',4)]` | MLX 提供的模板特化接口；不等于任意 function-constant 绑定接口。 |

Apple [function specialization](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants)与 [WWDC23 编译优化](https://developer.apple.com/videos/play/wwdc2023/10127/)说明常量可裁掉关闭的分支；代价是 pipeline 变体及编译时间。更新 [`MTLFunctionConstantValues`](https://developer.apple.com/documentation/metal/mtlfunctionconstantvalues) 不会回改已有 function。以少量稳定的 `HAS_BIAS/HAS_RESIDUAL`、每线程工作量建立可复用变体，避免无依据地为每个 token 长度建一个 pipeline。

真实源码例子是 RMSNorm **VJP** 的 `has_w [[function_constant(20)]]`：host 设置 index 20，并区分 `_w/_now` cache key。它说明 shader/host 必须共同维护特化与缓存身份，**不说明 forward 推理能省掉权重乘法**。禁用分支后的可选输入访问、tail fallback 和输出形状仍要用共同 oracle 检查。

矩阵场景进一步看 [公开 SIMD-group API 与 Steel tile 复用](msl-matrix.md)：手工 lane fragment 映射属于实现细节，FP16 buffer、矩阵/累加类型和输出 dtype 分别决定数值行为，不能把降低中间精度当成无语义变化的优化。

[MSL tile 比较](msl-tiles.md)进一步给出direct load、staging、四SIMD共享与BK8/32的原创代码片段和M4有限测量。扩大BK减少外层同步但增加scratch及K-tail工作；完成时间发生排序反转，不能从“少读、少barrier”直接挑选赢家。

[GEMV的SIMD分工与归约顺序](msl-gemv.md)给出F16存储/F32计算的具体失败：8lane同号局部和产生明显抵消误差，相邻K配对候选在同门内通过。工作分配也属于数值设计，不能只按连续访存或更少barrier选方案。

[量化GEMV的参数提取](msl-quantized-gemv.md)比较word与group分工、逐项FMA与bias factoring。后者减少源码affine表达式，却在合成抵消输入上未过相同数值门；记录解码和算术误差后才能进入性能选择，不能把更少运算直接当作可采用优化。

[MSL softmax](msl-softmax.md)比较三遍重读、在线统计和三阶段分块合并，说明空块neutral与跨SIMD初始化；同门数值通过与读次数、指数函数精度、实际速度分别记录。[后续softmax计时](msl-softmax-timing.md)保留长短行的不同表现和两种统计量的排序反转，不用单次最小值选择通用赢家。MLX API的precise参数也不能直接解释成MSL precise::exp。

[低精度softmax](msl-softmax-lowp.md)把输入存储、float统计和输出窄化分开：half长行全零与次正规输出保留可以同时出现在不同计算路线，不能只看输出dtype判断原因，也不能用精度提升恢复已经量化掉的logits信息。

[half表达式边界](msl-half-arithmetic.md)解释如何显式选择float倒数再窄化，并区分MSL算术FTZ、转换规则和源码编译选项；sizeof只能观察表达式宽度。独立后继28项通过数值门，同时记录F32差异被half窄化掩盖的情形；保留次正规值与排除有限FTZ模型均不构成全局硬件或速度结论。

[F32除法与输入来源](msl-f32-divide.md)直接对比普通除法、fast和precise，区分质量门、精确舍入、ULP及编译模式。half提升后与同值float输入出现不同结果；独立后继先在GPU完成F32转换后再用共享consumer除法，结果与CPU-F32对照匹配RNE。该应用可观察边界仍未定位指令或编译原因，也没有速度结论。


## 调参要回到证据

实际 dispatch 的线程组大小、pipeline 报告的合法上限、编译 descriptor 的 `maxTotalThreadsPerThreadgroup` 是相关但不同的量。不能把 MLX 的 `threadgroup=(...)` 当作已经修改了编译器资源提示；也不能把最大合法值当最快值。[Apple 线程组指南](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

对每个候选保留：实际 MSL/模板值与输入布局、外部数值结果、首次调用与预热后完成时间、可取得的指令/访存/寄存器/spill/同步证据。低 occupancy 只有结合瓶颈才有意义；M3/M5 工具的计数器与动态资源行为有代际条件。[Apple9 profiling](https://developer.apple.com/videos/play/tech-talks/111374/)、[M5 profiling](https://developer.apple.com/videos/play/tech-talks/111431/)

本轮 [MSL 用法检查](msl-programming.md)只验证有限 F32 示例，未测这些优化维度的速度。缺少 profiler 时可以保留完成时间的有限观察，不能据此声称向量指令、无 spill 或最优 tile。最终仍需同一模型下的 prefill/decode、KV depth、质量及服务延迟验证。
