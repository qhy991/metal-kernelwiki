# MLX 执行：惰性计时、编译融合与 custom Metal

证据状态：官方文档；核查于 2026-10-07，页面显示 MLX 0.32.3。求值边界已有 [M4/MLX 0.31.2 限定观测](mlx-async-evaluation.md)；残差 RMSNorm 的编译/fast 候选已有 [三路限定比较](mlx-residual-rms.md)；custom Metal 双输出 RMSNorm 也已有[连续/隔列输入限定比较](mlx-custom-rms.md)。先 profile，再决定是否改 kernel。

## 测的是计算还是构图

MLX 操作先记录图；仅计 Python 函数返回时间可能没有计入 GPU 工作。微基准应预热，并求值所需输出。`print`、`.item()`、NumPy 转换也会触发求值，可能破坏批量执行。[惰性执行](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)、[eval API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html)

使用多流时同步实际工作的流；无参数的 `mx.synchronize()` 只针对默认设备的默认流，不应描述为全设备所有流屏障。[synchronize API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html) 分开保存编译/预热时间与稳态时间，整模型仍需测 TTFT、decode 和端到端延迟。

## 先用已有 fast 原语

手写 attention 优先与 `mx.fast.scaled_dot_product_attention` 比较。它支持 GQA/MQA，不应先将 K/V 复制到 query head 数；softmax 使用 float32。字符串 causal mask 采用右下对齐。`force_fused=True` 可能更慢，只是部分情况内存更小；无可用 fused kernel 时会报错。[SDPA API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) 检查非方阵 attention、cache offset、mask、head 比例与精度。

上述 `force_fused` 来自新版文档。本机 MLX 0.31.2 的 docstring 签名没有这个参数，未用调用验证其拒绝行为；先查看安装版本再生成代码。[M4 本地检查](local-mlx-m4.md) 仅验证了普通 SDPA 调用及限定形状的 oracle，没有确认后端融合或内部复制策略。

float32 输入输出并不充分说明内部计算精度。按安装版本和硬件确认数值契约，保留原任务容差；诊断构造和本机范围见 [Float32 精度边界](mlx-float32-precision.md)。

## 编译稳定子图

profile 若显示 elementwise 中间读写或许多小 dispatch，可比较 `mx.compile`。在循环外建立 compiled function；shape、dtype、输入数变化会重编译。可变状态须显式传入/返回或声明 inputs/outputs。`shapeless=True` 不能修复依赖首个 shape 的 Python 分支或 reshape 常量。[Compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) 从纯子图开始验证多 shape 和状态更新，避免盲目包住整个生成循环。

## 最后才写 custom Metal

`mx.fast.metal_kernel` 创建可能触发 JIT，应复用对象。默认 `ensure_row_contiguous=True` 可能产生输入复制；若关闭，kernel 必须正确处理 shape/strides。当前文档默认 safe math；masked softmax 依赖 `exp(-inf)=0`，不能无条件改为 relaxed/fast。[Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

本机 0.31.2 的接口没有上面新版文档的 `compile_options` 参数；[版本源码](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp)与所保留安装接口相符。[本轮双输出探针](mlx-custom-rms.md)在安装版本下验证了连续/隔列切片与 D4103 尾部，不能扩展为任意布局支持。

以连续、转置、切片、尾部、不同 dtype 和极值输入对照参考；将隐式复制计入路径。搜索 threadgroup 与每线程工作量时观察 profiler 的分配、调度、访存及占用相关证据。采用条件是正确性、代表性 shape 和端到端收益同时成立；某个孤立 kernel 更快不足以通过。所有建议均为候选生成规则，不是跨 Apple 芯片的固定调参答案。

## 记录诊断与停止条件

对每次修改写明瓶颈假设、改变的计算或访存、验证输入、观测指标和不支持的范围。若运行时间下降但答案误差超出约定容差，应退回候选，不用降低参考精度使它通过。若 profiler 显示主要时间来自其他阶段，停止微调该 kernel，回到端到端热点。线程组参数需针对目标芯片和输入搜索，不能直接复制某个 CUDA warp 或占用率配方。保留编译失败及不支持形状的原因，使下一次调优知道缺少的是能力还是性能。
