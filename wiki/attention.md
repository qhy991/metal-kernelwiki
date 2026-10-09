# Attention：prefill、decode 和 GQA 分开选路径

[English companion](en/attention.md)

适用于 profiler 已定位 attention，或长上下文带来显存/延迟压力。先用框架 fast primitive 比较，再考虑手写。候选默认待验证；有限 SDPA 数值与主机调用计时见 [M4 本地记录](local-mlx-m4.md)，没有端到端收益结论。

## 保持数学语义

MLX 的 fast SDPA 支持 MHA/GQA/MQA；输入是 `[B,H,T,D]`，GQA/MQA 的 K/V 不要先扩成 query head 数。softmax 在 float32 计算。`mask="causal"` 使用右下对齐，这对 `Tq != Tkv` 的 decode/chunked prefill 很关键；`force_fused` 是选择路径的实验变量，可能更慢或报不支持。[SDPA API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)

该参数来自所读新版文档，本机 MLX 0.31.2 签名未出现它。调用前核对安装版本；不能从普通 SDPA 数值通过推断强制融合支持或具体 kernel dispatch。

配方：`mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask="causal")`。
这是接口示意，未执行；`q/k/v` 必须来自已校验形状的模型。分离 attention sinks、滑动窗口、padding mask 和 positional offset；不要为方便而改变原模型 mask。

单 key、无 sinks 的输出恒等于 V 可作为诊断关系。M4 的 48 组 singleton 对照见 [精度边界](mlx-float32-precision.md)；这不覆盖多 key 归约，也不证明任意 float32 内部计算路径。

位置编码也要覆盖 B>1 的单 token 和真实 offset。[RoPE/QK 本地验证](mlx-rope-qk.md)在 MLX 0.31.2 观察到标量 offset 与布局相关的批量失败，以及两路共同的长位置误差；只测 offset=0 或只比合并前后输出会漏掉问题。该页列出上游修复版本线索，尚无修复版本或模型复验。

## 结构不同的优化候选

- Prefill：tile 化融合 softmax 与 value 聚合，避免写出完整 score/probability 矩阵。需要较多计算与复用，调 tile 时同时检查 threadgroup memory 和寄存器压力。
- Decode：查询少而 KV 长，比较不同 KV 分块与并行归约，不能把 prefill GEMM tile 原样套用。额外 combine pass、临时输出与 dispatch 必须计入总耗时。
- GQA：复用同组 K/V；收益随 head 比例、序列长度、dtype 改变，预复制 KV 可能抵消带宽优势。

这些是机制层面的候选。实际 llama.cpp FA-vec 已有 `(Q,NE)` 搜索与热漂移检查，可参考 [调优页](llamacpp-metal-tuning.md)；不要把别的 GPU family 调参表视为本机最优。

llama.cpp 的 [固定快照路径与 scratch](llamacpp-fa-paths.md)显示：预反量化、普通 vec、Tensor 与稀疏 vec 有不同 gate。内部 `n_kv_max` 是每行有限 mask 项数的上界，低报会截断索引，不是可随意压小的性能参数。该页将源码结论与旧本机 binary 的45项有限对照分开记录。

[MSL softmax](msl-softmax.md)进一步给出在线normalizer、跨SIMD与split256状态合并的原创实现，明确空行/空块策略并保存中间状态。它只验证给定logits到概率，不是QK/PV或完整FlashAttention验证；减少一遍源码读取也不等于已测端到端收益。

## 正确性矩阵

独立 reference 用高精度 score/softmax 与约定 tolerance；覆盖 Q 长度 1/多 token，K 长度小/大/尾部，GQA 比例，padding/causal/window/sinks，cache offset 与不同 dtype。核查 fully-masked row 的约定、`-inf` 和极端 logits。`fast/relaxed` math 不能默认保持 masked softmax 特殊值语义，见 [MLX custom Metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)。

profile 证据应能区分 attention 内核、KV 更新复制与提交间隙；报告临时内存和端到端变化。KV 量化、窗口截断和 prefix cache 分别改变存储、语义与复用条件，不能混称为 FlashAttention 优化。
