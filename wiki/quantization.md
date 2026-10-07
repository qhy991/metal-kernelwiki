# 权重、KV 与低位宽：先分清压缩了什么

适用于容量或带宽已成为限制的部署。以下为文档机制加工程推断，没有目标设备速度结论。

## 三条独立决策轴

| 对象 | 可能收益 | 必须付出的代价/检查 |
|---|---|---|
| 权重量化 | 模型常驻内存与读取字节减少 | 格式、scale/group、解码成本、质量；同位宽不等于同算法 |
| KV 量化 | 随 context/batch 增长的缓存变小 | 每步量化/反量化、FA 兼容、长上下文误差 |
| 激活/矩阵低精度 | 可能使用不同计算路径 | GPU family、OS、MSL/SDK、实际框架 dispatch |

MLX 的 quantized matmul 接受打包权重、group scales/biases 与模式；位宽、group size、transpose 必须匹配转换时的约定。GGUF 的 Q4 类格式与 MLX affine 4-bit 不可仅凭名称互换。具体支持查 [MLX quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) 与 [quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)，Metal Tensor 路径另读 [能力页](metal-tensors.md)。

## 预算方法

工程估算，仅适用于标准 attention cache，K/V 同宽同 dtype 时：

`KV bytes ≈ 2 × batch × layers × cached_tokens × KV_heads × head_dim × bytes_per_element`

异构层要逐层求和，K/V dtype 或维度不同要分开算；量化还有 scale、zero/bias、padding 与 allocator 开销。滑动窗口、共享 prefix、recurrent state 和混合模型不直接套这个式子。GQA 用 KV head 数；用 query head 数会高估。MoE 的 active parameters 影响每步计算，不能把它当成驻留全部专家权重的容量。

总预算还包括 activation/scratch、logits、草稿模型、host cache 和系统余量。RSS 与 Metal 分配可能重叠，不要直接相加。先缩减超出用户需要的 context/concurrency，再比较精度候选。

llama.cpp 的 [FA 临时分配示例](llamacpp-fa-paths.md)按固定源码推导：某单层形状的 Q4_0 逻辑 KV 为36 MiB，FA输出的F16 scratch项仍为128 MiB。该项即使未执行预转换也被计入分配需求；这是源码容量计算，不是本机峰值，也不能直接按层数相乘。

## 如何选择候选

固定同一模型/tokenizer，保留较高精度参考；从一种权重量化开始。分别测 prefill 与 decode，避免 decode 字节减少掩盖 prefill 解码计算增加。之后单独加入 KV 量化，测内存、长文本检索、困惑度或任务准确率；再测组合。

MLX 的 rotating KV、model-specific cache、batch 资格会限制组合，见 [KV 页](mlx-kv-cache.md) 与 [服务页](mlx-serving.md)。llama.cpp 当前量化 V 需要 FA，格式与头维度仍应以实际后端为准，见 [部署页](llamacpp-deployment.md)。不能通过跳过质量门槛或静默截断上下文来“提速”。

采用时报告具体格式、group、模型覆盖层、额外峰值内存和质量变化。更小 checkpoint、格式 API 存在、某个 M5 演示更快，都不足以证明这台 Mac 上更快。
