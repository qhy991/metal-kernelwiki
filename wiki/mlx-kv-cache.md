# MLX KV cache：前缀复用、容量和量化是不同决策

证据状态：官方说明与上游源码；核查于 2026-10-07。`main` 可变，缓存类型和功能组合必须与安装版本核对。另有不加载模型的 [MLX-LM 0.31.3 本地 API 观察](local-mlx-m4.md)，不构成缓存质量或性能结论。

## 先识别缓存类型

检查模型每层实际 cache 类、token offset、容量、字节占用及是否能 trim/quantize。上游 `make_prompt_cache` 优先采用模型的 `make_cache`；只有默认路径依据 `max_kv_size` 创建旋转缓存。因此不能声称设置一个参数就约束所有模型的缓存。[cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)

## 重复上下文：优先测前缀复用

相同长上下文上的多次提问可保存并读取 prompt cache。当前 `mlx_lm.cache_prompt` 与 `--prompt-cache-file` 将已缓存 token 作为新 prompt 的前缀，模型信息从缓存读取。[官方用法](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations)

验证模型权重、tokenizer、模板、位置与 token 前缀一致；文本相似不是可复用条件。分别测全命中、部分命中、未命中，记录实际跳过的 token、TTFT、缓存内存和读取成本。多轮生成会推进状态；保留可复用基准前缀，避免把已被某条分支更新的缓存误用于另一条分支。

文件读回还需验证恢复后能否续写。本机 MLX-LM 0.31.3 的批量旋转缓存可读回相同 K/V，却存在旋转标志错误和缺字段续写异常；其他三类缓存的指定对照通过。完整版本、上游格式变化与边界见 [缓存保存恢复](mlx-cache-persistence.md)，不能将一次加载成功或 PR 合入当作完整兼容证明。

## 内存不足：分别评估旋转和量化

`max_kv_size` 通过丢弃较早信息限制默认旋转缓存；小容量会牺牲长上下文质量。所见默认路径保留最初 4 个 token，且 `RotatingKVCache.to_quantized` 仍拒绝执行；不能假定旋转缓存与 KV 量化可组合。[容量说明](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations)、[缓存实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)

`generate_step` 另有 `kv_bits`、`kv_group_size`、`quantized_kv_start`。所见默认起点为 5000，`kv_bits=None` 不量化；达到 offset 后尝试调用相应 cache 的转换方法。[量化入口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) 测试必须跨过触发点，确认哪些层实际转换，记录转换瞬间延迟、KV 字节与 decode 性能。方法存在也不保证组合实现可用。

版本差异已实际出现：本机 MLX-LM 0.31.3 的 `generate_step` 默认起点为 **0**。同次合成 API 检查确认旋转缓存量化抛出 NYI，模型自有 `make_cache` 也优先于传入的最大容量。见 [原始记录说明](local-mlx-m4.md)。不要把本段的任一默认值当作跨版本常数。

## 批处理：先检查保留策略能否跨 merge/extract

本机 MLX-LM 0.31.3 的直接合并探针观察到 keep=4 在 extract 时变为 0，跨容量后丢失前缀；keep=0 对照通过。模型 make_cache 的另一路入口已有明确拒绝，不能把直接合并行为扩大成所有服务路径。完整输入、对照、失败和未测范围见 [缓存生命周期](mlx-cache-lifecycle.md)。吞吐候选先保留各请求语义，不能把改 keep=0 当等价修复。

独立 keep=0 探针又发现：不等长左填充 batch 旋转后切回 3-token 追加，缓存内容与 offset 正确仍可能使用错误 mask。90 项检查有 4 项失败且影响零 Q attention，详见 [mask 对齐](mlx-cache-mask-alignment.md)。窗口、padding 和单步/多步切换应与返回 K/V 一起验证；此结果未覆盖服务或模型。

## 自定义缓存：避免每 token 复制历史

若 profiler 显示分配开销与核间空闲，比较逐步 concatenate 和分块预分配加 slice update。前者每步复制历史并改变 buffer 大小，后者可摊销增长。[Fast KV Cache](https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html) 该文档的“256 倍数启用 cuDNN fused attention”是 CUDA 条件，不是 Metal 规则。

验证跨容量边界、扩容、trim、前缀分叉、长距离检索与多轮任务；对旋转或量化候选用业务质量门槛验收。将 prompt 复用、容量、KV 量化与权重量化分开记录，不用一个“cache enabled”概括。

保留无优化缓存作为外部对照，核查缓存后的输出或 logits 在约定容差内。若目标是节省容量，报告缓存自身字节与进程峰值两项，区分缓存减少和临时张量增长。遇到混合注意力或其他状态缓存时，逐类记录覆盖范围，不将某层成功转换写成整模型已经量化。
