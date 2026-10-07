# MLX 部署：按 prefill、decode 和内存分别选择候选

证据状态：官方文档与上游源码；核查于 2026-10-07。文档显示 MLX 0.32.3，MLX-LM 链接指向可变的 `main`。本页方法未在本机执行，不含本机测量结果。

## 适用与第一步

用于 Apple Silicon 上的 MLX / MLX-LM 文本模型部署。先确认运行的是 Metal 后端，记录芯片、GPU 核数、统一内存、macOS、框架版本、模型与量化配置、tokenizer 和聊天模板。固定 prompt、输出长度、采样、并发和缓存状态。模型加载、冷启动、预热后性能分开报告，避免把首次编译摊入某个候选而排除另一个。

先得到四组结果：短输入短输出、长输入短输出、短输入长输出、目标业务混合分布。分别报告 TTFT、prefill tok/s、decode tok/s、端到端时间与峰值内存。上游 `mlx_lm.benchmark` 已分开输出 prompt / generation 吞吐并预热，但其随机 token 和禁用 EOS 的处理只适合性能比较，不能作为答案质量依据。[基准实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py)

## 权重量化候选

权重占用或带宽压力突出时，独立比较量化格式、bits、group size。当前 affine 文档支持 2/3/4/5/6/8 bit 与 32/64/128 group size；默认 4 bit、64。量化输入至少二维，最后维度须能整除 group size。mxfp4、mxfp8、nvfp4 的格式支持不意味着目标 Apple GPU 有对应原生计算单元。[量化 API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)

保持权重为框架支持的量化表示，让量化线性层使用 `quantized_matmul`；不要在每个生成 token 前展开整套权重。[QuantizedLinear 源码](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py) 将更低位宽当作候选，比较质量、加载后内存、prefill 和 decode；小文件不自动等于低延迟。

## Prefill 分块候选

长输入峰值内存过高时搜索 `prefill_step_size`。小步长减少一次处理的 token 数和峰值内存，可能损失 prompt 处理速度；当前 README 默认 2048，不能直接当作所有模型的最优值。[长提示词说明](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations)

将步长和权重量化先分别比较，再评估交互。只保留满足质量与内存预算的候选；对入选者重复业务请求，保存中位数与离散程度。服务有延迟目标时补充 P95 TTFT/TPOT，而不是用聚合吞吐代替体验。

每个结果附上原始配置、输入标识、是否命中缓存及测量范围。若设备、模型或工具链不支持某个候选，记录不支持的原因；不要偷偷换模型、截短提示词或改变输出长度后继续排名。统一内存不足的运行单独标为容量失败，不把它的超时混入正常延迟统计。

## 未执行的入口示例

以下只用于定位已安装版本的选项；本页未运行命令：

```sh
python -m mlx_lm.benchmark --help
python -m mlx_lm.convert --help
```

flags、格式与模型支持均以安装版本为准。缓存策略见 [KV cache](mlx-kv-cache.md)，并发见 [Serving](mlx-serving.md)，准确计时与 kernel 调优见 [Execution](mlx-execution.md)。
