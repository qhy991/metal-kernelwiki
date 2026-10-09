---
template: doc
theme: shadcn
lang: zh
title: Metal KernelWiki
subtitle: Metal LLM 部署与 MSL 优化 · 中英文知识总览
style: 80
---

## A 现在的 skill 有什么？

这个 skill 面向 Apple Silicon、MLX 与 llama.cpp。它帮助检索部署和 kernel 优化知识。

| 内容 | 数量 |
|---|---|
| 主题 | 44 |
| 英文伴读版 | 44 |
| 来源条目 | 177 |
| 本地实验来源 | 25 |

所有主题都有英文伴读版，覆盖机制、主要结果与限制。中文保留详细表格、代码和历史记录。

英文版不是逐句翻译。来源注释保留原语言。

[English overview](index.en.html)
[English README](https://github.com/qhy991/metal-kernelwiki/blob/main/README.en.md) · [中文 README](https://github.com/qhy991/metal-kernelwiki/blob/main/README.md)

## B 如何使用？

1. 确定设备、框架、工作负载和目标指标。
2. 检索相关主题，查看一手来源。
3. 用相同数值合同比较不同候选。
4. 先验正确性，再测完整调用成本。

薄 skill 入口指向这个仓库，正文无需复制。

[SKILL.md](https://github.com/qhy991/metal-kernelwiki/blob/main/skill/metal-kernelwiki/SKILL.md) · [Workflow](https://github.com/qhy991/metal-kernelwiki/blob/main/references/en/optimization-workflow.md)

## C 部署与服务 (7 topics)

| 中文主题 | English guide | 证据来源 |
|---|---|---|
| [llama.cpp：Metal 部署、容量与阶段基准](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-deployment.md) | [llama.cpp Metal deployment: offload, capacity and PP/TG](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-deployment.md) | documented |
| [llama.cpp：Metal FA 调参与后端证据](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-metal-tuning.md) | [llama.cpp Metal tuning: GPU families, numerical gates and Tensor paths](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-metal-tuning.md) | source-reported |
| [llama.cpp：并发、前缀缓存与推测解码](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-serving.md) | [llama.cpp serving: slots, prefix reuse and speculative decoding](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-serving.md) | documented |
| [MLX 部署：按 prefill、decode 和内存分别选择候选](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-deployment.md) | [MLX-LM deployment: capacity, quantization and staged benchmarks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-deployment.md) | documented |
| [MLX 执行：惰性计时、编译融合与 custom Metal](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-execution.md) | [MLX execution: lazy graphs, compilation and custom Metal](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-execution.md) | documented |
| [MLX KV cache：前缀复用、容量和量化是不同决策](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-kv-cache.md) | [MLX KV cache: reuse, rotation, quantization and correctness](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-kv-cache.md) | source-reported |
| [MLX Serving：先检查 batch 路径，再搜索并发](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-serving.md) | [MLX serving: batching eligibility, cache budgets and speculation](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-serving.md) | source-reported |

## D MSL 用法与优化 (11 topics)

| 中文主题 | English guide | 证据来源 |
|---|---|---|
| [MSL F32 除法：函数选择、舍入与输入来源](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-f32-divide.md) | [MSL F32 division: function choice, rounding and input provenance](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-f32-divide.md) | locally-measured |
| [MSL GEMV：SIMD 分工、权重布局与归约顺序](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-gemv.md) | [MSL GEMV: lane assignment, weight layout and reduction order](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-gemv.md) | locally-measured |
| [MSL half 算术：类型提升、倒数与次正规数](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-half-arithmetic.md) | [MSL half arithmetic: promotion, reciprocal and subnormal boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-half-arithmetic.md) | locally-measured |
| [MSL 矩阵乘：SIMD-group、尾块、累加精度与 tile 复用](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-matrix.md) | [MSL matrix multiplication: fragments, tails and accumulation precision](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-matrix.md) | source-reported |
| [MSL 优化：从源码写法到待验证的性能机制](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-optimization.md) | [MSL optimization: candidate mechanisms and their costs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-optimization.md) | source-reported |
| [MSL 用法：地址空间、向量布局、同步与数值契约](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-programming.md) | [MSL programming: address spaces, vector layout and synchronization](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-programming.md) | source-reported |
| [MSL 量化 GEMV：4-bit 解包、参数复用与 affine 舍入](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-quantized-gemv.md) | [MSL quantized GEMV: 4-bit unpacking and affine rounding](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-quantized-gemv.md) | locally-measured |
| [MSL softmax：在线归一化、分块合并与 mask 契约](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax.md) | [MSL softmax: online state, split reduction and empty masks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax.md) | locally-measured |
| [MSL softmax 低精度：存储、累加与次正规输出](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax-lowp.md) | [Low-precision softmax: storage, accumulation and subnormal outputs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax-lowp.md) | locally-measured |
| [MSL softmax 计时：线程分工、分块与实际调用成本](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax-timing.md) | [MSL softmax timing: full call cost and statistical scope](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax-timing.md) | locally-measured |
| [MSL tile 优化：直接装载、跨 SIMD 共享与 BK 的代价](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-tiles.md) | [MSL tiles: direct loading, shared operands and BK tradeoffs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-tiles.md) | locally-measured |

## E 本地观测与失败边界 (16 topics)

| 中文主题 | English guide | 证据来源 |
|---|---|---|
| [llama.cpp Metal Attention：分派、量化 KV 与临时内存](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-fa-paths.md) | [llama.cpp Metal FA: dispatch, quantized KV and temporary memory](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-fa-paths.md) | source-reported |
| [llama.cpp Metal 验证：防止空跑，定位数值差异](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-validation.md) | [llama.cpp Metal validation: executed cases, numerical scope and timing](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-validation.md) | source-reported |
| [M4 本地观测：attention、量化 matmul 与 capture](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/local-mlx-m4.md) | [Local M4 baseline: SDPA, quantized matmul and tool limits](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/local-mlx-m4.md) | locally-measured |
| [MLX 求值边界：依赖链延迟与独立任务吞吐](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-async-evaluation.md) | [MLX async evaluation: dependency chains and completion boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-async-evaluation.md) | locally-measured |
| [旋转 KV cache：批处理必须保留每个请求的策略](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-lifecycle.md) | [Rotating KV cache: preserve request policy through batching](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-lifecycle.md) | locally-measured |
| [旋转缓存 mask：内容正确仍可能屏蔽有效历史](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-mask-alignment.md) | [Rotating-cache masks: align visibility with returned slots](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-mask-alignment.md) | locally-measured |
| [缓存保存与恢复：读回张量不等于能够续写](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-persistence.md) | [Cache persistence: tensor round-trip and continuation are separate gates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-persistence.md) | locally-measured |
| [自定义 Metal RMSNorm：双输出与非连续输入的成本](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-custom-rms.md) | [Custom Metal RMSNorm: two outputs and stride costs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-custom-rms.md) | locally-measured |
| [Float32 精度边界：singleton attention 恒等诊断](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-float32-precision.md) | [MLX float32 precision: singleton GQA as a diagnostic](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-float32-precision.md) | locally-measured |
| [量化线性层：反量化成本、常驻内存与 shape 一起比较](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-qmm-path-comparison.md) | [Quantized linear layers: packed execution, decode cost and resident dense memory](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-qmm-path-comparison.md) | locally-measured |
| [量化 KV attention：量化损失与多 token 执行错误](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-quantized-attention.md) | [Quantized KV attention: isolate QK, softmax and PV](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-quantized-attention.md) | locally-measured |
| [MLX 边界探针：异步依赖、singleton GQA、多行 qvm 与缓存生命周期](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-regression-probes.md) | [MLX boundary probes: observed results and remaining scope](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-regression-probes.md) | experimental |
| [残差 RMSNorm：编译、专用原语与舍入顺序一起比较](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-residual-rms.md) | [Residual RMSNorm: expressions, compile and fast primitives](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-residual-rms.md) | locally-measured |
| [RoPE 合并 Q/K：位置、布局与批量解码先过数值门](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-rope-qk.md) | [Joint Q/K RoPE: positions, layout and numerical gates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-rope-qk.md) | locally-measured |
| [SwiGLU：已有编译函数、显式表达式与自定义 Metal](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-swiglu.md) | [SwiGLU: existing compiled helpers and custom Metal arithmetic](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-swiglu.md) | locally-measured |
| [量化矩阵乘的数值与性能路径](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/quantized-matmul-validation.md) | [Quantized matmul: separate quantization loss, decoding and execution error](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/quantized-matmul-validation.md) | source-reported |

## F 机制、硬件与测量 (10 topics)

| 中文主题 | English guide | 证据来源 |
|---|---|---|
| [Attention：prefill、decode 与 GQA](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/attention.md) | [Attention: separate prefill, decode and GQA paths](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/attention.md) | inferred |
| [融合与提交：小算子和 command buffer 开销](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/fusion.md) | [Fusion and submission: establish the source of overhead](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/fusion.md) | inferred |
| [GEMM、GEMV、Split-K 与 MoE gather](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/gemm-moe.md) | [GEMM, GEMV and MoE: choose algorithms for actual shapes](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/gemm-moe.md) | source-reported |
| [测量与瓶颈诊断：TTFT、prefill、decode、服务延迟](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/measurement.md) | [Measurement: turn slow inference into testable bottlenecks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/measurement.md) | inferred |
| [Metal 能力与部署边界](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-capabilities.md) | [Metal capabilities and deployment boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-capabilities.md) | documented |
| [Metal 内存、线程组与同步](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-memory-threadgroups.md) | [Metal memory, threadgroups and synchronization](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-memory-threadgroups.md) | documented |
| [Metal LLM 性能诊断与证据](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-profiling.md) | [Metal LLM profiling and evidence](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-profiling.md) | documented |
| [TensorOps、量化与融合候选](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-tensors.md) | [TensorOps, quantization and fusion candidates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-tensors.md) | documented |
| [权重、KV 与低位宽：压缩和加速分开验证](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/quantization.md) | [Weights, KV and low precision: choose compression boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/quantization.md) | inferred |
| [有限工具环境：运行时验证与 GPU 计时](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/runtime-measurement.md) | [Runtime validation and GPU timing with limited tools](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/runtime-measurement.md) | documented |

## G 证据能说明什么？

| Label | 含义 |
|---|---|
| documented | 官方说明的机制 |
| source-reported | 上游实现或作者报告 |
| inferred | 基于分析的推断 |
| experimental | 待验证候选 |
| locally-measured | 限定设备和版本的本机观测 |

一页可以包含多种证据。采用结论前，应查看对应范围。

原语通过不等于模型质量或部署收益已验收。导出 trace 不等于 profiler 已解析。

失败结果保留在记录中。原始实验存于仓库外，尚未公开。

[Catalog](https://github.com/qhy991/metal-kernelwiki/blob/main/data/catalog.json) · [Provenance](https://github.com/qhy991/metal-kernelwiki/blob/main/PROVENANCE.en.md)

## H 检索、安装与维护

~~~bash
./mwiki query "MSL reduction precision" --lang en
./mwiki get msl-f32-divide --lang en --follow-sources
./mwiki get measurement --lang zh
./mwiki validate
python3 scripts/install_skill.py
~~~

检索只用 Python 标准库，不启动 GPU。跟随来源只显示元数据。

主题链接打开 GitHub 的 Markdown 页面。总览可直接在本机打开，无需服务。

[Maintenance](https://github.com/qhy991/metal-kernelwiki/blob/main/MAINTENANCE.en.md) · [Examples](https://github.com/qhy991/metal-kernelwiki/blob/main/references/en/examples.md) · [Rebuild HTML](https://github.com/qhy991/metal-kernelwiki/blob/main/docs/README.md)
