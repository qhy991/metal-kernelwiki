---
template: doc
theme: shadcn
lang: en
title: Metal KernelWiki
subtitle: English guide to the skill and its knowledge library
style: 80
---

## A What is in this skill?

The skill retrieves deployment and kernel knowledge for Apple Silicon, MLX and llama.cpp.

| Item | Count |
|---|---|
| Topics | 44 |
| English companions | 44 |
| Registered sources | 177 |
| Local-experiment sources | 25 |

English guides cover mechanisms, main observations and limits. Chinese records retain full tables, code and history.

This is a companion edition, not a line-by-line translation. Shared source notes retain their original language.

[中文 / bilingual overview](index.html)
[English README](https://github.com/qhy991/metal-kernelwiki/blob/main/README.en.md) · [中文 README](https://github.com/qhy991/metal-kernelwiki/blob/main/README.md)

## B How does the skill work?

1. Identify the device, framework, workload and target metric.
2. Retrieve the relevant topics and inspect their sources.
3. Compare distinct candidates under the same numerical contract.
4. Validate correctness, then measure full call costs.

The thin skill points to this repository. It does not duplicate the wiki.

[SKILL.md](https://github.com/qhy991/metal-kernelwiki/blob/main/skill/metal-kernelwiki/SKILL.md) · [Workflow](https://github.com/qhy991/metal-kernelwiki/blob/main/references/en/optimization-workflow.md)

## C Deployment and serving (7 topics)

| English topic | 中文 record | Evidence basis |
|---|---|---|
| [llama.cpp Metal deployment: offload, capacity and PP/TG](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-deployment.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-deployment.md) | documented |
| [llama.cpp Metal tuning: GPU families, numerical gates and Tensor paths](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-metal-tuning.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-metal-tuning.md) | source-reported |
| [llama.cpp serving: slots, prefix reuse and speculative decoding](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-serving.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-serving.md) | documented |
| [MLX-LM deployment: capacity, quantization and staged benchmarks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-deployment.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-deployment.md) | documented |
| [MLX execution: lazy graphs, compilation and custom Metal](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-execution.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-execution.md) | documented |
| [MLX KV cache: reuse, rotation, quantization and correctness](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-kv-cache.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-kv-cache.md) | source-reported |
| [MLX serving: batching eligibility, cache budgets and speculation](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-serving.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-serving.md) | source-reported |

## D MSL programming and optimization (11 topics)

| English topic | 中文 record | Evidence basis |
|---|---|---|
| [MSL F32 division: function choice, rounding and input provenance](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-f32-divide.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-f32-divide.md) | locally-measured |
| [MSL GEMV: lane assignment, weight layout and reduction order](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-gemv.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-gemv.md) | locally-measured |
| [MSL half arithmetic: promotion, reciprocal and subnormal boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-half-arithmetic.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-half-arithmetic.md) | locally-measured |
| [MSL matrix multiplication: fragments, tails and accumulation precision](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-matrix.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-matrix.md) | source-reported |
| [MSL optimization: candidate mechanisms and their costs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-optimization.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-optimization.md) | source-reported |
| [MSL programming: address spaces, vector layout and synchronization](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-programming.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-programming.md) | source-reported |
| [MSL quantized GEMV: 4-bit unpacking and affine rounding](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-quantized-gemv.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-quantized-gemv.md) | locally-measured |
| [MSL softmax: online state, split reduction and empty masks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax.md) | locally-measured |
| [Low-precision softmax: storage, accumulation and subnormal outputs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax-lowp.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax-lowp.md) | locally-measured |
| [MSL softmax timing: full call cost and statistical scope](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-softmax-timing.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-softmax-timing.md) | locally-measured |
| [MSL tiles: direct loading, shared operands and BK tradeoffs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/msl-tiles.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/msl-tiles.md) | locally-measured |

## E Local observations and failure boundaries (16 topics)

| English topic | 中文 record | Evidence basis |
|---|---|---|
| [llama.cpp Metal FA: dispatch, quantized KV and temporary memory](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-fa-paths.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-fa-paths.md) | source-reported |
| [llama.cpp Metal validation: executed cases, numerical scope and timing](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/llamacpp-validation.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/llamacpp-validation.md) | source-reported |
| [Local M4 baseline: SDPA, quantized matmul and tool limits](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/local-mlx-m4.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/local-mlx-m4.md) | locally-measured |
| [MLX async evaluation: dependency chains and completion boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-async-evaluation.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-async-evaluation.md) | locally-measured |
| [Rotating KV cache: preserve request policy through batching](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-lifecycle.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-lifecycle.md) | locally-measured |
| [Rotating-cache masks: align visibility with returned slots](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-mask-alignment.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-mask-alignment.md) | locally-measured |
| [Cache persistence: tensor round-trip and continuation are separate gates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-cache-persistence.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-cache-persistence.md) | locally-measured |
| [Custom Metal RMSNorm: two outputs and stride costs](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-custom-rms.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-custom-rms.md) | locally-measured |
| [MLX float32 precision: singleton GQA as a diagnostic](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-float32-precision.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-float32-precision.md) | locally-measured |
| [Quantized linear layers: packed execution, decode cost and resident dense memory](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-qmm-path-comparison.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-qmm-path-comparison.md) | locally-measured |
| [Quantized KV attention: isolate QK, softmax and PV](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-quantized-attention.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-quantized-attention.md) | locally-measured |
| [MLX boundary probes: observed results and remaining scope](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-regression-probes.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-regression-probes.md) | experimental |
| [Residual RMSNorm: expressions, compile and fast primitives](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-residual-rms.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-residual-rms.md) | locally-measured |
| [Joint Q/K RoPE: positions, layout and numerical gates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-rope-qk.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-rope-qk.md) | locally-measured |
| [SwiGLU: existing compiled helpers and custom Metal arithmetic](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/mlx-swiglu.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/mlx-swiglu.md) | locally-measured |
| [Quantized matmul: separate quantization loss, decoding and execution error](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/quantized-matmul-validation.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/quantized-matmul-validation.md) | source-reported |

## F Mechanisms, hardware and measurement (10 topics)

| English topic | 中文 record | Evidence basis |
|---|---|---|
| [Attention: separate prefill, decode and GQA paths](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/attention.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/attention.md) | inferred |
| [Fusion and submission: establish the source of overhead](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/fusion.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/fusion.md) | inferred |
| [GEMM, GEMV and MoE: choose algorithms for actual shapes](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/gemm-moe.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/gemm-moe.md) | source-reported |
| [Measurement: turn slow inference into testable bottlenecks](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/measurement.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/measurement.md) | inferred |
| [Metal capabilities and deployment boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-capabilities.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-capabilities.md) | documented |
| [Metal memory, threadgroups and synchronization](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-memory-threadgroups.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-memory-threadgroups.md) | documented |
| [Metal LLM profiling and evidence](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-profiling.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-profiling.md) | documented |
| [TensorOps, quantization and fusion candidates](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/metal-tensors.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/metal-tensors.md) | documented |
| [Weights, KV and low precision: choose compression boundaries](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/quantization.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/quantization.md) | inferred |
| [Runtime validation and GPU timing with limited tools](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/en/runtime-measurement.md) | [中文](https://github.com/qhy991/metal-kernelwiki/blob/main/wiki/runtime-measurement.md) | documented |

## G What does the evidence establish?

| Label | Meaning |
|---|---|
| documented | Documented mechanism |
| source-reported | Upstream implementation or author report |
| inferred | Analysis-based inference |
| experimental | Candidate awaiting validation |
| locally-measured | Bounded local device/version observation |

A page may contain several evidence bases. Read the scope of each claim.

Primitive success does not establish model quality or deployment speed. Exporting a trace does not establish profiler analysis.

Failures remain in the records. Raw experiments are external and unpublished.

[Catalog](https://github.com/qhy991/metal-kernelwiki/blob/main/data/catalog.json) · [Provenance](https://github.com/qhy991/metal-kernelwiki/blob/main/PROVENANCE.en.md)

## H Retrieve, install and maintain

~~~bash
./mwiki query "MSL reduction precision" --lang en
./mwiki get msl-f32-divide --lang en --follow-sources
./mwiki get measurement --lang zh
./mwiki validate
python3 scripts/install_skill.py
~~~

Retrieval uses Python's standard library and starts no GPU work. Source following prints metadata only.

Topic links open rendered Markdown on GitHub. This overview also opens locally without a server.

[Maintenance](https://github.com/qhy991/metal-kernelwiki/blob/main/MAINTENANCE.en.md) · [Examples](https://github.com/qhy991/metal-kernelwiki/blob/main/references/en/examples.md) · [Rebuild HTML](https://github.com/qhy991/metal-kernelwiki/blob/main/docs/README.md)
