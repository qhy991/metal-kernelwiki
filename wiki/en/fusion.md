# Fusion and submission: establish the source of overhead

[中文](../fusion.md) · English companion edition

Use fusion candidates when traces show short dispatches, CPU encoding, waits or intermediate traffic. Fewer kernels are not a universal speed rule.

At the expression level, compare `mx.compile` for stable pure subgraphs. Reuse functions outside token loops and inspect shape/dtype specialization and captured mutable state. At the kernel level, consider residual RMSNorm, RoPE/cache writes or GEMM epilogues while preserving dependencies and dtypes. At submission level, reuse resources/pipelines and remove unnecessary readback; retain required queue/encoder ordering.

[Residual RMSNorm](mlx-residual-rms.md) passed its F32/BF16 numerical gates but showed small-M rank reversals and process variability. MLXv0.31.2's generic fusion list excludes Reduce/RMSNorm primitives; compilation around a reduction does not establish one dispatch. [Dual-output custom RMSNorm](mlx-custom-rms.md) includes residual output and layout conversion costs; noncontiguous M512 results did not establish copy shares or model benefits.

[Merged Q/K RoPE](mlx-rope-qk.md) retained 12 configurations where separate calls passed and merged calls failed, plus shared long-offset failures. It never entered timing. [SwiGLU](mlx-swiglu.md) retained 216 prechecks and 1512 online timing checks, process rank reversals and BF16 arithmetic differences.

Custom Metal's default row-contiguous handling can copy inputs. Disabling it requires actual stride addressing; either choice belongs in full-path timing. [Custom-kernel documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

[Issue #4521](https://github.com/ml-explore/mlx/issues/4521) motivates studying command-buffer limits, not blindly adopting variables or thresholds. Inspect installed behavior, overlap, memory and stability first.

If kernel time falls but request latency does not, inspect another critical path. If fusion reduces dispatches while increasing register/spill/cache pressure, reconsider granularity. If only first-call cost changes, report compile/cache benefit separately. Adoption needs original-input correctness, tails, sustained runs and real requests.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) (official-doc)
- [mlx-custom-metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) (official-doc)
- [apple-command-buffers](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html) (official-doc)
- [apple-resource-sync](https://developer.apple.com/documentation/metal/resource-synchronization) (official-doc)
- [mlx-command-buffer-issue](https://github.com/ml-explore/mlx/issues/4521) (issue)
- [mlx-compile-fusion-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp) (upstream-code)
- local-mlx-residual-rms-20261007: `2026-10-07-mlx-residual-rms/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-custom-rms-20261007: `2026-10-07-mlx-custom-rms/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-rope-qk-20261007: `2026-10-07-mlx-rope-qk/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-swiglu-20261007: `2026-10-07-mlx-swiglu/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-lm-swiglu-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/activations.py) (upstream-code)

[Shared source catalog](../../data/catalog.json)
