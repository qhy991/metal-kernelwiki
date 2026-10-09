# Custom Metal RMSNorm: two outputs and stride costs

[中文](../mlx-custom-rms.md) · English companion edition

This contract returns both `s=x+r`, rounded to input dtype, and weighted RMSNorm(s). The native baseline creates s once and evaluates both outputs together. It differs from the earlier one-output experiment.

Three routes compare native fast RMSNorm, a custom kernel with automatic row-contiguous copies, and a custom kernel addressing each input's actual strides. Disabling copies requires correct X, R and W indexing rather than assuming shared layout.

On M4/MLX 0.31.2, F32/BF16, M=1/32/512, D=4096/4103 and the tested contiguous/step-two views, all 216 numerical prechecks passed. Nine later processes produced 1512 first/warmup/timed online checks, all passing.

Performance samples used random inputs and D4096. Completion time includes required layout processing, framework work and waiting. Different dtypes, sizes and layouts produced different observations; no universal fused winner follows.

Only precheck and first outputs were saved as arrays; later successful samples retained online metrics. Do not claim independent array recomputation of all 1512 results.

No GPU timestamps, dispatch counts, profiler, buffer-donation measurement, arbitrary strides, FP16, mixed dtypes or model gain were established. Preserve both-output lifetimes and copying costs in target-model comparisons. External raw records are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-custom-metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) (official-doc)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- [mlx-normalization-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/normalization.cpp) (upstream-code)
- local-mlx-custom-rms-20261007: `2026-10-07-mlx-custom-rms/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
