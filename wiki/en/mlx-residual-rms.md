# Residual RMSNorm: expressions, compile and fast primitives

[中文](../mlx-residual-rms.md) · English companion edition

Define output ownership before evaluating fusion. This probe returned only normalized `x+r`, not the residual itself. A model needing both has a different allocation and lifetime contract.

M4/MLX 0.31.2 compared an ordinary expression, `mx.compile` and `mx.fast.rms_norm` on common-dtype X/R/W, float32/BF16, M=1/32/512, D=4096 and epsilon 1e-5. The oracle used actual stored inputs and explicit residual/normalization rounding.

All 90 prechecks and 756 timed online checks passed their predefined gates. Nine benchmark processes rotated route and condition order, keeping compiled callables alive and warming tested signatures.

Host-completion timings varied substantially between processes. They include framework and synchronization costs; they do not establish one fused kernel, device time, allocator behavior or a model speedup.

Use fast primitives and compiled expressions as distinct candidates. Preserve output dtype, weight dtype, residual rounding and output lifetime. The probe did not cover FP16, noncontiguous inputs, tail dimensions, mixed weight types, backward execution or models.

The [custom RMSNorm](mlx-custom-rms.md) page examines a separate two-output contract. Its timings cannot be substituted for this baseline. Raw local records remain outside the public repository.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) (official-doc)
- [mlx-rms-fast-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/fast.cpp) (upstream-code)
- [mlx-rms-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/rms_norm.metal) (upstream-code)
- [mlx-compile-fusion-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp) (upstream-code)
- local-mlx-residual-rms-20261007: `2026-10-07-mlx-residual-rms/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-custom-rms-20261007: `2026-10-07-mlx-custom-rms/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
