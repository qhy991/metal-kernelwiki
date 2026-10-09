# MLX float32 precision: singleton GQA as a diagnostic

[中文](../mlx-float32-precision.md) · English companion edition

Float32 input/output does not guarantee full float32 internal arithmetic. Current documentation and a historical M5 issue describe precision risks; they must be distinguished from installed M4/MLX 0.31.2 observations.

For exactly one visible key with no sinks, attention output should equal stored V. The local probe fixed zero Q/K, Hq=16, Hkv=1, D=512 and B=1/2. Four routes compared compact/repeated-KV SDPA and broadcast/repeated-V matmul.

Three V distributions, two storage dtypes and both batches formed 48 configurations, repeated in three same-input processes. Every output was finite and exactly equal to actual stored V. `MLX_ENABLE_TF32` was unset.

This is a narrow identity test. It does not validate multiple keys, nonzero scores, arbitrary masking, model quality or reduced-precision behavior elsewhere. A passing M4 result cannot establish that a closed M5 issue was fixed in a particular release.

Explicitly repeating K/V may add allocation and work; it is a diagnostic control, not an accepted deployment repair. Current MLX 0.32.3 documentation cannot retroactively define 0.31.2 internal behavior. Raw local records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-numerical-precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html) (official-doc)
- [mlx-singleton-gqa-issue](https://github.com/ml-explore/mlx/issues/3953) (issue)
- [mlx-sdpa-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (official-doc)
- local-mlx-boundaries-20261007: `2026-10-07-mlx-boundaries/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
