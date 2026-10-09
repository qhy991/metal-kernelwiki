# MLX execution: lazy graphs, compilation and custom Metal

[中文](../mlx-execution.md) · English companion edition

Separate graph construction, submission and completion. MLX uses lazy evaluation; printing, scalar extraction and NumPy conversion may trigger evaluation and interrupt batching. A default-stream synchronization is not proof that every stream has completed.

Keep compiled callables alive outside hot loops. Shapes, dtypes and input signatures affect compilation; `shapeless=True` does not make shape-dependent Python branches safe. Declare mutable state explicitly. Warm each tested signature and report first-call costs separately.

Inspect existing optimized helpers before introducing a custom kernel. MLX-LM 0.31.3 already compiles SwiGLU, and MLX 0.31.2 compiles SiLU. Source-level fusion and fewer Python calls do not establish one GPU dispatch.

For custom Metal, inspect generated argument types and actual GPU strides. `ensure_row_contiguous=True` may introduce copies. With it disabled, correctly address every input through its own strides. Small inputs may generate constant-address-space arguments.

Attention semantics include GQA broadcasting, float softmax calculations and lower-right causal alignment when query and key lengths differ. Float32 input/output alone does not define internal precision.

New documentation can describe options absent from installed 0.31.2, including `force_fused` or custom compile-option interfaces. Do not infer runtime flags from current documentation or change tolerances to accommodate a failed candidate.

See [async evaluation](mlx-async-evaluation.md), [fusion](fusion.md), [MSL programming](msl-programming.md) and [precision diagnostics](mlx-float32-precision.md).

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-lazy-evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) (official-doc)
- [mlx-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html) (official-doc)
- [mlx-synchronize-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html) (official-doc)
- [mlx-sdpa-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (official-doc)
- [mlx-compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) (official-doc)
- [mlx-custom-metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) (official-doc)
- local-mlx-m4-r1: `2026-10-07-mlx-m4-r1/validation/results.json` (local experiment; raw record unpublished).
- local-mlx-boundaries-20261007: `2026-10-07-mlx-boundaries/summary.json` (local experiment; raw record unpublished).
- [mlx-compile-fusion-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp) (upstream-code)
- local-mlx-residual-rms-20261007: `2026-10-07-mlx-residual-rms/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-custom-rms-20261007: `2026-10-07-mlx-custom-rms/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- local-mlx-rope-qk-20261007: `2026-10-07-mlx-rope-qk/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-swiglu-20261007: `2026-10-07-mlx-swiglu/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-lm-swiglu-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/activations.py) (upstream-code)

[Shared source catalog](../../data/catalog.json)
