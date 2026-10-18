# MLX execution: lazy graphs, compilation and custom Metal

[中文](../mlx-execution.md) · English companion edition

Separate graph construction, submission and completion. MLX uses lazy evaluation; printing, scalar extraction and NumPy conversion may trigger evaluation and interrupt batching. A default-stream synchronization is not proof that every stream has completed.

Keep compiled callables alive outside hot loops. Shapes, dtypes and input signatures affect compilation; `shapeless=True` does not make shape-dependent Python branches safe. Declare mutable state explicitly. Warm each tested signature and report first-call costs separately.

Inspect existing optimized helpers before introducing a custom kernel. MLX-LM 0.31.3 already compiles SwiGLU, and MLX 0.31.2 compiles SiLU. Source-level fusion and fewer Python calls do not establish one GPU dispatch.

For custom Metal, inspect generated argument types and actual GPU strides. `ensure_row_contiguous=True` may introduce copies. With it disabled, correctly address every input through its own strides. Small inputs may generate constant-address-space arguments.

Attention semantics include GQA broadcasting, float softmax calculations and lower-right causal alignment when query and key lengths differ. Float32 input/output alone does not define internal precision.

New documentation can describe options absent from installed 0.31.2, including `force_fused` or custom compile-option interfaces. Do not infer runtime flags from current documentation or change tolerances to accommodate a failed candidate.

2026-10-18 note (v0.32.3 source, installed binary not checked): `compile_options` exists in 0.32.3, but the Python binding accepts only the key `math_mode` (`safe`/`relaxed`/`fast`) and raises on any other key (`python/src/fast.cpp` L90-112). The MSL language version follows the OS (4.1 on macOS 27, 4.0 on 26, 3.2 on 15, else 3.1), and `build_library_` sets only math mode and language version (`device.cpp` L36-79, L650-662), so this interface cannot pin the language version or other `MTLCompileOptions` fields. The signature writer is now `mlx/backend/common/metal_kernel.cpp`: the body is appended verbatim (L174); `<name>_shape`, `<name>_strides` (`int64_t`) and `<name>_ndim` are added only when the source text contains them (L117-133, L238-240), and thread attributes likewise (L243-270). Libraries are cached by kernel name + `hash(source)` + options (`custom_kernel.cpp` L50-54; `device.cpp` L800-818). `mx.fast.metal_kernel` returns an `nb::cpp_function` (`fast.cpp` L434-435) with no accessor for compiled-pipeline properties.

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
- [mlx-metal-kernel-writer-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/common/metal_kernel.cpp) (upstream-code)
- [mlx-custom-metal-impl-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- [mlx-python-fast-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/fast.cpp) (upstream-code)
- [mlx-metal-device-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
