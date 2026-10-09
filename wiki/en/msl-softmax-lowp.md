# Low-precision softmax: storage, accumulation and subnormal outputs

[中文](../msl-softmax-lowp.md) · English companion edition

Specify input storage, statistic dtype and final narrowing separately. Changing F16/BF16 input arguments does not change explicit float state in custom MSL. MLX v0.31.2 source selects float accumulation with precise=True and output-dtype accumulation otherwise.

The M4 probe ran on macOS 27.0.1 (26A434), MLX 0.31.2. It covered eleven widths from 31 to 65537, R8, F32/F16/BF16 storage and contiguous/step-two layouts.

Of 330 retained outputs, 304 passed and 26 native_false outputs failed. The run preserved exit 2, the original gate and every raw output bit. Other passing routes did not convert the overall result into success.

Separate input information loss, execution error against stored values and final output rounding. A large constant shift may cease to be a true shift after low-precision storage. Maximum subtraction protects exponentials but not every low-precision normalizer or reciprocal.

Half long-row zero outputs motivate diagnostics, but final zeros alone do not uniquely establish accumulator overflow or arithmetic subnormal flushing. A later standalone half reciprocal probe preserved subnormal outputs in its own context.

Do not apply F32 tolerances unchanged to low-precision storage or infer internal arithmetic from output dtype. No timing, native intermediate states, compile flags, profiler or model-quality conclusion was established. Raw records and failed outputs remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-msl-softmax-lowp-20261008: `2026-10-08-msl-softmax-lowp/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-metal-bf16-type-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16.h) (upstream-code)
- [mlx-bf16-math-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16_math.h) (upstream-code)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- [mlx-compiled-layout-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/compiled.cpp) (upstream-code)
- [mlx-msl-utils-v0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/utils.h) (upstream-code)
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-softmax-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.h) (upstream-code)
- [mlx-softmax-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/softmax.cpp) (upstream-code)
- [mlx-softmax-factory-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/jit_kernels.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
