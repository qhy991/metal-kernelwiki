# SwiGLU: existing compiled helpers and custom Metal arithmetic

[中文](../mlx-swiglu.md) · English companion edition

Check existing fusion before creating a custom shader. MLX-LM 0.31.3 already compiles SwiGLU with shapeless compilation and gate/x argument order. MLX 0.31.2 SiLU is also compiled; the ordinary baseline here used sigmoid directly.

M4/MLX 0.31.2 compared ordinary expression, compiled helper and custom Metal on F32/BF16, M=1/32/512, D=4096/4103, contiguous/step-two GPU views and three input distributions. All 216 prechecks passed.

Physical noncontiguity was established with evaluated GPU backing and stride metadata, rather than uploading a NumPy view and assuming its strides survived. Actual stored values defined the independent reference.

Custom arithmetic can change exponential selection, intermediate precision and rounding as well as dispatch structure. Preserve those differences when interpreting fusion.

After numerical checks, nine processes produced 1512 passing first/warmup/timed online checks on the restricted random/D4096 performance matrix. The interval was host completion, not a GPU kernel timestamp.

Independent array audit covered saved prechecks and first outputs; later samples retained online metrics. No one-kernel proof, resource counters, arbitrary-layout guarantee or model gain was established. Retain slow rounds and original tolerances. Raw local records are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-lm-swiglu-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/activations.py) (upstream-code)
- [mlx-silu-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/python/mlx/nn/layers/activations.py) (upstream-code)
- [mlx-compiled-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/compiled.cpp) (upstream-code)
- [mlx-compiled-layout-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/compiled.cpp) (upstream-code)
- [mlx-sigmoid-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/unary_ops.h) (upstream-code)
- [mlx-bf16-math-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/bf16_math.h) (upstream-code)
- [mlx-compile-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/usage/compile.rst) (official-doc)
- [mlx-compile-fusion-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp) (upstream-code)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- local-mlx-swiglu-20261007: `2026-10-07-mlx-swiglu/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
