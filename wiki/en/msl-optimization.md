# MSL optimization: candidate mechanisms and their costs

[中文](../msl-optimization.md) · English companion edition

After locating a kernel hotspot, change a specific mechanism and measure its full cost. These source-derived patterns use MLX v0.31.2 and Apple documentation; they do not establish universal parameters.

Increase work per thread only with a matching access and tail contract. Fixed small loops can reduce thread overhead but extend live values and increase register pressure. A scalar array or vector type does not prove physical register residence or vectorized instructions.

For reductions, compare rereading input against retaining thread-local values. Retention can save source-level loads while causing longer lifetimes or spills. SIMD partials followed by threadgroup reduction also add synchronization and scratch costs.

Keep template constants, Metal function constants, uniform runtime values and constant-address-space buffers distinct. Specialization may remove branches, but adds compilation variants and cache costs. Warm the actual signatures used by a workload.

For fusion, preserve intermediate rounding, output ownership and layouts. Removing a source-level operation is not proof that the compiler removed its work. Compare native primitives and existing compiled helpers before hand-written replacements.

Generate structurally different candidates: one SIMD per row, multiple SIMD groups with shared partials, or split work across kernels. Validate external numerical gates first, then compare repeated completion timing and profiler evidence on the target.

See [programming](msl-programming.md), [GEMV](msl-gemv.md), [tiles](msl-tiles.md) and [softmax timing](msl-softmax-timing.md). No speed, occupancy, bandwidth or register claim follows solely from the source patterns.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-msl-unary-v0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/unary.h) (upstream-code)
- [mlx-msl-utils-v0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/utils.h) (upstream-code)
- [mlx-rms-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/rms_norm.metal) (upstream-code)
- [mlx-normalization-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/normalization.cpp) (upstream-code)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- [apple-msl-compute-tech10580](https://developer.apple.com/videos/play/tech-talks/10580/) (official-doc)
- [apple-msl-renderers-wwdc23-10127](https://developer.apple.com/videos/play/wwdc2023/10127/) (official-doc)
- [apple-msl-function-constant-values](https://developer.apple.com/documentation/metal/mtlfunctionconstantvalues) (official-doc)
- [apple-msl-function-specialization](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants) (official-doc)
- [apple-threadgroup-sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes) (official-doc)
- [apple-m3-profiling](https://developer.apple.com/videos/play/tech-talks/111374/) (official-doc)
- [apple-m5-profiling](https://developer.apple.com/videos/play/tech-talks/111431/) (official-doc)

[Shared source catalog](../../data/catalog.json)
