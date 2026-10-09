# MSL half arithmetic: promotion, reciprocal and subnormal boundaries

[中文](../msl-half-arithmetic.md) · English companion edition

Half output storage does not establish half expression evaluation. Specify operand types, promotion, reciprocal spelling, intermediate narrowing and final bits separately.

MSL precision rules permit certain arithmetic rounding and subnormal flushing. Storage conversion is a different boundary. A CPU RNE mismatch alone is not a complete specification violation analysis.

The first M4/MLX 0.31.2 probe stopped after one metadata call because its checker rejected the generated `float16_t` alias. It produced zero arithmetic outputs. The failed record was retained.

An independent successor accepted the verified half/float16_t alias while preserving input domain, oracle and numerical gate. Twenty-eight arithmetic outputs passed; all generated wrappers and stored values were independently checked.

Half reciprocal routes preserved 2047 positive subnormal results for denominators above 16384, including 2^-16 at the largest tested denominators. This disproves universal flushing only for these observed programs, not every half kernel.

F32 reciprocal results differed from exact RNE at 1248 values per layout, while subsequent half narrowing hid all differences. Final half equality therefore cannot prove F32 equality.

Verbose MLX output records the generated wrapper, not the entire compilation unit or runtime compiler state. No timing or unique explanation of native softmax zeros was established. See [F32 division](msl-f32-divide.md). Raw records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-msl-half-arithmetic-20261008: `2026-10-08-msl-half-arithmetic/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [apple-half-promotion-wwdc16-606](https://developer.apple.com/videos/play/wwdc2016/606/) (official-doc)
- [mlx-metal-device-compile-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp) (upstream-code)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- [mlx-msl-utils-v0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/utils.h) (upstream-code)
- [mlx-compiled-layout-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/compiled.cpp) (upstream-code)
- local-msl-half-arithmetic-values-20261008: `2026-10-08-msl-half-arithmetic-values/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
