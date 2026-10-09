# MSL F32 division: function choice, rounding and input provenance

[中文](../msl-f32-divide.md) · English companion edition

Compare ordinary `a/b`, `metal::fast::divide(a,b)` and `metal::precise::divide(a,b)` while retaining float outputs. Quality tolerance, correct rounding, specification bounds and deployment speed are separate judgments.

The M4/MLX 0.31.2 probe on macOS 27.0.1 compared mathematically equal half-origin and float-buffer inputs. All eighteen outputs, totaling 221190 values, passed their predefined quality gates.

For explicit precise division with half-origin inputs, 1248 values per layout differed from exact RNE; 1232 matched neither exact RNE nor exact RTZ. Ordinary/precise division with equal values read from float buffers matched RNE in the examined domain.

Do not erase this unexplained rounding difference because quality passed. Normal finite F32 inputs/results also prevent a simple F32-subnormal explanation for those cases. No compiler phase or instruction cause was established.

An independent successor compared direct half promotion, CPU-produced F32 and separately evaluated GPU half→F32 promotion over 16384 denominators. It retained 49152 division outputs and 16384 conversion values.

Conversion was exact; all quality gates passed. CPU-F32 and completed GPU-F32 inputs passed through a shared consumer and matched RNE, while inline half promotion retained the 1248 differences.

This establishes an application-visible materialization effect only in the tested programs. Wrapper/host binding checks do not prove hardware resource binding. No timing, compiler-cause, global rounding or model-quality conclusion was made. Raw records are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-msl-f32-divide-20261008: `2026-10-08-msl-f32-divide/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [apple-metal-compile-mathmode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode) (official-doc)
- [mlx-metal-device-compile-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp) (upstream-code)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- local-msl-divide-provenance-20261008: `2026-10-08-msl-divide-provenance/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
