# MSL programming: address spaces, vector layout and synchronization

[中文](../msl-programming.md) · English companion edition

Write access and participation contracts before tuning. The guide reads Apple's MSL 4.1 specification dated 2026-06-04; that does not establish the installed compiler's selected language version.

Match `device`, `constant`, `threadgroup` and `thread` pointer address spaces to generated signatures. A cast cannot legalize a constant-to-device pointer conversion. MLX custom kernels can generate constant arguments for small arrays.

Vector storage and scalar work assignment differ. `float3` occupies/aligned to 16 bytes, while `packed_float3` occupies 12 bytes with four-byte alignment. Four scalar elements per thread do not prove a vector load instruction.

Within a SIMD group, use reductions with valid active lanes. To share partials across SIMD groups, publish threadgroup memory and use a threadgroup barrier before consumers read it. Every participating thread must reach required barriers uniformly; masking arithmetic is safer than returning before a collective.

Math namespace, math mode, FP32 function set, contraction and storage narrowing are separate controls. Explicit `precise::exp` does not make all surrounding arithmetic CPU-equivalent. Atomics require their own supported-type and ordering contract.

The local M4/MLX 0.31.2 teaching run preserved an initial address-space compilation failure. Its independent corrected candidate passed 170 numerical outputs but failed singleton-stride metadata expectations; the overall record remained failed.

Compilation, numerical correctness, generated instruction behavior and speed are separate evidence. See [optimization](msl-optimization.md), [matrix](msl-matrix.md) and [softmax](msl-softmax.md). Raw local examples and records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [apple-msl-generic-pointers](https://developer.apple.com/documentation/metal/writing-reusable-gpu-functions-with-generic-pointers) (official-doc)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- local-msl-usage-20261007: `2026-10-07-msl-usage/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
