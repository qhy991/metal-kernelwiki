# MSL matrix multiplication: fragments, tails and accumulation precision

[中文](../msl-matrix.md) · English companion edition

Public `simdgroup_matrix` operations cooperate across a SIMD group. The specification defines supported matrix types and uniform participation, but does not define the element-to-lane map. Use public load/store operations instead of assuming fragment ownership.

The teaching contract uses F16 stored A[M,K]/B[K,N] with F32 output. One complete 32-thread group handles an 8×8 tile. Public load/store lacks a mask, so tails require bounded scalar staging, explicit zero fill and guarded output stores.

All participating lanes execute cooperative matrix operations and barriers uniformly, including lanes without a valid output cell. Match pointer address spaces; small MLX arguments may be constant pointers.

F16 storage, F16 matrix arithmetic and F32 output are different contracts. Promoting input values to float fragments does not have the same rounding behavior as accumulating into half fragments and widening afterward.

On M4/MLX 0.31.2, four routes produced 324 checks from 81 input conditions. Three F32 computation routes passed all 81 each. The half-matrix route failed all 81 under the shared F32-output gate; 27 included nonfinite outputs. Original failures were retained.

Steel source provides candidate reuse and scheduling patterns, not evidence that these examples reached its performance or dedicated matrix hardware. The run did not time candidates or inspect instructions/counters. Raw inputs, outputs and independent CPU audit are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-steel-mma-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/mma.h) (upstream-code)
- [mlx-steel-gemm-fused-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h) (upstream-code)
- [mlx-metal-matmul-host-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp) (upstream-code)
- [mlx-steel-nax-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/nax.h) (upstream-code)
- [apple-feature-tables-20260521](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) (official-doc)
- local-msl-matrix-20261007: `2026-10-07-msl-matrix/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
