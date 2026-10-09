# MSL tiles: direct loading, shared operands and BK tradeoffs

[中文](../msl-tiles.md) · English companion edition

Compare direct 8×8 loading, staged 8×8 tiles, four SIMD groups sharing a 16×16 output tile, and larger K staging blocks. These examples use F32 buffers; they are not F16-storage or model benchmarks.

For a full 16×16×8 block, sharing operands changes explicit source loads from 512 to 256 elements. This is reuse in source, not proof that DRAM traffic halves.

Increasing BK8 to BK32 raises A/B scratch from 1 KiB to 4 KiB, plus 1 KiB output scratch. At K256, staging rounds and associated barriers decrease while MMA fragment count stays the same. K tails can add zero-filled fragment work.

Direct loading requires the example's full-block and stride-one predicate. Otherwise it uses bounded staging. Input and output direct eligibility are independent. A K tail does not disable direct loads for earlier full K blocks.

All four SIMD groups help fill shared A/B tiles and reach threadgroup barriers, even if their output subtile lies outside M/N. Never skip a cooperative barrier based on only one group's output validity.

On M4/MLX 0.31.2, 270 prechecks and 750 retained first/warmup/timed outputs passed independent numerical checks. Process timing rankings varied, including BK reversals; M1-row cases never entered this direct fast path.

Report host completion rather than GPU time. No resource-allocation, occupancy, instruction, DRAM or model claim was established. Actual raw records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-msl-tiles-20261007: `2026-10-07-msl-tiles/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-steel-mma-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/mma.h) (upstream-code)
- [mlx-steel-gemm-fused-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h) (upstream-code)
- [mlx-custom-metal-guide-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/docs/src/dev/custom_metal_kernels.rst) (official-doc)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
