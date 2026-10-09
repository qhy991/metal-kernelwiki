# GEMM, GEMV and MoE: choose algorithms for actual shapes

[中文](../gemm-moe.md) · English companion edition

Upstream changes are mechanism references, not reproduced local speedups. [Matrix](msl-matrix.md), [tile](msl-tiles.md) and [GEMV](msl-gemv.md) studies retain accumulation, layout and cancellation limits; none establishes model or expert-routing throughput.

| Workload | Candidate | Include these costs |
|---|---|---|
| Dense single-token decode | GEMV, fused quantized decoding | Weight bytes, registers, reduction |
| Prefill or continuous batch | Tiled GEMM, supported TensorOps | Padding, conversion, temporary storage |
| Large K with insufficient output tiles | Split-K | Partial buffers, combine, changed summation |
| Small expert groups | Gather/group/sort for reuse | Sort, reorder, scatter, empty/hot experts |

[MLX PR #3018](https://github.com/ml-explore/mlx/pull/3018), merged 2026-01-26, illustrates NAX Split-K on particular M5/large-K workloads. Its test variable `MLX_DISABLE_SPLITK_NAX` was removed; do not copy it as a current API. [PR #4572](https://github.com/ml-explore/mlx/pull/4572), merged 2026-09-28, supplies gather_qmm/MoE optimization leads, not local whole-model claims.

`gather_qmm` indices select flattened batch matrices; scales/bias batches must correspond to weights. Set `sorted_indices=True` only for actually sorted indices; it does not sort them. [API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)

[Issue #4632](https://github.com/ml-explore/mlx/issues/4632) reports a specific M5 Max/macOS27 sorted-gather tail problem. It is an unreplicated author report, not a defect established across all versions.

Validate small M, K variants, non-tile tails, duplicate/empty experts, strides, quantization and quality. Measure gather→compute→scatter. Test fallback guards and missing/overlapping regions. [SwiGLU](mlx-swiglu.md) covers resident activation inputs only, excluding gate/up/down GEMM and routing.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-splitk-pr](https://github.com/ml-explore/mlx/pull/3018) (merged-pr)
- [mlx-gather-tiling-pr](https://github.com/ml-explore/mlx/pull/4572) (merged-pr)
- [mlx-gather-qmm-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html) (official-doc)
- [mlx-gather-tail-issue](https://github.com/ml-explore/mlx/issues/4632) (issue)
- local-mlx-swiglu-20261007: `2026-10-07-mlx-swiglu/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-lm-swiglu-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/activations.py) (upstream-code)

[Shared source catalog](../../data/catalog.json)
