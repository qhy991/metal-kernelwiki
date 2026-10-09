# Quantized linear layers: packed execution, decode cost and resident dense memory

[中文](../mlx-qmm-path-comparison.md) · English companion edition

Compare three routes on identical packed parameters: A packed QMM; B dequantization plus dense matmul inside each measured call; C predecoded resident dense weights. C retains packed and dense storage, so it has a different memory budget.

The M4/MLX 0.31.2 experiment used BF16, W=[2048,1024], affine 4/8-bit group-64, transpose=True and M=1/8/32/128/512. Thirty numerical conditions passed before timing.

Eighteen separate timing processes rotated routes and M order, with three warmups and twelve samples per condition. Reports use median [min,max] across three process medians. Host completion includes graph creation and waiting, not pure GPU time.

For 4-bit M128, the tested dense routes had lower median completion time than packed in all three rounds. Other configurations showed large variation or ranking changes. This cannot establish a general token threshold.

Resident dense added 4 MiB of base allocation in this matrix. Active memory excludes idle allocator cache; resetting peak counters does not free arrays. Report baseline and operation-window peak separately rather than interpreting every peak as temporary memory.

This transpose=True domain does not repair the separate multirow transpose=False failure. Actual model shapes, prompt chunks, decode, quality, tails and memory pressure need evaluation before a dispatcher change. Raw inputs, timing samples and memory records are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-prefill-dequant-4621](https://github.com/ml-explore/mlx/issues/4621) (issue)
- [mlx-quantized-matmul-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) (official-doc)
- [mlx-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html) (official-doc)
- [mlx-active-memory-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html) (official-doc)
- [mlx-cache-memory-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html) (official-doc)
- [mlx-peak-memory-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html) (official-doc)
- [mlx-reset-peak-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reset_peak_memory.html) (official-doc)
- local-mlx-qmm-paths-20261007: `2026-10-07-mlx-qmm-paths/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
