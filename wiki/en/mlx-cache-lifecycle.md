# Rotating KV cache: preserve request policy through batching

[中文](../mlx-cache-lifecycle.md) · English companion edition

Rotation retains a prefix of `keep` tokens and a recent-history suffix. Merge compatibility must account for policy, offsets, valid history and padding, not only capacity.

In MLX-LM 0.31.3, examined direct merge/extract code did not preserve `keep=4`. A separate model-generation entry point rejected positive-keep rotating caches. That guard does not protect every direct API call.

The M4 probe used two float32 requests, H=1, D=64 and capacity eight. After six initial tokens, merging and ten single-token appends, exact encoded K/V checks found prefix-policy loss for equal positive-keep and mixed policies. `keep=0` controls passed. Shapes, sizes and offsets passed even when retained content was wrong.

Grouping requests by keep alone would not fix the equal-policy failure. Replacing keep with zero changes semantics and is not an equivalent repair.

Verify policy, logical history and continuation across capacity transitions before accepting a batching candidate. This run did not execute service, attention, mask, persistence, quantized caches or model-quality tests, and did not measure performance. Its exact-array comparison applies to the observed completed ring cycle, not arbitrary physical ring positions.

See [mask alignment](mlx-cache-mask-alignment.md) and [KV cache](mlx-kv-cache.md). The original failed records remain external and unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-rotating-keep-lifecycle-issue](https://github.com/ml-explore/mlx-lm/issues/1631) (issue)
- [mlx-lm-cache-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) (upstream-code)
- [mlx-lm-generate-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py) (upstream-code)
- local-mlx-qvm-cache-20261007: `2026-10-07-mlx-qvm-cache/results-summary.json` (local experiment; raw record unpublished).
- local-mlx-cache-mask-20261007: `2026-10-07-mlx-cache-mask/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
