# MLX boundary probes: observed results and remaining scope

[中文](../mlx-regression-probes.md) · English companion edition

This page tracks bounded M4/MLX 0.31.2 observations rather than a general backend qualification. Fix versions, input seeds, actual storage, shapes, oracle, tolerance and evaluation boundary before extending a probe.

| Boundary | Observed result | Remaining scope |
|---|---|---|
| Async submission and dependent tail | Four modes passed in three processes; completion times varied | Real generation, streams and GPU timing |
| Singleton GQA | 48 configurations exactly equaled stored V | Multiple keys, M5 reports and models |
| Multirow nontransposed QVM | All 32 cases failed external numerical gates | Verified repair and performance |
| Rotating keep policy | Direct merge/extract lost keep=4 | Service and model paths |
| Rotating masks | Four of 90 checks failed despite correct cache content | Broader positions, masks and attention |
| Persistence | Tensor round-trip passed; selected restored continuation failed | Cross-version and model acceptance |

Keep original design proposals separate from executed observations. Same-input process repeats are not new random configurations. A passing diagnostic control does not convert a failed original path into acceptance.

Route failures to cache policy, mask alignment, arithmetic or execution before benchmarking. The linked topic pages carry detailed configurations and unpublished artifact references. This tracker does not authorize model downloads, formal campaigns or changes to frozen project revisions.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-async-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html) (official-doc)
- [mlx-queue-batched-timing-issue](https://github.com/ml-explore/mlx/issues/4265) (issue)
- [mlx-singleton-gqa-issue](https://github.com/ml-explore/mlx/issues/3953) (issue)
- [mlx-qvm-multirow-fix](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68) (merged-pr)
- [mlx-qvm-multirow-test](https://github.com/ml-explore/mlx/pull/3497/commits/2ecf184f9150b85b0aa139f7d827751011874d35) (upstream-code)
- [mlx-rotating-keep-lifecycle-issue](https://github.com/ml-explore/mlx-lm/issues/1631) (issue)
- local-mlx-boundaries-20261007: `2026-10-07-mlx-boundaries/summary.json` (local experiment; raw record unpublished).
- local-mlx-qvm-cache-20261007: `2026-10-07-mlx-qvm-cache/results-summary.json` (local experiment; raw record unpublished).
- local-mlx-cache-mask-20261007: `2026-10-07-mlx-cache-mask/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-quantized-attention-20261007: `2026-10-07-mlx-quantized-attention/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
