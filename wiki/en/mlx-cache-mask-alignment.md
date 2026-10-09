# Rotating-cache masks: align visibility with returned slots

[中文](../mlx-cache-mask-alignment.md) · English companion edition

Correct K/V content and offsets do not establish correct attention visibility. A mask created before a multi-token update must match the physical columns returned after that update.

The M4/MLX 0.31.2/MLX-LM 0.31.3 probe used B=2, H=1, D=64, capacity eight, `keep=0` and unequal left-padded request lengths. Direct construction and merge entry points appended mixed one-/multi-token blocks.

An independent logical-token reference required each visible token exactly once and allowed keys satisfying `0<=k<=q` and `q-k<window`. Zero Q made attention equal the mean of visible V, separating mask failures from numerical dot-product behavior.

Four of 90 window checks failed after rotation and a three-token append with window eight. The short request wrongly hid token zero: eight mask bits and eight query outputs were wrong, with maximum absolute error 0.03125. Cache values, visibility-set completeness and offsets passed.

All reference-mask controls passed. Window-three and equal-length controls also passed within this matrix; overlapping controls are not extra independent samples.

Retain exit 2 and the original gate. A reference mask is a diagnostic, not a measured general repair. Nonzero Q, RoPE, positive keep, services, model quality and timing were not tested. Raw arrays and logs are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-lm-cache-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) (upstream-code)
- [mlx-lm-base-mask-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py) (upstream-code)
- [mlx-lm-cache-tests-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/tests/test_prompt_cache.py) (upstream-code)
- local-mlx-cache-mask-20261007: `2026-10-07-mlx-cache-mask/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
