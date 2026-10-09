# Cache persistence: tensor round-trip and continuation are separate gates

[中文](../mlx-cache-persistence.md) · English companion edition

Saving a prompt cache can avoid repeated prefill only when model, tokenizer, template, tokens and positions match. Include file I/O in user-facing TTFT. Loading equal tensors is insufficient: control state and subsequent operations must also work.

The M4/MLX 0.31.2/MLX-LM 0.31.3 probe saved twelve synthetic snapshots. Tensor data and user metadata round-tripped, but two rotating flags restored incorrectly. All six tested continuations of restored batched rotating caches raised missing-field exceptions.

Reload the original snapshot independently for single-token and multi-token continuation. Compare with an unsaved same-history control and logical-token oracle. Do not repair fields in a failed object and relabel the original run.

Upstream PR #1778, pinned at `ee19be43625b9385f979de6133938f683aba6e8e`, changed scalar serialization and metadata layout. The inspected loader lacked a legacy-format conversion branch; neither backward compatibility nor explicit rejection was established.

The pinned setter still omitted `_lengths` initialization in an examined path. That is a source observation, not execution of the newer version.

The local probe did not test cross-version restoration, nonzero padding, positive keep, quantized cache, restored attention, model quality, file-size savings or latency. Closed issue status does not establish full continuation acceptance. Raw snapshots, exceptions and arrays remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-lm-cache-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) (upstream-code)
- [mlx-cache-rotated-1250](https://github.com/ml-explore/mlx-lm/issues/1250) (issue)
- [mlx-cache-full-state-1778](https://github.com/ml-explore/mlx-lm/pull/1778) (merged-pr)
- [mlx-cache-full-state-ee19be4](https://github.com/ml-explore/mlx-lm/blob/ee19be43625b9385f979de6133938f683aba6e8e/mlx_lm/models/cache.py) (upstream-code)
- local-mlx-cache-roundtrip-20261007: `2026-10-07-mlx-cache-roundtrip/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
