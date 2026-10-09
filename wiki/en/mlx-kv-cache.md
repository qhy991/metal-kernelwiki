# MLX KV cache: reuse, rotation, quantization and correctness

[中文](../mlx-kv-cache.md) · English companion edition

Prefer the model's `make_cache()` contract over a generic cache assumption. Cache type, layer structure and supported capacity controls vary by model. Prefix reuse requires identical model, tokenizer, template, token prefix and positions; mutable state needs clear request ownership.

Rotation changes retained history. Preserve prefix retention, logical offsets, padding and mask alignment together. Equal capacity does not imply equal policy. In MLX-LM 0.31.3, direct merge/extract lost `keep=4` while a separate model entry point guarded against that policy. Rotating-cache quantization was not implemented in the examined path.

Quantization trigger defaults are version-specific: installed 0.31.3 used `quantized_kv_start=0` while examined mutable main used 5000. Quantized attention calls QK, softmax and PV through a helper rather than passing packed cache directly to ordinary SDPA.

Local M4 probes found 48 failed PV cases among 128 quantized-attention configurations. Independent dense controls on the same decoded data passed. Storage compression therefore cannot authorize numerical acceptance.

Validate merge, append, filter, extract, persistence and continuation at capacity transitions. An unequal-length, left-padded `keep=0` probe found four mask failures among 90 checks even though cache content and offsets passed. Saved tensors can also round-trip while continuation fails.

Track actual cache payload separately from allocator peak, resident weights and temporary buffers. CUDA-specific alignment advice is not a Metal requirement. See [lifecycle](mlx-cache-lifecycle.md), [mask alignment](mlx-cache-mask-alignment.md), [persistence](mlx-cache-persistence.md) and [quantized attention](mlx-quantized-attention.md).

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-cache-code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) (upstream-code)
- [mlx-lm-readme](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations) (upstream-code)
- [mlx-generate-code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) (upstream-code)
- [mlx-fast-kv-cache](https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html) (official-doc)
- local-mlx-m4-r1: `2026-10-07-mlx-m4-r1/validation/results.json` (local experiment; raw record unpublished).
- local-mlx-qvm-cache-20261007: `2026-10-07-mlx-qvm-cache/results-summary.json` (local experiment; raw record unpublished).
- local-mlx-quantized-attention-20261007: `2026-10-07-mlx-quantized-attention/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
