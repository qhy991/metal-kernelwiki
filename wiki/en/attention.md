# Attention: separate prefill, decode and GQA paths

[中文](../attention.md) · English companion edition

Start from framework primitives when attention is a measured hotspot. [Local M4 checks](local-mlx-m4.md) cover bounded SDPA outputs and host-call timing, not model acceleration.

MLX fast SDPA uses [B,H,T,D], supports MHA/GQA/MQA and computes softmax in float32. Preserve grouped K/V rather than expanding them to query heads. Causal masks align at the lower right, especially when Tq differs from Tkv. [API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)

The newer documentation's `force_fused` parameter was absent from installed MLX0.31.2. Check the real signature; ordinary SDPA correctness proves neither forced fusion nor a particular dispatch.

~~~python
mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask="causal")
~~~

This is an interface sketch. Validate shapes, sinks, windows, padding and offsets against the model contract. A single key without sinks should return V; the [48 singleton controls](mlx-float32-precision.md) do not cover multi-key reductions.

Compare structurally different candidates: prefill tiled score/softmax/value fusion; decode KV partitions plus complete combine costs; GQA reuse without costly replication. [llama.cpp paths](llamacpp-fa-paths.md) have distinct conversion/vector/tensor/sparse gates. `n_kv_max` bounds finite-mask entries per row; under-reporting truncates indices rather than merely changing performance.

[RoPE validation](mlx-rope-qk.md) retains batch/scalar-offset and long-position failures. [MSL softmax](msl-softmax.md) validates logits-to-probabilities and intermediate states, not QK/PV or complete FlashAttention.

Use a high-precision independent oracle across query/key lengths, tails, GQA ratios, padding/causal/windows/sinks, cache offsets and dtypes. State fully-masked-row behavior. Distinguish attention execution, KV copies and submission gaps; measure temporary memory and request latency.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-sdpa-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (official-doc)
- [llama-tuning](https://github.com/ggml-org/llama.cpp/blob/master/tools/tuning/README.md) (official-doc)
- [mlx-custom-metal](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) (official-doc)
- local-mlx-m4-r1: `2026-10-07-mlx-m4-r1/validation/results.json` (local experiment; raw record unpublished).
- local-mlx-boundaries-20261007: `2026-10-07-mlx-boundaries/summary.json` (local experiment; raw record unpublished).
- local-mlx-rope-qk-20261007: `2026-10-07-mlx-rope-qk/derived/summary.json` (local experiment; raw record unpublished).
- [llama-metal-fa-ops-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- [llama-metal-fa-buffer-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
