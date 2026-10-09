# Weights, KV and low precision: choose compression boundaries

[中文](../quantization.md) · English companion edition

Weight quantization reduces resident storage and traffic but adds format/scaling/decode and quality requirements. KV quantization changes context-dependent storage and per-step conversion/attention behavior. Lower-precision arithmetic may select another device path. These are independent decisions.

[MLX quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) and [quantized_matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) require compatible packed weights, scales/biases, mode, group size, bit width and transpose convention. GGUF Q4 formats are not interchangeable with MLX affine4 because their labels contain “4”.

[MSL quantized GEMV](msl-quantized-gemv.md) shows bit unpacking, group indexing and retained affine-cancellation failures. F32 activations/parameters/output do not remove evaluation-order error.

For standard equal-shape/equal-dtype K and V:

~~~text
KV bytes ≈ 2 × batch × layers × cached_tokens × KV_heads × head_dim × element_bytes
~~~

Sum heterogeneous layers separately; add scales, biases, padding and allocation overhead. Windows, shared prefixes, recurrent states and mixed architectures need their own accounting. Use KV heads, not query heads. Active MoE parameters do not describe all resident expert weights. Include scratch, logits, draft model, host cache and system headroom; RSS and Metal allocation can overlap.

A fixed [llama.cpp source example](llamacpp-fa-paths.md) derives 36MiB logical Q4_0 KV and a 128MiB F16 scratch term for one shape. It is not measured peak memory or a per-layer quantity to multiply blindly.

Fix model/tokenizer and a higher-precision reference. Evaluate weights, KV and their combination separately. Compare prefill/decode, peak memory and long-context/task quality. Cache/batching/FA compatibility is version-specific. Neither smaller files nor supported formats establish a speedup on the target Mac.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) (official-doc)
- [mlx-quantized-matmul-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) (official-doc)
- [mlx-cache-code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) (upstream-code)
- [llama-context](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp) (upstream-code)
- [apple-m5-ml](https://developer.apple.com/videos/play/tech-talks/111432/) (official-doc)
- [llama-metal-fa-ops-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- [llama-metal-fa-buffer-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
