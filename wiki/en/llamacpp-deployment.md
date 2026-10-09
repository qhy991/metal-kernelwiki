# llama.cpp Metal deployment: offload, capacity and PP/TG

[中文](../llamacpp-deployment.md) · English companion edition

Pin the llama.cpp commit, GGUF model/quantization, chip, memory and macOS. Check each binary's own help: server and benchmark flags need not match. Build with Metal enabled and inspect actual device, offload, KV and compute-buffer logs.

`-ngl 0` provides a CPU candidate. A large numeric offload request does not guarantee every layer uses Metal. Current server accepts `all` and `auto` in some interfaces; older versions or bench may require numbers.

Measure prompt processing (`pp`), generation (`tg`) and combined execution (`pg`) separately. Vary prefilled KV depth as well as prompt/output length. `llama-bench` excludes tokenizer and sampler work, so its timings are not client TTFT.

After checking installed flags, compare a small set of FA and microbatch settings with logical batch at least as large as physical microbatch. Retain every repetition and alternate candidate/baseline order. Overlapping distributions do not establish a winner.

Start KV comparisons with F16. Test Q8 candidates only after confirming FA, operator support and model quality. Quantized V requires FA in the examined initialization source. K/V storage savings do not imply equal compute-buffer savings: FA may reserve F16 scratch even without executing conversion.

Budget weights, KV, temporary computation, host prompt cache, draft model and system demand. Recommended working-set size is not a hard allocation limit. Record memory pressure and swap; do not sum RSS and Metal allocation as independent physical memory.

See [FA paths](llamacpp-fa-paths.md), [serving](llamacpp-serving.md), [validation](llamacpp-validation.md) and [measurement](measurement.md). Commands in the detailed edition are recipes, not locally measured best settings.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [llama-build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) (official-doc)
- [llama-args](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) (upstream-code)
- [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) (official-doc)
- [llama-context](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp) (upstream-code)
- [llama-metal-device](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (upstream-code)
- [llama-metal-fa-ops-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- [llama-metal-fa-buffer-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
