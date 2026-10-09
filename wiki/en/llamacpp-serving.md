# llama.cpp serving: slots, prefix reuse and speculative decoding

[中文](../llamacpp-serving.md) · English companion edition

Keep request lengths, templates, sampling, cache state and quality targets fixed. Slots (`-np`), logical batch (`-b`), physical microbatch (`-ub`) and continuous batching control different resources. Verify installed help and startup logs before interpreting capacity.

Compare one, two and four slots under the same workload. Unified-KV options can change the relationship between total context and per-request capacity; do not assume a universal `context/slots` formula.

Shared-prompt and independent-prompt batched benchmarks have different storage requirements. Rough token counts are `PP+B*TG` and `B*(PP+TG)` respectively. Neither benchmark alone represents production queueing.

Prefix caching requires exact reusable tokens. Bound host prompt-cache RAM and measure cold, warm, unrelated-prefix and eviction cases. Warm-cache speed is not first-request speed. The host cache competes with model and GPU demand in unified memory.

For speculation, compare the same baseline with one candidate at a time. N-gram modes can avoid a second model; draft-model modes add weights, KV and verification. MTP/EAGLE/DFlash have model-specific requirements and are not interchangeable.

Record useful-output latency, acceptance, quality and memory by request category. Coding gains cannot be generalized to high-entropy text. Synthetic acceptance can admit tokens that differ from the target and is not deployment evidence.

The detailed edition provides version-dependent recipes and upstream sources. No local service or speculative-decoding gain has been established. See [deployment](llamacpp-deployment.md) and [measurement](measurement.md).

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (official-doc)
- [llama-batched-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md) (official-doc)
- [llama-prompt-cache-pr](https://github.com/ggml-org/llama.cpp/pull/16391) (merged-pr)
- [llama-speculative](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) (official-doc)
- [llama-spec-code](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (upstream-code)
- [llama-speed-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md) (official-doc)

[Shared source catalog](../../data/catalog.json)
