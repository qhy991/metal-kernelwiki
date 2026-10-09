# MLX serving: batching eligibility, cache budgets and speculation

[中文](../mlx-serving.md) · English companion edition

Treat service throughput as a request-distribution measurement. Retain prompt/output lengths, arrival patterns, concurrency, sampling, cache state and latency targets. Local primitive results in this repository do not establish service gains.

In the examined mutable server source, decode concurrency maps to completion batch size and prompt concurrency to prefill batching. Batch eligibility also depends on model support, request seed and CLI KV quantization settings. Incompatible requests can drain an existing batch before sequential execution.

Do not change sampling or cache semantics to make a request eligible. Test mixed lengths, cancellation, requests joining an active batch, oversized prompts and policy changes. Measure queueing, TTFT, per-token latency, total throughput and memory together.

Active cache bytes reduce available prompt-cache budget. Prefix reuse must preserve exact token history and request ownership. A warm repeated request is a separate workload from a cold or unrelated request.

Speculative decoding adds draft computation, verification, weights and cache. Verify target/draft tokenizer and vocabulary compatibility, sampling behavior, cache trimming and output quality. Acceptance rate is a diagnostic rather than proof of speed.

Use [measurement](measurement.md), [KV cache](mlx-kv-cache.md) and [cache lifecycle](mlx-cache-lifecycle.md). Inspect the installed server implementation before applying current-main defaults. No local service benchmark or end-to-end speculative-decoding validation is claimed here.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-server-code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py) (upstream-code)
- [mlx-generate-code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) (upstream-code)

[Shared source catalog](../../data/catalog.json)
