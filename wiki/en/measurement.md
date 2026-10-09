# Measurement: turn slow inference into testable bottlenecks

[中文](../measurement.md) · English companion edition

These are measurement methods and candidate designs, not newly measured GPU results. Fix the workload before comparing changes.

Record the exact chip/SKU, GPU cores, memory, OS/SDK, framework revision, model, tokenizer/template, weight/KV format, prompt/output lengths, concurrency, sampling, cache state, power and thermal conditions. Different quantized models represent deployment tradeoffs; they are not identical arithmetic benchmarks.

| Symptom | Separate these costs | Candidate directions |
|---|---|---|
| Slow cold start | Download, loading/page-in, JIT, allocation | Resident models, reusable pipelines, metallib |
| High TTFT | Queueing, tokenization, prefill, sampling/transport | Chunked prefill, prefix reuse, GEMM |
| Slow tokens | Submission, weights, KV attention, readback | Fusion, quantization, KV paths |
| Low concurrent throughput | Ineffective batch, slots, CPU serialization | Scheduling; include request tail latency |
| Context degradation | KV bytes, attention, copies, swap | Budgeting, FA, quantization, cache policy |

[Apple's M5 research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) is a workload-specific starting point, not a prediction for every batch or MoE model.

Separate three timing boundaries. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) excludes tokenization and sampling. Service TTFT includes request processing and queueing. MLX graph construction ends before lazy computation finishes: [evaluate needed outputs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html) and wait for the producing stream when necessary.

Retain prefill/decode throughput, TTFT, token intervals, end-to-end latency, aggregate throughput and peak memory. State first-token/EOS/sampling inclusion. Warm up, interleave baseline/candidate runs, preserve samples, and report median and [min,max]. Five kernel repetitions do not establish service p95. Allocator-cache clearing is not hardware-cache flushing. Profile separately from ordinary timing.

Check tails, strides, extremes, masks, shapes and dtypes against an external oracle. Quantization also needs task quality. Exercise cache reuse/eviction/offsets and speculative acceptance/rejection paths. Failures remain visible.

A bandwidth estimate is a hypothesis until measured traffic supports it. Adoption requires both agreed quality and end-to-end goals. Every result names its configuration, oracle, timing boundary, samples, profiler coverage, artifact and unexamined domain.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-mlx-m5-research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) (official-doc)
- [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) (official-doc)
- [mlx-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html) (official-doc)
- [mlx-synchronize-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html) (official-doc)

[Shared source catalog](../../data/catalog.json)
