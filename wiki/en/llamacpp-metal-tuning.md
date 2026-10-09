# llama.cpp Metal tuning: GPU families, numerical gates and Tensor paths

[中文](../llamacpp-metal-tuning.md) · English companion edition

Tune only after profiling identifies the relevant Metal operation. Upstream FA-vector tuning provides a method: sweep constrained candidates by dtype, head size, KV depth and query width, check thermal drift using anchors, and retain baseline protection.

PR #26570 merged FA-vector tuning in August 2026. Later documentation groups tables by Apple GPU family. Historical device rows cannot be transplanted into current tables, and M1/M4 measurements do not establish M5 behavior.

Use an independently pinned llama.cpp checkout and inspect `ggml-metal-tuning --help` before narrowing a sweep. Keep raw timing logs and rejected candidates. The tuner does not validate numerical correctness, and exit zero alone does not prove useful work.

The documented forced numerical matrix covers particular head dimensions, including 128 and 576. Add the target model's actual shapes and boundary cases, then evaluate model quality, long context and PP/TG. A microkernel winner remains a candidate.

Metal Tensor execution depends on GPU capability, library availability and successful kernel compilation. An environment switch or Metal 4 API label does not establish accelerated dispatch. Inspect `has tensor`, GPU family, metallib loading and actual execution.

FA-vector thresholds, preconversion thresholds and sparse-mask conditions are different gates. Read [FA paths](llamacpp-fa-paths.md) before changing them. This guide reports source mechanisms; its tuning recipe was not executed locally and does not authorize edits to frozen host-project compiler revisions.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [llama-fa-tuning-pr](https://github.com/ggml-org/llama.cpp/pull/26570/files) (merged-pr)
- [llama-fa-discussion](https://github.com/ggml-org/llama.cpp/discussions/27668) (discussion)
- [llama-tuning](https://github.com/ggml-org/llama.cpp/blob/master/tools/tuning/README.md) (official-doc)
- [llama-metal-device](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (upstream-code)
- [llama-contributing](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md) (official-doc)
- [llama-metal-fa-ops-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- [llama-metal-fa-buffer-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
