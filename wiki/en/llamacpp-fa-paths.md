# llama.cpp Metal FA: dispatch, quantized KV and temporary memory

[中文](../llamacpp-fa-paths.md) · English companion edition

Source mechanisms below use upstream commit `36a73916ee0cb3b457f356066afabd47cce68884`. Separate local checks used an existing M4 binary with unknown build commit. They do not validate that upstream snapshot.

An FA operation's query count is not server concurrency or CLI logical batch. Validate F32 Q, equal K/V dtype, supported head pairs, GQA divisibility, mask layout and device features together.

Vector, regular, Tensor and sparse paths have distinct gates. For non-sparse quantized DK=128, the examined source converts K/V to F16 at Nq≥32. DK=512/576 also converts some non-vector Nq20–31 cases. These are source thresholds, not measured M4 optima.

Allocation reserves F16 scratch independently of conversion execution. For separate K/V, one component is `align16(2*DK*Nk*Hkv*B)+align16(2*DV*Nk*Hkv*B)` bytes. DK=DV=128, Hkv=8, Nk=32768, B=1 yields 128 MiB. It excludes other temporary regions and allocator lifetimes.

The sparse `n_kv_max` hint bounds finite mask entries. Finite large-negative values still count. Underreporting can discard valid positions without a dense fallback and changes semantics.

The existing M4 binary executed exactly 45 supported/pass CPU-comparison cases: DK=DV=64, F16/Q8_0/Q4_0 KV, Nq1/3/32, Nk113/512, selected GQA and layouts. NMSE was the numerical gate; arrays and random seed were not exported. Generated masks did not establish causal-block coverage.

No sparse-hint, model, performance, profiler, long-context or current-commit acceptance is claimed. Preserve unknown build identity and unpublished raw-record limits when using these results.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [llama-metal-fa-ops-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- [llama-metal-fa-buffer-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp) (upstream-code)
- [llama-metal-fa-device-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-device.m) (upstream-code)
- [llama-metal-fa-regular-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_f16.metal) (upstream-code)
- [llama-metal-fa-vec-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_vec_f32.metal) (upstream-code)
- [llama-fa-construction-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml.c) (upstream-code)
- [llama-fa-hint-api-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/include/ggml.h) (upstream-code)
- [llama-metal-fa-aux-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_aux.metal) (upstream-code)
- [llama-quant-blocks-36a7391](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-common.h) (upstream-code)
- [llama-fa-predequant-pr](https://github.com/ggml-org/llama.cpp/pull/27390) (merged-pr)
- local-llamacpp-fa-20261007: `2026-10-07-llamacpp-fa/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
