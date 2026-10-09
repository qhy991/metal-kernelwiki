# TensorOps, quantization and fusion candidates

[中文](../metal-tensors.md) · English companion edition

This is an unmeasured candidate plan for prefill GEMM, quantized linear layers and attention. Profile the framework first.

Large GEMM may justify MPP `mpp::tensor_ops`. Small-batch decode first calls for weight/KV traffic and submission analysis. M5 matrix hardware does not imply equal small-matrix or model speedups. The [M5 ML talk](https://developer.apple.com/videos/play/tech-talks/111432/) identifies 26.1 BF16 tensors, 26.3 cooperative tensor inputs to matmul, and 26.4 INT4/INT8 tensors; check individual SDK declarations.

[MSL matrices](msl-matrix.md) distinguish public 8×8 SIMD-group APIs from cooperative tensors. Its M4 study did not execute MPP/TensorOps candidates.

The [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) emphasizes direct device-memory access and cache reuse for GEMM. Do not transplant CUDA shared-memory staging by default. Candidate dimensions include SIMD/threadgroup tiles, traversal locality, static full-tile extents, K partitioning and register epilogues. Larger tiles can reduce parallelism or increase private-memory pressure. Tail tiles need safe extents and cooperative participation. `mem_none` does not order producer/consumer memory.

[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/) describes OS 27 FP4/FP8/INT2 and E8M0 scales, row reductions and cooperative-tensor compatibility/reuse. Distinguish earlier cooperative-input support from newer helper APIs. Native tensor formats must match stride, offsets, scales and alignment; dtype renaming is insufficient.

Fix target, shape, quantization scheme, tolerance and framework baseline. Compare tile/cache, dequantization and fusion independently. Cover tails, transpose/strides, masks, empty rows, accumulation and task quality. Measure kernel and complete-path costs and inspect actual hardware selection. A smaller checkpoint or successful compilation does not establish faster inference.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-m5-ml](https://developer.apple.com/videos/play/tech-talks/111432/) (official-doc)
- [apple-mpp-guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) (official-doc)
- [apple-wwdc26-tensors](https://developer.apple.com/videos/play/wwdc2026/330/) (official-doc)

[Shared source catalog](../../data/catalog.json)
