# TensorOps, quantization and fusion candidates

[中文](../metal-tensors.md) · English companion edition

This is an unmeasured candidate plan for prefill GEMM, quantized linear layers and attention. Profile the framework first.

Large GEMM may justify MPP `mpp::tensor_ops`. Small-batch decode first calls for weight/KV traffic and submission analysis. M5 matrix hardware does not imply equal small-matrix or model speedups. The [M5 ML talk](https://developer.apple.com/videos/play/tech-talks/111432/) identifies 26.1 BF16 tensors, 26.3 cooperative tensor inputs to matmul, and 26.4 INT4/INT8 tensors; check individual SDK declarations.

[MSL matrices](msl-matrix.md) distinguish public 8×8 SIMD-group APIs from cooperative tensors. Its M4 study did not execute MPP/TensorOps candidates.

The [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) emphasizes direct device-memory access and cache reuse for GEMM. Do not transplant CUDA shared-memory staging by default. Candidate dimensions include SIMD/threadgroup tiles, traversal locality, static full-tile extents, K partitioning and register epilogues. Larger tiles can reduce parallelism or increase private-memory pressure. Tail tiles need safe extents and cooperative participation. `mem_none` does not order producer/consumer memory.

[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/) describes OS 27 FP4/FP8/INT2 and E8M0 scales, row reductions and cooperative-tensor compatibility/reuse. Distinguish earlier cooperative-input support from newer helper APIs. Native tensor formats must match stride, offsets, scales and alignment; dtype renaming is insufficient.

Fix target, shape, quantization scheme, tolerance and framework baseline. Compare tile/cache, dequantization and fusion independently. Cover tails, transpose/strides, masks, empty rows, accumulation and task quality. Measure kernel and complete-path costs and inspect actual hardware selection. A smaller checkpoint or successful compilation does not establish faster inference.

## MPP `matmul2d` spellings (headers and specification checked 2026-10-18)

Evidence class: source/documentation (MetalPerformancePrimitives headers of the MacOSX27.0 SDK, Xcode 27.0; MSL 4.1). Nothing here was compiled or run; the Metal Toolchain was not installed, so the `metal_tensor` standard header was not read.

- `run` takes non-const lvalue references (`thread T &`) for left, right and destination; the header example binds `auto mA = A.slice(...)` before calling it (`MPPTensorOpsMatMul2d.h` L134-139, L434).
- The specification lists `slice<SliceExtents...>(index...)` and `slice(index...)` and has no `static_slice`; the MPP header examples use `static_slice<32, 64>(...)` as well as `slice(...)` (MSL 4.1 pp. 94, 100; header L165-167, L205).
- Coordinates: for an M×K `A`, `A.slice(0, tgid.y*64)` gives `mA[x,y] == A[x,tgid.y*64+y]`, and the m=64, n=32 destination tile is called "32 x 64". Index [0] runs along columns (N/K), [1] along rows (M) (header L112-113, L132, L305-309).
- `is_valid_element(idx)` false means access is undefined behavior; `get_multidimensional_index(idx)` returns the logical index (MSL 4.1 p. 109). The header example writes `get_mask(i)`; no declaration of it was found in the MPP headers (L282-283, L302).
- A cooperative destination's `store` has a `static_assert` that the target tensor's element type equals its own (`__impl/MPPTensorOpsMatMul2dImpl.h` L4815-4822); convert explicitly to write another type.

Where header examples and the specification differ, the installed SDK's compiler decides; this page does not say which spelling compiles.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-m5-ml](https://developer.apple.com/videos/play/tech-talks/111432/) (official-doc)
- [apple-mpp-guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) (official-doc)
- [apple-wwdc26-tensors](https://developer.apple.com/videos/play/wwdc2026/330/) (official-doc)
- apple-mpp-sdk-headers-270: MetalPerformancePrimitives headers, MacOSX27.0 SDK (official-doc; read locally, no per-file URL)
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)

[Shared source catalog](../../data/catalog.json)
