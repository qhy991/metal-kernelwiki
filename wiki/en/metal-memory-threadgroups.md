# Metal memory, threadgroups and synchronization

[中文](../metal-memory-threadgroups.md) · English companion edition

These are documented candidates for reductions, GEMV, normalization, RoPE and small matrices. They are not universally validated device recipes. See [MSL contracts](msl-programming.md) and [optimization patterns](msl-optimization.md).

Query pipeline `threadExecutionWidth`, `maxTotalThreadsPerThreadgroup` and static threadgroup memory. Legal limits depend on the kernel's resources. Compare execution-width multiples and per-thread work; maximum legal size is not necessarily fastest. [Grid sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

Do not assume every Metal GPU has the same SIMD width. Nonuniform dispatch support does not make partial groups safe for matrices, collectives or barriers. Tail implementations must preserve participation and bounds.

Unified memory still requires ordering. Shared resources suit CPU-filled/GPU-read data; private resources are candidates for GPU-only intermediates. Shared access does not authorize CPU mutation before GPU completion. Private on Apple Silicon does not imply separate VRAM. Memoryless textures are not LLM buffer storage. [Storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

Reuse buffers, keep intermediates on GPU, and reduce readback/conversions when the real dependency graph allows it. Independent requests may overlap CPU/GPU work; dependent token generation does not inherit rendering's buffering pattern automatically.

SIMD reduction is not shared-memory ordering. All participating threads must reach the required barrier, even in a single SIMD group with dependent memory access. Low occupancy is useful only with ALU/bandwidth and grid/resource evidence. Check threadgroup memory, live registers, cache/MMU and spills before changing tile size. [Apple9 profiling](https://developer.apple.com/videos/play/tech-talks/111374/), [M5 profiling](https://developer.apple.com/videos/play/tech-talks/111431/)

Validate tails, tiny/long rows, masks, strides and cancellation against a fixed oracle. Retain output errors, timing boundaries, allocation/copy/wait costs and available counters. Missing counters are coverage limitations. Restore correct ordering before performance tuning.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-threadgroup-sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes) (official-doc)
- [apple-silicon-porting](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon) (official-doc)
- [apple-storage-modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus) (official-doc)
- [apple-shared-storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) (official-doc)
- [apple-cpu-gpu-sync](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work) (official-doc)
- [apple-m3-profiling](https://developer.apple.com/videos/play/tech-talks/111374/) (official-doc)
- [apple-m5-profiling](https://developer.apple.com/videos/play/tech-talks/111431/) (official-doc)

[Shared source catalog](../../data/catalog.json)
