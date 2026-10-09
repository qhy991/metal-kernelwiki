# Metal capabilities and deployment boundaries

[中文](../metal-capabilities.md) · English companion edition

Use this page when selecting a backend or diagnosing a missing fast path. Its evidence is official capability documentation; the proposed checks are not local performance results.

Check four independent conditions: SDK symbols, OS availability, device hardware support, and backend support for the actual dtype/shape. Record the chip, memory, OS build, Xcode/SDK, MSL standard, backend revision and selected pipeline. `supportsFamily` and availability guards answer different questions. [Apple feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions)

The capability table read on 2026-05-21 assigns M1/M2/M3–M4/M5 to Apple7/8/9/10. SIMD reductions, matrix operations and Metal 4 tensor availability do not share the same starting point as dedicated matrix acceleration. Confirm the current [capability table](https://developer.apple.com/metal/capabilities/) and query the real device. Do not extend Apple Silicon findings to Intel/AMD Macs.

- A newer GPU without acceleration calls for inspecting dtype, dispatch and pipeline selection.
- A development/deployment mismatch calls for SDK, minimum OS, runtime guards and device checks.
- A dtype fallback should be explicit and retain its cause.
- A faster operator without faster generation calls for separate prefill, decode, sampling and wait analysis.

M5 GPU Neural Accelerators live in shader cores and differ from the separate ANE. Older GPUs can implement TensorOps through optimized shaders; equal acceleration is not promised. [M5 ML talk](https://developer.apple.com/videos/play/tech-talks/111432/)

Retain compilation diagnostics, pipeline identity, correctness and target-framework evaluation. [MLX PR #2772](https://github.com/ml-explore/mlx/pull/2772) records merged support, not proof that an installed build enables it for every model. Check version-specific [llama.cpp Metal interfaces](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-metal.h).

Support determines whether a program can execute. Performance requires measurements on its stated target and inputs. Missing target hardware remains missing coverage; another chip's latency cannot fill that gap.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-feature-detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions) (official-doc)
- [apple-capabilities](https://developer.apple.com/metal/capabilities/) (official-doc)
- [apple-m5-ml](https://developer.apple.com/videos/play/tech-talks/111432/) (official-doc)
- [apple-mlx-nax-pr](https://github.com/ml-explore/mlx/pull/2772) (merged-pr)
- [apple-llama-metal-api](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-metal.h) (upstream-code)

[Shared source catalog](../../data/catalog.json)
