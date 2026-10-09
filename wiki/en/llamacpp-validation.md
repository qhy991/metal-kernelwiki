# llama.cpp Metal validation: executed cases, numerical scope and timing

[中文](../llamacpp-validation.md) · English companion edition

Require successful target-device initialization, exact expected parameter cases, a positive execution count, support and numerical pass. Preserve stdout, stderr and exit status. A success exit can include skipped or empty selections.

The source inspection checked mutable upstream and local test code on 2026-10-07. It did not execute the proposed RMSNorm test. Local source-to-binary identity was unknown. Later [FA checks](llamacpp-fa-paths.md) executed 45 cases but did not retroactively validate this source inspection.

Check operation and parameter filters against the actual binary. Some extra FA loops do not receive the same parameter filter, so narrowing `-p` need not narrow every internal test. List-only CTest commands do not run tests.

Align dtype, shape, strides, epsilon, in-place behavior, input distribution and oracle. Ordinary RMS_NORM lacks weighted multiplication and cannot be directly compared with weighted RMSNorm. NMSE thresholds are not elementwise atol/rtol.

Identify whether comparison executes individual nodes or compares a selected tensor after whole-graph execution. Operator success does not establish fused-graph or model correctness.

The inspected performance path measures repeated graph execution with host timing and backend waiting. Report amortized graph completion time rather than kernel GPU time, model throughput or TTFT. Keep numerical failures under their original thresholds.

Build metadata `b0/unknown` and nearby source files do not prove a commit-to-binary mapping. Missing identity limits attribution. External local notes are unpublished; the public page is not an independently replayable experiment package.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [llama-backend-tests](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp) (upstream-code)
- [llama-backend-comparison](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp) (upstream-code)
- [llama-debugging-tests](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/debugging-tests.md) (official-doc)
- [llama-build-info](https://github.com/ggml-org/llama.cpp/blob/master/cmake/build-info.cmake) (upstream-code)
- local-llamacpp-fa-20261007: `2026-10-07-llamacpp-fa/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
