# MSL softmax timing: full call cost and statistical scope

[中文](../msl-softmax-timing.md) · English companion edition

This M4/MLX 0.31.2 comparison reused numerically accepted F32 softmax functions and six input conditions: R8, K128/257/8193, contiguous/step-two layouts and mixed masks. It did not execute complete attention.

Six routes included one-SIMD, cross-SIMD, online and split-kernel work plus a native masked wrapper. Include mask expressions, temporary split states and output consumption in the common call boundary.

Three sequential processes alternated route order across blocks and rotated conditions. Each route/condition/process retained one first, two warmups and twelve timed outputs. All 1620 outputs passed numerical checks; 36 metadata checks were also retained.

Reports use microseconds and median [min,max] of three process medians. This is not the range of every sample or a confidence interval. Other aggregate statistics can change rankings.

Timing starts at new call/graph construction and ends at evaluation completion. It includes framework, submission, allocation and waiting. Native wrapper layout handling differs from custom direct-stride handling, so a step-two difference cannot be assigned to a single identical softmax kernel.

Fewer source passes exchange memory work for exponential rescaling and serial dependencies. More threads and fewer launches are hypotheses, not winners.

No GPU timestamps, counters, instructions, register data, model throughput or stable universal width threshold was established. Preserve slower samples and process variation. Raw arrays and timings remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-msl-softmax-timing-20261007: `2026-10-07-msl-softmax-timing/derived/summary.json` (local experiment; raw record unpublished).
- local-msl-softmax-20261007: `2026-10-07-msl-softmax/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-softmax-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.h) (upstream-code)
- [mlx-softmax-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/softmax.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
