# MLX async evaluation: dependency chains and completion boundaries

[中文](../mlx-async-evaluation.md) · English companion edition

The M4/MLX 0.31.2 probe compared 32 dependent tanh-matmul steps with per-step sync evaluation, per-step async submission plus final evaluation, and evaluation only at the dependent tail. An independent-output tail case performed different work and was kept separate.

Actual stored float32 inputs defined an independent float64 oracle. All four modes passed atol=1e-5/rtol=1e-4 across three same-input processes; maximum dependent-chain error was about 1.251e-6.

Each process used three warmups and twelve samples with rotated mode order. Median-of-process-medians completion times were 11.618 ms for per-step sync, 4.160 ms for async, 1.748 ms for dependent-tail evaluation and 0.629 ms for independent outputs. These summarize different host execution arrangements; individual samples varied substantially.

The interval included graph/submission/completion work rather than GPU timestamps. Default-stream behavior, same-seed repeats and a cooperative lock do not control every application, thermal condition or cache. The first process overlapped a CPU installation test.

Use this result to propose fewer unnecessary host waits while preserving dependencies. Do not remove required synchronization or infer kernel acceleration. Real token generation, multiple streams, model quality and service throughput remain untested; raw records are unpublished.

Command-buffer commits (v0.31.2 and v0.32.3 source, 2026-10-18; commit boundaries not observed locally): after each primitive is encoded, `needs_commit()` ends encoding and commits (`eval.cpp` L59-64). It is true when `buffer_ops_ > max_ops` or `(buffer_sizes_ >> 20) > max_mb` (`device.cpp` L511-514; v0.31.2 L445-447). `buffer_ops_` counts dispatches (L408-421); `buffer_sizes_` adds the `data_size` of each buffer first bound in the command buffer, outputs included through `set_output_array` (L345-351, L360-366); both reset on commit (L558-561). Limits follow the last letter of the architecture name: `p` 20 ops / 40 MB, `g` 40 / 40, `s` 50 / 50, `d` 50 / 50, others 40 / 40 (L603-624; same in v0.31.2 L499-520); `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` override them (`mlx/utils.h` L193-202). Inferred from the source: dispatch count and bound bytes are separate counters, so an extra output allocation can move the commit point with no change in dispatch count; record command-buffer boundaries directly instead of inferring them from dispatch counts.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html) (official-doc)
- [mlx-async-eval-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html) (official-doc)
- [mlx-synchronize-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html) (official-doc)
- [mlx-eval-implementation-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/transforms.cpp) (upstream-code)
- local-mlx-boundaries-20261007: `2026-10-07-mlx-boundaries/summary.json` (local experiment; raw record unpublished).
- [mlx-metal-device-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp) (upstream-code)
- [mlx-metal-device-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/device.cpp) (upstream-code)
- [mlx-metal-eval-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/eval.cpp) (upstream-code)
- [mlx-env-utils-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/utils.h) (upstream-code)

[Shared source catalog](../../data/catalog.json)
