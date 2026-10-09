# Local M4 baseline: SDPA, quantized matmul and tool limits

[中文](../local-mlx-m4.md) · English companion edition

The 2026-10-07 probe used Apple M4/16 GB, macOS 27.0 (26A428), MLX 0.31.2, MLX-LM 0.31.3, NumPy 2.4.3 and Python 3.14.3. It loaded no language model.

Sixteen SDPA checks passed independent float64 references on stored F32/F16 inputs. They compared compact GQA against repeated K/V, including unequal query/key lengths and lower-right causal masks. Largest absolute errors were about 8.02e-8 and 2.42e-4.

Host-completion timing included graph creation, evaluation and repeated-KV work. Distributions overlapped and some rankings reversed. No stable acceleration ratio, GPU timing or profiler attribution follows.

Quantized matmul used affine 4-bit/group-64 weights and M=16/32/33/64/65/128. Outputs were finite, but identical input prefixes differed across larger-M paths. BF16 reference narrowing explicitly used float64→float32→BF16. This did not establish Split-K dispatch or model quality.

Tool discovery failed: `xcrun` calls exited 72; a later sandboxed trace-template call exited 137 with unknown cause. Those failures were retained. A separate trace export succeeded but was not parsed, numerically checked or timed.

Allocator active/cache/peak figures are distinct counters, not process RSS or per-candidate physical memory. Raw validation and capture records remain outside the repository and unpublished. Consult the detailed Chinese record for the full matrix and timing table.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- local-mlx-m4-r1: `2026-10-07-mlx-m4-r1/validation/results.json` (local experiment; raw record unpublished).
- local-mlx-m4-capture-r1: `2026-10-07-mlx-m4-r1/capture/results.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
