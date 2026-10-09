# MLX-LM deployment: capacity, quantization and staged benchmarks

[中文](../mlx-deployment.md) · English companion edition

Fix the chip, memory, macOS, MLX/MLX-LM versions, model revision, tokenizer, quantization and request distribution before comparing settings. The original guide checked live MLX 0.32.3 documentation on 2026-10-07; local probes elsewhere use 0.31.2. Documented features do not establish installed support.

Measure cold loading, warm prefill, decode and user-facing TTFT separately. Include short/long prompts and short/long outputs. Random-token benchmarks without EOS are performance probes, not quality evaluation. Retain individual samples, memory pressure and failed allocations.

Affine quantization supports version-specific bit widths and group sizes. The documented choices include 2/3/4/5/6/8 bits and groups 32/64/128, with 4-bit/group-64 defaults. Verify dimension divisibility. MXFP/NVFP formats do not themselves prove accelerated hardware dispatch.

Retain packed weights through supported quantized operations. Full dequantization on every token can erase storage savings. Compare packed execution, per-call dequantization and resident dense weights only under the same numerical and memory contract.

Smaller prefill chunks can lower temporary memory while reducing prompt throughput. The documented `prefill_step_size=2048` is version-specific. Preserve requested context, sampling and quality when comparing capacity candidates.

Use [measurement](measurement.md), [execution](mlx-execution.md), [KV cache](mlx-kv-cache.md) and [quantized matmul validation](quantized-matmul-validation.md). No complete-model deployment gain has been established by the local primitive probes.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-benchmark](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) (upstream-code)
- [mlx-quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) (official-doc)
- [mlx-quantized-layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py) (upstream-code)
- [mlx-lm-readme](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations) (upstream-code)

[Shared source catalog](../../data/catalog.json)
