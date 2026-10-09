# Quantized KV attention: isolate QK, softmax and PV

[中文](../mlx-quantized-attention.md) · English companion edition

The installed MLX-LM 0.31.3 helper scales Q, computes quantized QK, applies the mask, uses precise softmax and computes nontransposed quantized PV. Packed KV is not passed directly to ordinary fast SDPA.

The M4/MLX 0.31.2 probe used F32, B=1, Hq=4, Hkv=1/2, D=128, Tk=512/1024/2048/4096, Lq=1/2/3/4 and affine 4/8-bit group-64. Stored packed parameters and independent decoded references separated quantization loss from execution error.

Of 128 configurations, 48 failed their numerical gates. Failures occurred at Tk=1024/2048/4096 and Lq=2/3. QK, softmax and dense PV controls passed; failed execution was localized to quantized PV.

The examined source routes M<4 and K≥1024 to split-K QVM; GQA broadcasting leaves PV's M=Lq in this domain. This is a source-based explanation, not profiler proof of binary dispatch.

Actual payload compression versus F32 KV was 6.4× for 4-bit and approximately 3.556× for 8-bit, including scales/biases at exact tested capacity. It does not establish allocator peak or deployment savings.

The run retained exit 2 without package changes, tolerance changes or performance acceptance. FP16/BF16, other groups, all-masked rows, rotation, models and timing remain untested. Raw records and independent CPU audit are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-lm-base-mask-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py) (upstream-code)
- [mlx-lm-cache-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) (upstream-code)
- [mlx-quantized-frontend-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/ops.cpp) (upstream-code)
- [mlx-qvm-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp) (upstream-code)
- [mlx-qvm-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h) (upstream-code)
- [mlx-qvm-multirow-fix](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68) (merged-pr)
- [mlx-qvm-multirow-test](https://github.com/ml-explore/mlx/pull/3497/commits/2ecf184f9150b85b0aa139f7d827751011874d35) (upstream-code)
- local-mlx-quantized-attention-20261007: `2026-10-07-mlx-quantized-attention/derived/summary.json` (local experiment; raw record unpublished).
- [mlx-release-0320](https://github.com/ml-explore/mlx/releases/tag/v0.32.0) (official-doc)

[Shared source catalog](../../data/catalog.json)
