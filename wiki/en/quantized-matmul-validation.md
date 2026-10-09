# Quantized matmul: separate quantization loss, decoding and execution error

[中文](../quantized-matmul-validation.md) · English companion edition

Reuse one packed-weight/scales/biases set across candidates. Build independent references from actual stored inputs. Keep ideal affine decoding, decoding rounded to input dtype and kernel accumulation distinct from original-weight quantization loss.

Identical prefix outputs across batch sizes are useful diagnostics, not correctness proofs. The M4/MLX 0.31.2 multirow probe used transpose=False, float32, M=2/3, K=2048/4096, N=128, bits4/8 and groups32/64 with two seeds.

All 32 cases failed the predefined maximum-error gate. First rows were accurate to about 2.84e-7, while later rows had errors 1.587–2.847. Dense execution on the same decoded weights passed; CPU decoding matched native dequantization.

All sixteen M2/M3 prefix comparisons were exactly equal despite those errors. A saved-array CPU model using an incorrect K/8 row stride matched outputs within about 2.91e-7. This supports the historical stride-defect hypothesis but does not prove live binary dispatch.

Do not loosen the gate, inspect only row zero, or accept an unvalidated per-row workaround. Preserve failed packed execution before performance acceptance.

The separate transpose=True [three-route comparison](mlx-qmm-path-comparison.md) passed its limited numerical domain. Quantized [attention](mlx-quantized-attention.md) later found related multi-token PV failures. Neither expands the original probe retroactively or establishes model quality. Raw failure records are unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-splitk-precision-4613](https://github.com/ml-explore/mlx/issues/4613) (issue)
- [mlx-prefill-dequant-4621](https://github.com/ml-explore/mlx/issues/4621) (issue)
- local-mlx-m4-r1: `2026-10-07-mlx-m4-r1/validation/results.json` (local experiment; raw record unpublished).
- local-mlx-qvm-cache-20261007: `2026-10-07-mlx-qvm-cache/results-summary.json` (local experiment; raw record unpublished).
- [mlx-qvm-multirow-fix](https://github.com/ml-explore/mlx/pull/3497/commits/1ea24e11f068af5949cda98e5d3eb0ca5f86ea68) (merged-pr)
- [mlx-qvm-multirow-test](https://github.com/ml-explore/mlx/pull/3497/commits/2ecf184f9150b85b0aa139f7d827751011874d35) (upstream-code)
- [mlx-qvm-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h) (upstream-code)
- [mlx-qvm-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp) (upstream-code)
- local-mlx-qmm-paths-20261007: `2026-10-07-mlx-qmm-paths/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-quantized-attention-20261007: `2026-10-07-mlx-quantized-attention/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
