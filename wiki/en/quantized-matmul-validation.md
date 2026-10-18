# Quantized matmul: separate quantization loss, decoding and execution error

[中文](../quantized-matmul-validation.md) · English companion edition

Reuse one packed-weight/scales/biases set across candidates. Build independent references from actual stored inputs. Keep ideal affine decoding, decoding rounded to input dtype and kernel accumulation distinct from original-weight quantization loss.

Identical prefix outputs across batch sizes are useful diagnostics, not correctness proofs. The M4/MLX 0.31.2 multirow probe used transpose=False, float32, M=2/3, K=2048/4096, N=128, bits4/8 and groups32/64 with two seeds.

All 32 cases failed the predefined maximum-error gate. First rows were accurate to about 2.84e-7, while later rows had errors 1.587–2.847. Dense execution on the same decoded weights passed; CPU decoding matched native dequantization.

All sixteen M2/M3 prefix comparisons were exactly equal despite those errors. A saved-array CPU model using an incorrect K/8 row stride matched outputs within about 2.91e-7. This supports the historical stride-defect hypothesis but does not prove live binary dispatch.

Do not loosen the gate, inspect only row zero, or accept an unvalidated per-row workaround. Preserve failed packed execution before performance acceptance.

The separate transpose=True [three-route comparison](mlx-qmm-path-comparison.md) passed its limited numerical domain. Quantized [attention](mlx-quantized-attention.md) later found related multi-token PV failures. Neither expands the original probe retroactively or establishes model quality. Raw failure records are unpublished.

2026-10-18, v0.32.3 source (tag, not an observed dispatch): `QuantizedMatmul::eval_gpu` (`quantized.cpp` L1884-1962) picks the kernel by M. With `transpose=True`, M below `get_qmv_batch_limit(K, N, d)` (6-33 by architecture generation, the architecture name's last letter, K and N; L89-149) uses the qmv family (L1825-1866); with `transpose=False`, M < 4 uses qvm (K < 1024) or qvm_split_k. At or above the limit with `transpose=True` and B = 1, qmm_splitk uses 32×32 output tiles and `split_k = max(1, 512 / tiles)`, bounded by `K / k_align` and divisibility, falling back to qmm at `split_k <= 1` (L1152-1202). qmm takes qmm_nax when NAX is available (architecture generation >= 17, >= 18 for a trailing `p`; macOS >= 26.2; `device.cpp` L952-971) and the mode allows it (`transpose` or `mode == "affine"`), K % 64 == 0, N % 64 == 0 when `transpose=False`, and the dtype is not float32 unless TF32 is enabled (`quantized.cpp` L1062-1066). Otherwise it takes the `<mode>_qmm_t` / `<mode>_qmm_n` kernel; for affine mode with `transpose=True` that is `affine_qmm_t` with template defaults BM = BK = BN = 32 (`kernels/quantized.h` L1900-1903). Inferred from the source: the reduction order can change with M, so bitwise equality of two kernels at one M does not carry to another M, or to another K, N, architecture or OS.

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
- [mlx-quantized-dispatch-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp) (upstream-code)
- [mlx-quantized-kernel-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/quantized.h) (upstream-code)
- [mlx-metal-device-0323](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp) (upstream-code)
- local-mlx-qmm-paths-20261007: `2026-10-07-mlx-qmm-paths/derived/summary.json` (local experiment; raw record unpublished).
- local-mlx-quantized-attention-20261007: `2026-10-07-mlx-quantized-attention/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
