# MSL quantized GEMV: 4-bit unpacking and affine rounding

[中文](../msl-quantized-gemv.md) · English companion edition

The example computes a dot product with F32 x/scales/biases/output and unsigned 4-bit weights packed eight codes per uint32, low bits first. Groups run along K with G=32/64/128 and K divisible by G.

A full SIMD group can own one output row. Load a word, unpack its eight codes and reuse its scale/bias before advancing. This describes source reuse; actual instruction/load counts were not measured.

Real-number algebra allows factoring scale and bias outside group sums. Floating-point arithmetic changes where affine cancellation and dot-product rounding occur. Treat the factored expression as a distinct numerical candidate.

On M4/MLX 0.31.2, 72 input conditions produced 360 dot outputs and 144 decode outputs. Two original per-element routes passed the tested numerical gate. Group-factored and native QMM routes each failed six affine-cancellation cases.

For the diagnostic scale/bias, decoded residual was exactly representable at approximately 1.490116119e-7. Decode controls agreed; the discrepancy was not original-weight quantization loss. CPU stepwise rounding analysis retained the failed common gate.

Source dispatch conditions for quad/fast/ordinary QMV are not profiler proof of actual execution. Do not hide failed shapes behind an unvalidated fallback or enlarge tolerance after observing errors.

No timing, bandwidth, register, model-quality or decode-throughput claim was made. External packed parameters, programs, outputs and audit remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-quantization-api-bindings-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/python/src/ops.cpp) (upstream-code)
- local-msl-qmv-20261007: `2026-10-07-msl-qmv/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-qvm-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/quantized.h) (upstream-code)
- [mlx-qvm-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/quantized.cpp) (upstream-code)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
