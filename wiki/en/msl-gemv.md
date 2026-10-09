# MSL GEMV: lane assignment, weight layout and reduction order

[中文](../msl-gemv.md) · English companion edition

For `y[n]=sum_k x[k]*W[n,k]`, divide lanes between independent outputs and K reduction. A matrix tile is only one possible decomposition. Fix actual stored F16 inputs, F32 output and each view's strides.

Contiguous W[N,K], transposed backing and step-two views have different access patterns despite identical logical values. Compare a scalar output owner, an entire SIMD per output and smaller lane groups; ensure every shuffle source is valid and active.

Local M4/MLX 0.31.2 checks covered twelve K/N shapes, including tails and a K65537 cancellation stress case. The initial five-route matrix produced 540 outputs with four precision failures. The eight-lane candidate lost small terms in long-K cancellation despite F32 accumulation.

An independent adjacent-K pairing candidate subsequently passed 108 checks under the same gate. It preserves the original failures rather than replacing them. All 648 outputs were retained and independently checked.

Reduction order determines which terms meet before rounding. Wider accumulation types alone do not prevent cancellation error. Pairing that helps this constructed input is not a universal numerical guarantee.

MLX and llama.cpp source heuristics provide candidates by output size, reduction length and layout, but do not prove actual binary dispatch or speed here. No candidate timing, bandwidth, instruction or model-quality measurement was made. Raw sources and records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-gemv-kernels-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/gemv.metal) (upstream-code)
- [llamacpp-msl-gemv-kernels-20261007](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/kernels/mul_mv.metal) (upstream-code)
- [llamacpp-msl-gemv-pipelines-20261007](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/ggml-metal-device.cpp) (upstream-code)
- [llamacpp-msl-gemv-dispatch-20261007](https://github.com/ggml-org/llama.cpp/blob/42b021b4dc42be573f1e1463528532fc8294c650/ggml/src/ggml-metal/ggml-metal-ops.cpp) (upstream-code)
- local-msl-gemv-20261007: `2026-10-07-msl-gemv/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-metal-matmul-host-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp) (upstream-code)
- [mlx-custom-metal-impl-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/custom_kernel.cpp) (upstream-code)

[Shared source catalog](../../data/catalog.json)
