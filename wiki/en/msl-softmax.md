# MSL softmax: online state, split reduction and empty masks

[中文](../msl-softmax.md) · English companion edition

Define the valid set before choosing an algorithm. This F32 example uses uint32 masks, returns positive zero at masked positions and explicitly defines an all-masked row as all positive zero. That extra empty-row contract is not implied by an API name.

For a nonempty block retain maximum m, normalized exponential sum l and valid count. Merge two states using `m=max(m1,m2)` and `l=l1*exp(m1-m)+l2*exp(m2-m)`. Handle empty states separately to avoid undefined infinity subtraction.

Compare three-pass reductions, sequential online updates and split kernels that produce local state, merge it and normalize output. Each changes reads, serial arithmetic, scratch, launches and dependencies.

Cross-SIMD state requires threadgroup synchronization; cross-kernel state requires actual completion ordering. Uniform collective participation remains necessary at tails.

`precise::exp`, `fast::exp` and MLX `precise=True` are separate controls. Precise exp is not a correctly-rounded guarantee; neither it nor F32 output defines the reduction tree.

M4/MLX 0.31.2 checked twelve widths from one to 8193, two layouts and six routes. All 144 outputs passed probability, sum, nonnegativity, mask/empty-row and shift gates. Twenty-four saved split states also passed independent maximum/count/sum checks.

This computes probabilities from logits, without QK/PV or complete FlashAttention. The original run had no timing. See [timing](msl-softmax-timing.md) and [low precision](msl-softmax-lowp.md); raw records remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-softmax-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.h) (upstream-code)
- [mlx-softmax-dispatch-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/softmax.cpp) (upstream-code)
- [online-softmax-1805.02867-v2](https://arxiv.org/pdf/1805.02867v2) (research-paper)
- local-msl-softmax-20261007: `2026-10-07-msl-softmax/derived/summary.json` (local experiment; raw record unpublished).
- [apple-msl-spec-41](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (official-doc)
- [mlx-quantization-api-bindings-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/python/src/ops.cpp) (upstream-code)
- [mlx-softmax-factory-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/jit_kernels.cpp) (upstream-code)
- [mlx-softmax-defines-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/defines.h) (upstream-code)
- [mlx-softmax-instantiation-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/softmax.metal) (upstream-code)

[Shared source catalog](../../data/catalog.json)
