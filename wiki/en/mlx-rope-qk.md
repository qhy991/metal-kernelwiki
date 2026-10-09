# Joint Q/K RoPE: positions, layout and numerical gates

[中文](../mlx-rope-qk.md) · English companion edition

Compare separate RoPE(Q)/RoPE(K) with concatenation along heads, one RoPE call and a split at Hq. Batch, token length, dtype, head dimension and positional parameters must match; GQA head counts need not match.

M4/MLX 0.31.2 examined 360 configurations and two routes. Of 720 checks, 486 passed and 234 failed under the original gates.

At B2/T1 and scalar offset 8192, eight tested step-two configurations passed separately and failed jointly. Equal-valued vector-offset controls passed in the corresponding subset. Other long-offset configurations failed both routes, so there are distinct layout/position and precision boundaries.

For a tested F32 long-prompt case, separate-Q maximum error increased from about 5.49e-5 at offset zero to 0.002148 at 8192 and 0.033861 at 131072. Zero-position checks cannot replace long-position validation.

Concatenation allocates and copies. Split views may retain batch strides spanning all joined heads and affect downstream layout/lifetime. One API call therefore does not prove lower cost.

An independent CPU complex-rotation audit preserved the failure set. No timing, profiler or model acceptance followed the failed gates. Scalar/vector control differences are scoped observations, not a universal workaround. Raw arrays and failures remain unpublished.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [mlx-rms-fast-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/fast.cpp) (upstream-code)
- [mlx-rope-api](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html) (official-doc)
- [mlx-rope-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/rope.cpp) (upstream-code)
- [mlx-rope-kernel-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/kernels/rope.metal) (upstream-code)
- [mlx-concat-metal-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/slicing.cpp) (upstream-code)
- [mlx-split-views-0312](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/common/common.cpp) (upstream-code)
- [mlx-rope-batch-fix](https://github.com/ml-explore/mlx/pull/3498/commits/b545a35b9baa8f471a134718a930d97ef46cf504) (merged-pr)
- [mlx-rope-batch-test](https://github.com/ml-explore/mlx/pull/3498/commits/bf6421bfb05d9e39d182432611b99ee9a3dca933) (upstream-code)
- [mlx-release-0320](https://github.com/ml-explore/mlx/releases/tag/v0.32.0) (official-doc)
- [mlx-lm-llama-rope-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/llama.py) (upstream-code)
- [mlx-lm-qwen3-rope-0313](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3.py) (upstream-code)
- local-mlx-rope-qk-20261007: `2026-10-07-mlx-rope-qk/derived/summary.json` (local experiment; raw record unpublished).

[Shared source catalog](../../data/catalog.json)
