# ANE placement: class, batch and energy decide between CPU, GPU and ANE

[中文](../ane-placement.md) · companion edition

One Apple die carries a CPU, a GPU and the ANE over a shared DRAM pool. Placement is not "which is faster"
but three ordered questions: which workload class, at which batch, at what energy. The numbers below are the
third-party measurements in chapters 10–14 of [the ANE paper](https://arxiv.org/abs/2606.22283) on M1 and M5;
none of it is reproduced here, M2 is only partly measured, and M3/M4 columns are decompiled or predicted.

## The class table

| workload | placement | paper's condition |
|---|---|---|
| 256-channel 3×3 convolution | ANE ≈3.8× faster, ≈9× efficiency vs the same chip's GPU | M1 |
| 16-deep 3×3 conv stack | ANE ≈4.2× faster, ≈13× efficiency | M5 |
| fused five-point stencil, 32 iterations | ANE ≈10× faster, ≈49× efficiency | M5 |
| medium square GEMM | ANE | below the working-set and shape thresholds |
| large square GEMM | GPU ≈2.8× faster, 3.0× at saturation | — |
| short-sequence attention (S≈197) | ANE fastest and most efficient | — |
| long-sequence attention (S≈512+) | GPU ≈2.1× faster, ANE keeps a 1.8× energy edge | — |
| autoregressive decode | GPU at every batch: 2.7× faster, 4.6× more efficient at batch 16 | — |
| trivial op below the 0.23 ms floor | CPU (0.026 ms for a 64×256×256 matmul) | — |

Serving moves the line: an encoder block goes to the GPU past a batch of about 23, a self-attention block past
about 6, while vision convolution stays on the engine at batches 1–256 (§11.6, ch13). These crossovers are M1
measurements; do not transplant them to another chip or sequence length.

## Why decode loses

Decode has an arithmetic intensity near 1 MAC per weight, far below the 141 FLOP/byte ridge, and a layer stack
issues 40–50 dispatches at a ≥0.23 ms floor each — the engine's two weak regimes at once. Switching weights from
fp16 to int8 leaves the step at about 0.99×, because the step is dispatch-bound rather than weight-bandwidth-bound;
the control is batching, which amortizes one round trip over more samples (§14.1, §14.4). Resident KV by buffer
aliasing removes a host copy but not the dispatch count or the weight stream, and the paper states the aliased form
runs at the same rate as one host call per step (§2.6, §6.4).

## Energy is not always the engine's side

The engine is 2–14× more energy-efficient than the GPU on M1, and the margin holds across M1/M2/M5 (ch10). Two
exceptions matter. A dispatch-bound tiny op is the one class where the engine loses outright: the floor dominates
and efficiency falls to about 22 pJ/FLOP against about 0.37 pJ/FLOP at the compute optimum (§10.4). And quantized
weights do not necessarily cut power: instantaneous power stays roughly flat across fp16/int8/int4/sparse
(2.06–2.59 W) while energy falls through shorter latency, and the sparse stream can draw the most watts (§10.6).
Sustained convolution sits near 1.66 W for 176 s with a ≈4.7 W densest peak at 6.96 TFLOP/s, and the engine has no
DVFS and no in-engine thermal logic (§31), so budget a fixed power envelope.

## Splitting work between GPU and ANE

The paper's LLM case is a front/back split, not an arbitrary partition: prefill, encoders and the vision front end
on the engine, decode on the GPU. As checkable rules:

1. Move only compute-bound work (at or above the ridge, 141 FLOP/byte on the M1 basis) that fuses into one segment.
2. Keep each largest live operand at or below 2 MB; above it intensity collapses onto the memory slope, so tile.
3. Fuse the segment into one program: each extra segment pays a launch plus a backend-crossing transfer (0.05 engine
   segment, 0.09–0.23 per crossing in the placement model's relative units, §18.6).
4. Pay one channel-interleaved fp16 repack plus host round trip per boundary, and keep small ops away from boundaries.
5. Size batch or tiles so each segment's compute term clears the dispatch floor (0.23 ms M1, 0.11 ms M5).

The roofline *form* transfers to Metal/MLX placement reasoning, but the constants do not: 141 FLOP/byte, 2 MB and
0.23 ms are engine numbers. What carries over is the order — shrink the working set, amortize the floor, then compare
compute.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)
- [apple-coreml-docs](https://developer.apple.com/documentation/coreml) (official-doc)

[Shared source catalog](../../data/catalog.json)
