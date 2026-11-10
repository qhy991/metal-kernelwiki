# ANE roofline and dispatch cost model

[中文](../ane-roofline-dispatch.md) · English companion edition

Decide what limits a layer on the Neural Engine — compute roof, memory slope, or the fixed per-dispatch floor — before optimizing it. Every number is the paper's, measured on M1 (H13) unless a row says M5; M2–M4 rows in the paper are decompile-derived or predicted. No local measurement, and no CPU/GPU placement (see [placement](ane-placement.md)) or execution/observability (see [execution and measurement](ane-execution-measurement.md)) here.

## Two ceilings and the ridge

`R(I) = min(P, I·B)` and `t ≈ t0 + work / R(I)`. On M1, `P ≈ 12 fp16 TFLOP/s` (fused matmul-chain slope, fixed I/O), `B ≈ 85 GB/s` (measured at the memory controller), ridge `I* = P/B ≈ 141 FLOP/byte` on that same 12/85 basis, and `t0 ≈ 0.23 ms` per evaluation (§9.1, §9.3).

## Three compute readings, three protocols

12 / 4.8 / 1.8 TFLOP/s are one chip under three protocols, not three peaks. 12 = overhead-isolated slope: a fused matmul chain is one dispatch, so deepening it cancels the fixed cost. 4.8 = absolute single 4096-dim matmul whose call time is far above the floor, about 87% of the ~5.5 convention (fp16 by convention half the marketed ~11 int8 TOPS; not a measured ceiling). 1.8 = one conv dispatch paying the fixed cost back. The paper reads the peak as 1.8 → 3.25 (model fit) → 4.8, while a fused steady state reaches ~8.1 TFLOP/s (§10.5).

## Bandwidth has several bases

85 GB/s DRAM ceiling (the roofline `B`); 51 GB/s wall-clock weight stream (large matmul, one output row, each weight read once, ~60% of 85); 50 GB/s compiler internal constant [D]; 24.1 GB/s M5 standalone activation path; ~10 GB/s M1 single-relu effective stream, capped by dispatch overhead; 9.0 GB/s M1 and 57 GB/s M5 cost-model fits — jointly calibrated effective rates, neither the 51 single-row stream nor the ~40 GB/s broad-shape rate (§9.1, §9.5, §18.1–18.2). The M5 ridge of 424 FLOP/byte therefore uses a different basis (~10.2 TFLOP/s matmul / 18.8 conv against 24.1 GB/s, table 9.1) and is not a rescaling of 141. Likewise 466 FLOP/byte is a 3×3 conv's nominal intensity while ~1076 is a standalone conv block's intensity in §9.6 (~2854 fused).

## The working set is the primary limit

Not the ridge: 2 MB on-chip on M1 (4.72 MB on M5). Counters show the collapse — a 96-layer matmul chain at exactly 2 MB activation moved 426 MB of DRAM per dispatch, intensity fell to 60 FLOP/byte, and the matmul that held 12 TFLOP/s at 1 MB dropped to ~4.8 at 2 MB (§9.2). Only the largest single operand is compared, never the sum; the measured crossing sits slightly above the table value at ~2.28–2.34 MB because the tiler reserves ~0.3 MB; and throughput is *higher* above the threshold than below it, because below it the weight must fill before compute starts while above it tiles double-buffer (§21.1).

## Dispatch floor and fusion

The floor is host-side: ~190 µs per call for a tiny graph, ~98% software and firmware, budgeted ~25 µs bind/runtime/host copy, ~16 µs request build, ~2–3 µs doorbell, ~130 µs firmware round trip, ~10 µs completion (§2.3). At most one firmware command is in flight, so two submission threads serialize at 1.04×. Fusion pays the floor once: latency stays near 0.19 ms from 1 to 32 fused conv-relu layers (per-layer cost ~222 → ~6.3 µs) and per-sample cost falls ~196 → ~1.5 µs at batch 512; a fused three-conv stack reaches 20718 GFLOP/s, 110% of the 18771 single-conv saturation peak, lifting block intensity ~1076 → ~2854 (§9.4, §9.6). Energy follows the same line: ~0.37 pJ/FLOP at the compute optimum, ~22 pJ/FLOP at the floor, with §10.2's ~0.5 pJ/FLOP "in its datapath" kept as a separate basis.

## The compiler's estimate, and how far to trust it

`t ≈ max(cycles/f, bytes/B) + t0`, `f = cores × 4 × eff × clock`, wall time taken at a clock near 0.8× maximum. Per-chip anchors (table 18.1; HAL fields [D], fitted anchors [M]): M1 4 cores, 0.40–1.43 GHz, eff 1.0, fitted peak 3.25 TFLOP/s, fitted bandwidth 9.0 GB/s, t0 0.23 ms; M5 16 cores, 0.74–2.36 GHz, eff ~0.84, 8.9 TFLOP/s, 57 GB/s, 0.11 ms. Fidelity (table 18.1, §18.8): ±17% belongs to the five tuned M1 convolutions only; a 68-graph, 8-family sweep has ~31% median absolute error with only 11/68 inside ±17%, directional in the bandwidth-bound regime where 9.0 undershoots the ~40 GB/s effective rate; attention has no analytic estimate at all and must be timed on device; clock-scaling M1 anchors over-predicts M5 by a mean 99%. The placement model is separate and coarser — engine 800 GFLOP/s / 50 GB/s, GPU 120/40, CPU 20/10, launch 0.05/0.10, transfer 0.09/0.23 in relative units — an ordinal tool that prefers long single-backend runs.

## Static placement from shape alone (paper §9.7)

1. Count `total_flops` and `dram_bytes`; take `I = flops/bytes`, and `working_set` = largest single live operand.
2. If `working_set >` the on-chip threshold (2 MB M1, 4.72 MB M5): bandwidth-bound, reshape or split below it before any other tuning.
3. Else if `work_seconds = flops / R(I) < t0`: dispatch-bound, batch or fuse until compute clears the floor.
4. Else if `I ≥` the ridge (141 on the 12/85 basis; 424 on the M5 basis): compute-bound near the roof, no tuning.
5. Else: bandwidth-bound, compress weights or fuse to raise intensity. Worked case: 3×3 conv at 256 channels, ~466 FLOP/byte and under 2 MB, returns compute/no-tuning.

Two levers remain for an optimizing agent: shrink the working set below the on-chip threshold, and grow work per dispatch until the floor amortizes. Unestablished: the ~98% overhead share holds for one tiny graph only; ch4's HAL table lists a 2 MB matmul working set for every generation while ch9/ch12/ch21 give 2 MB M1 and 4.72 MB M5, and the paper does not reconcile them.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)

[Shared source catalog](../../data/catalog.json)
