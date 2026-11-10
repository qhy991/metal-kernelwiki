# Core ML groups weights along the output axis only: the structural limit on sub-8-bit for the ANE

[中文](../ane-coreml-weight-axis.md) · English companion edition

Use when a projection layer is bound for the Neural Engine, the checkpoint in hand is a published HuggingFace quantization (MLX affine Q4, GPTQ, AWQ, GGUF k-quant), and the question is whether it can be fed as published and whether a hardware-aware weight format has room to exist. This page covers only what the **public toolchain (coremltools 9.0 plus the Core ML compiler) actually places on the Neural Engine**, measured on M1 Pro, M2 and M4. Not covered: which forms the ANE firmware and silicon support, which is the paper evidence in [weight forms](ane-weights.md) and reaches a different conclusion (see "Tension with the paper"); placement tradeoffs are in [placement](ane-placement.md) and intensity and dispatch floors in [roofline](ane-roofline-dispatch.md).

## The rule: the grouping axis decides placement, neither bit width nor group count does

Built directly on the MIL op `constexpr_lut_to_dense` with LUT shape `[gN, gK, 1, 1, 16, 1]`, `gN` grouping along the output axis and `gK` along the input axis, everything else held fixed. A `[3072, 1024]` conv at T=256, reading per-op placement from `MLComputePlan` without executing:

| palettes along output | palettes along input | total | placement |
| ---: | ---: | ---: | --- |
| 1 (per tensor) | 1 | 1 | Neural Engine |
| 192 (per 16 rows) | 1 | 192 | Neural Engine |
| 3072 (**per row**) | 1 | 3072 | Neural Engine |
| 1 | 16 (per 64 inputs) | 16 | **CPU** |
| 3072 | 16 | 49152 | **CPU** |

Three thousand palettes along the output axis are accepted and sixteen along the input axis are not, so **the group count is not the variable, the axis is**. `constexpr_blockwise_shift_scale` behaves the same way: per-channel (one scale per output row) stays, per-block 32 or 64 (one scale per N inputs) sends the whole `conv` and the `identity` after it to `MLCPUComputeDevice`.

The rule was invisible because `palettize_weights` and `linear_quantize_weights` **only expose output-axis grouping** (`per_channel`, `per_grouped_channel`). The raw op accepts either; the compiler accepts one.

Full M4 (Mac16,10) placement table, `layers.12.mlp.gate_proj` `[6144, 2048]` at T=512, conv layout, budget 1.77e-03 (the probe's rule: 1.05x the MLX Q4 GPU arm on the same weights):

| encoding | MIL lowering | placement | package | wired | relative RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| fp16 | `const` | Neural Engine | 25.2 MB | 35.9 MB | 4.98e-04 **PASS** |
| int8 per-channel | `constexpr_blockwise_shift_scale` | Neural Engine | 12.6 MB | 23.5 MB | 8.66e-03 |
| LUT6 per_grouped_channel(16) | `constexpr_lut_to_dense` | Neural Engine | 9.5 MB | 20.2 MB | 2.59e-02 |
| LUT4 per_grouped_channel(16) | `constexpr_lut_to_dense` | Neural Engine | 6.3 MB | 17.1 MB | 1.10e-01 |
| int8 per-block 32 | `constexpr_blockwise_shift_scale` | **CPU** | 13.4 MB | 0.2 MB | 8.15e-03 |
| int8 per-block 64 | `constexpr_blockwise_shift_scale` | **CPU** | 13.0 MB | 0.1 MB | 8.55e-03 |
| uint4 per-block 32 | `constexpr_blockwise_shift_scale` | **CPU** | 7.3 MB | 0.1 MB | 6.12e-02 |

The three CPU rows wire nothing, which is the internal control that this measures Neural Engine residency and not process overhead. Falling off costs 10x: the same shape at per-block 64 runs at 0.77 TF/s on the M2 against 7.63 for int8 on the engine. M1 Pro (2026-10-16), M2 and M4 all behave this way.

## The rule closes sub-8-bit rather than leaving it untuned

A 4-bit palette holds 16 entries. Real checkpoint values do not fit that along the permitted axis: in Clef Flash9B `layers.0.mlp.gate_proj`, one **64-input group holds 14 distinct values**, a whole row (4096 weights) holds **544**, and 16 rows hold 1059.

So four bits suffice only over a narrow group; a narrow group is an input-axis group; and that is the grouping the engine refuses. Three independent confirmations:

- **Exact Q4-as-LUT4 runs on the CPU.** Within an affine Q4 group-64 every weight is `q*scale + bias` with `q` in 0..15, so a 16-entry palette reproduces the group bit for bit. Laid out by hand per (row, 64 inputs) it is indeed exact, and on `[12288, 4096]` it lands on the CPU at 1.12 TF/s against fp16's 5.46 TF/s on the engine, wiring 3.1 MB because nothing is resident there.
- **Finer palettes within the permitted axis buy no accuracy.** Per-row LUT4 is 1.01e-01, per-4-row 9.84e-02, per-16-row 9.95e-02 (real Clef Q4 weights, 1024 and 512 sampled rows, budget 2.47e-03). **Sixteen times the palettes changes nothing**, because each row on its own is too wide for 16 centroids. [Write-level levers](ane-write-level-levers.md) measures the other half of the same parameter on the speed side: one table per row is 3 to 10x slower than `per_grouped_channel(16)`. Finer codebooks within the permitted axis are therefore neither faster nor more accurate.
- **LUT4 is the fastest and smallest cell measured on the M4 and is still unusable.** 10.70 TF/s against fp16's 8.77, wired 17.1 MB against 35.9, relative RMSE 1.10e-01 — worse than the Q4 checkpoint it was re-encoded from.

The usable design space on the Neural Engine is therefore **two encodings**: fp16 (2.0 B/weight, exact against the checkpoint) and per-row int8 (1.0 B/weight, 8.8e-03). Any proposal for a sub-8-bit ANE format has to clear this compiler constraint first, and tuning does not clear it.

## Why published quantized checkpoints cannot be fed as they are

Every mainstream published format groups along the input axis: MLX affine carries one (scale, bias) per 64 inputs, GPTQ and AWQ one per 128, GGUF k-quants one per 32 or 256. **None can be handed to the Neural Engine as published.** This is not weak support; the grouping axis is orthogonal to what the compiler requires.

The consequence is that an ANE path must re-encode to per-row int8 or fp16. That step is forced by the constraint rather than chosen, and a port has to keep the property. Read the re-encoding's accuracy against a stated anchor: against the dequantized Q4 weights, per-row int8 is 8.8e-03; against a bf16 master, a single quantization to per-row int8 is 8.4e-03 where affine Q4 group-64 is 9.2e-02, so **int8 is about 11x more accurate than the 4-bit checkpoint it replaces**; requantizing from Q4 instead gives 9.2e-02, only 0.4% worse than Q4.

## The rule forbids an input-axis encoding, not an input-axis graph

Input-axis granularity is available from the **op graph**, which the compiler allows. A K split cuts one conv into parts along K and sums them; each part is a separate conv with its own weight tensor, so per-row scales on a `K/parts` slice are one scale per (row, `K/parts` inputs).

A second increment: `constexpr_blockwise_shift_scale` already carries a shift. An affine Q4 group's values sit around a nonzero offset, so a symmetric quantizer spends range covering it. Asymmetric per-channel is `mode='linear'` with uint8, and it **stays on the Neural Engine at the same package size** as the symmetric form.

Clef `layers.0.mlp.gate_proj` against a float64 oracle on its own Q4 weights, 1024 sampled rows, budget 2.47e-03:

| K parts | each scale covers | symmetric | asymmetric |
| ---: | ---: | ---: | ---: |
| 1 (as deployed) | 4096 inputs | 8.79e-03 | 8.15e-03 |
| 4 (`k_chunk` 1024) | 1024 | 7.78e-03 | 7.22e-03 |
| 16 | 256 | 6.83e-03 | 6.08e-03 |
| 64 (the checkpoint's own) | 64 | 5.95e-03 | 3.52e-03 |

At `k_chunk` 1024, the split already chosen for speed, the pair is **1.22x** better than the deployed arm at identical bytes, identical placement and a split that is already in the graph. **No cell reaches the budget**, so this does not make an all-int8 share admissible; what it moves is the line an int8/fp16 split sits on, by an amount not yet measured.

## Comparing weight formats without a K split overstates the compressed one

A K split helps fp16 and does not help int8, so an unsplit comparison inflates int8 by roughly the split's own gain. Each arm at its own best, same token count:

| host | shape | fp16 | int8 | unsplit would say |
| --- | --- | ---: | ---: | ---: |
| M4 | `down_proj` [2048, 6144], T 1408 | 1.264 ms/512tok | 1.198 | 2.032x |
| M4 | `gate_proj` [6144, 2048], T 512 | 1.132 ms/512tok | 1.106 | 1.201x |
| M2 | [12288, 4096], T 512 | 6.409 ms | 5.783 | 1.571x |

The real figure is 1.02x to 1.11x where unsplit arms read 1.20x to 2.03x. **Select an encoding on placement, accuracy and residency, then tune the K split separately**; do not mix the two into one comparison.

## Tension with the paper

[Weight forms](ane-weights.md) follows arXiv:2606.22283 §25.2 in listing blockwise affine as one of four supported forms, streaming from A15 by gate-byte inference. This page measures a different layer: **the Core ML compiler does not place a blockwise-grouped conv on the Neural Engine at all** on M1 Pro, M2 or M4, whatever the silicon supports.

The two do not contradict each other, but they must be read apart. The paper decompiles firmware and silicon capability; this page records the public toolchain's actual policy. A hardware capability is not a guarantee that the compiler emits it, and for work restricted to public APIs the binding constraint is the latter. Likewise, the paper's M1 int4 LUT streaming at 2.37x is consistent with LUT4 staying on the engine here, but the paper publishes no per-layer accuracy characterisation, and coremltools' LUT4 measures 1.10e-01 on a real checkpoint and is unusable. "Can stream" and "can be used" are separate questions.

## Adoption conditions

- Adoptable: to judge whether an encoding reaches the Neural Engine, look at which axis it groups along. Output axis, including one scale or palette per row, is accepted; input axis is not. Verify placement per op with `MLComputePlan` rather than inferring it from bit width or an API name.
- Adoptable: a published quantized checkpoint always needs re-encoding to per-row int8 or fp16 before it reaches the engine.
- Not implied: this page does not claim the silicon lacks blockwise or int4 arithmetic support, only that the public Core ML path does not emit them. Private paths are out of scope.
- Accuracy boundary: every relative RMSE names its anchor (the Q4 oracle or a bf16 master). The two differ by an order of magnitude, so figures from different anchors are not comparable. The 1.77e-03 and 2.47e-03 budgets come from the probe's rule (1.05x the GPU arm on the same weights) and are not a model-level acceptance.
- Not established: how many neurons move from an fp16 share to an int8 share once int8's error falls 1.22x; whether the single-program residency result holds at 32 programs; the asymmetric-plus-K-split pair has no end-to-end model acceptance.
- Measurement boundary: all figures are single-shape, single-process operator-level measurements. The M2 is a shared host (wired spread 90 to 145 MB across seven processes) and the M4 ran inside the host's Metal lock. Neither constitutes model acceptance nor profiler analysis.

Source `local-ane-coreml-axis-20261110`. Figures are local measurements with coremltools 9.0 on M1 Pro, M2 and M4, and do not extrapolate to other OS or coremltools versions or to private API paths.
