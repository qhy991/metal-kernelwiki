# ANE write-level levers: op spelling, split-K and palette granularity, measured locally

[中文原页](../ane-write-level-levers.md)

The same arithmetic on the ANE can differ by an order of magnitude, and the difference is not
which backend runs it — it is how it is written. This page records one round of first-hand
measurement on 2026-11-10 across M1 Pro, M4 and M2: changing backend buys at most 1.2x,
changing the spelling buys up to 9.7x. The three actionable levers are op spelling (which
decides placement at all), split-K (reduction-axis chunking) and palette table granularity
(which decides whether a LUT is faster than int8 or 3-10x slower).

Relation to [ANE reachability](ane-reachability.md), [weight forms](ane-weights.md) and
[placement](ane-placement.md): those pages synthesize the third-party
[ANE paper](https://arxiv.org/abs/2606.22283); this one is local re-measurement and addition,
under the discipline in [measurement](measurement.md). Not covered: end-to-end model
acceptance, energy (no passwordless sudo on this host), M3 and M5.

## The three classes of variable

One hardware, one set of weights, one variable changed at a time:

| Variable | Measured span | Character |
|---|---|---|
| Weight format / layout / chunking | **up to 9.7x** | continuous, has an optimum |
| Op spelling (on the ANE at all) | 0% vs 100% placement | **binary** |
| Backend (ANEForge vs Core ML) | <= 1.2x | same graph, roughly equal |

The 9.7x is an M2 4096->12288 GEMM: `int8b64` (48.0 ms, falls off the ANE) against `lut4` (4.9 ms).

## Lever one: op spelling decides placement

**Three mathematically equivalent MIL spellings land on different devices at batch 1.**
M2 / macOS 27.0.1, 4096x4096 GEMV, `ComputeUnit.ALL`, placement read with `MLComputePlan`:

| Spelling | Latency | Placement |
|---|---|---|
| `mb.matmul` | 1.505 ms | **ANE 0/3 (all CPU)** |
| `mb.linear` | 1.462 ms | **ANE 0/3 (all CPU)** |
| `mb.conv` (1x1) | 0.966 ms | ANE 3/3 |

`nn.Linear` converts to `linear`. So **a standard linear layer does not reach the ANE at batch 1**;
Core ML raises nothing and the result is still correct, just computed on the CPU. This is the real
reason behind the `[B,C,1,S]` + 1x1-conv convention: it is not tuning, it is admission.

At T>=64 `matmul`/`linear` return to the ANE (2.55/2.63 TF/s at T=64, placed on the ANE);
**only T=1 falls to the CPU.** conv3x3 has a separate size knee: on M2, `[1,64,32,32]` is all CPU
and `[1,128,32,32]` all ANE; the paper records the M5 Pro knee between 128 and 256. The knee moves
with chip and OS — do not treat it as a constant.

Below the knee, **placing work on the ANE is slower**: at C=32 Core ML on the CPU is 0.090 ms while
a direct ANE dispatch is 0.132 ms. Fixed dispatch cost dominates. "High ANE utilization" and
"faster" are inverted in this regime.

## Lever two: split-K

Split the reduction axis into several convolutions and sum them. The gain grows with K and the
curve is convex with a clear optimum:

| Shape | Chip | Best k_chunk | Speedup | rmse (unsplit -> best) |
|---|---|---|---|---|
| down_proj K=6144 | M1 Pro | 1536 | **2.46x** | 5.47e-04 -> 6.71e-04 |
| down_proj K=6144 | M4 | 768 | **2.83x** | same |
| gate_proj K=2048 | M1 Pro | 512 | 1.13x | 4.98e-04 -> 6.35e-04 |
| gate_proj K=2048 | M4 | 512 | 1.34x | same |

Three constraints:

1. **Larger K is worth more**: K=6144 gives 2.5x, K=2048 only 1.2x. Small-K layers are not worth splitting.
2. **Splitting too finely is a disaster**: k=256 drops to 0.52-0.61x on all three machines. Per-conv overhead eats the gain.
3. **The optimum moves with chip and format.** No portable default: M1 Pro wants 512 for fp16 and 1024 for int8; M4 wants 512 for fp16.

**A side effect is better accuracy, reproduced five times**: chunking shortens each fp16 accumulation
chain. Relative rmse of per-channel int8 fell on M2 random weights (8.12e-03->7.37e-03,
7.35e-03->6.96e-03) and on two sets of real M1 Pro weights (8.66e-03->7.05e-03, 8.85e-03->7.77e-03),
a 19% improvement at best.

**It does not transfer to the Metal GPU**: the same shapes under MLX give 0.72-1.03x, because the
matmul kernel already tiles K and a manual split only adds kernel launches and intermediates. This
is an ANE-specific lever.

## Lever three: palette table granularity

The number of code books in `constexpr_lut_to_dense` decides whether it is fast or 3-10x slow. The
parameter is undocumented and is only exposed by timing.

M1 Pro, real weights `[5632, 4096]`, normalized to a 128-token tile:

| tile | int8 ms | lut6 (one table per row) ms | ratio |
|---|---|---|---|
| 128 | 0.824 | 2.396 | **2.91x** |
| 256 | 0.941 | 6.990 | 7.43x |
| 512 | 0.754 | 6.782 | 8.99x |
| 1024 | 0.690 | 6.702 | 9.72x |

Same bit width, both confirmed entirely on the Neural Engine by `MLComputePlan`; only the table
count differs. **5632 tables (one per row)** against the 352 that Core ML's `palettize_weights`
default `per_grouped_channel, group_size=16` produces — a 16x difference in table count. The
op-level measurement with the grouped form reads lut6 at 3.075 ms (faster than int8's 3.166);
the per-row form on the same shape reads 2.4-7.0 ms.

The paper (§25.2) records the legal palette vector sizes `{1,2,4,5}` but gives no latency by table
count. This page adds that: **table count is a first-class performance parameter, and choosing it
badly turns "faster than int8" into "3-10x slower".**

Byte side holds no surprises: measured `.mlpackage` weight bytes match theory exactly (fp16 1.999x,
int8 1.000x, lut6 0.752x, lut4 0.500x relative to int8); LUT table overhead is negligible.

## Per-layer programs: the 32-layer switching cost

The [placement page](ane-placement.md) left an open question — whether per-layer Core ML programs
degrade into pure floor cost when 32 of them are called in sequence. The local answer is **no, not
floor cost, but a larger switching cost**:

| Measurement | Result |
|---|---|
| One layer's program, resident alone | 3.05 ms |
| All 32 layers resident, one pass | 1185.5 ms = **37.0 ms/layer** |
| Ratio | **12.1x** |

3.05 ms is far above the 0.23 ms floor, so this is not a dispatch-floor effect. The 12x appears when
**32 programs (about 3.2 GB of weights) are resident at once**, consistent with the retained
observation that resident program weights above roughly 2.4 GB reload on every switch. So the cost of
per-layer programs is not launch count but **reload once total resident bytes exceed the working
set**; the fix is fewer resident bytes or fused neighbouring layers, not a larger tile.

## Residency is not always the constraint

In the same round, a 17.2 GB M1 Pro holding 3.20 / 4.03 / 4.88 GB of resident ANE weights recorded
**zero `vm_stat` Swapouts** in every measured window.

This does not contradict the existing "high share is gated by residency" observation, but it does not
extrapolate either: that one came from a 16 GB host. **On a machine with more RAM the wall may simply
not be there** — confirm it exists on the target host with `vm_stat` before engineering around it.
This round built the workaround first and only then found no wall, at the cost of an entire
implementation path.

## Adoption conditions

- All figures are one round of first-hand measurement on 2026-11-10, macOS 27.0.1 / coremltools 9.0,
  on M2 (laptop), M1 Pro (16-core ANE) and M4 (mac mini). **Not re-checked on M3 or M5.**
- Latencies are medians; the Core ML side uses `last_predict_duration_in_nano_seconds` (engine time,
  excluding the Python wrapper) where available, while the direct backend has only wall clock, so the
  comparison is biased in Core ML's favour.
- Accuracy is single-matrix relative rmse against a float64 oracle. **It is not model acceptance.**
- Measurements ran inside a window holding both local Metal locks; one reading that overlapped a
  software install read 2.6% high and was retaken.
- **Random weights mislead**: split-K reads 1.59x on M2 random weights and falls to 1.13x on real
  trained weights. Re-check any write-level conclusion on real weights.
- No power measurement (no passwordless sudo on this host) and no end-to-end model acceptance.

Source `local-ane-write-levers-20261110`.
