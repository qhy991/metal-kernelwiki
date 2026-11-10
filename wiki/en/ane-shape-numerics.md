# ANE shape and numeric contract: validator limits, layout and saturation edges

[中文](../ane-shape-numerics.md) · English companion edition

Before compiling for the ANE two decisions are static: can the shape pass the validator, and how will the values degrade on an fp16 datapath with a wider accumulator. This page condenses ch17 validator limits and layout rules, the ch3 saturation/NaN/denormal/determinism boundaries and the ch19 width-slice hazard from the [ANE paper](https://arxiv.org/abs/2606.22283). Numbers keep their [M]/[D]/[P] marks; no local measurement is published. Cost lives in [roofline](ane-roofline-dispatch.md), compressed weight forms in [weights](ane-weights.md), the claimed-versus-reachable split in [reachability](ane-reachability.md) and observability in [execution](ane-execution-measurement.md).

## Validator envelope

Activations are fp16 only (int32 cast and bf16 program I/O rejected on M1); rank is at most 5, with Linear stricter at input rank ≤4; per-axis W and H are 16384 (16385 rejects) and 65536 from A16/M5; C is 65536; the fp16 convolution kernel is capped at kW 13 (14 rejects; the frontend pre-screen is ≤15) and kH 29 with a kernel taller than its input rejected; arg-min/arg-max reduce an axis of at most 2048 (2049 rejects, for fp16 accuracy); matrix multiply needs depth 1 on both operands and output channels inside the 64 KB kernel-memory budget; groups must divide in and out channels, and dividing the four M1 cores keeps the one-to-one lane mapping; the largest single live operand is 2 MB and larger operands tile and stream from DRAM without error.

The 50 `_ANECValidate<Op>Layer` checkers are the back-end legalizer, shared by model segmentation and hand-authored compilation, so acceptance does not drift from real compile outcomes — but it proves schema reachability only. On M1 the sort and dynamic-slice validators accept and the code generator rejects; top-K compiles only outside a forbidden band (K∈{3,4} fails at every width). Only compile-and-run on the target decides.

## Layout

Tensor order is `[N, D, C, H, W]` with rank at most 5 and transforms on the last four axes; a convolution is 4-D NCHW (rank 4 or 5, weights and bias folded into the op); a linear layer reaches the convolution datapath only once rewritten as 1×1 convolution. Tensors are stored channel-interleaved, so the channel count should be a multiple of the interleave factor (the value is unstated; a mismatch pads out and wastes lanes, a warning rather than a reject), and width aligns to the 16-byte DMA granule, which §17.5 checks as `width mod 16`. The 2 MB rule compares the largest single live operand, not the sum; the compiler compares against its 2 MB field (HAL `0x1b8`), while M1/H13 measured the throughput threshold at 2.28–2.34 MB — two bases, not one number. `(1,1,64,56,56)` fills 16-byte lanes; `(1,1,64,56,1)` uses one of sixteen.

## Numerics

fp16 is the arithmetic width for activations, weights, outputs, per-channel scale and bias, and LUT entries. The accumulator is the paper's "wide accumulator of fp32 class", physically wider than fp16 and fixed on every device, with no bit width ever stated — it must not be described as an fp32-precision accumulator. Rounding happens twice: inputs (including the input tile of a reduction, fp16-rounded per 4-lane tile) and the output port. bf16 is a frontend annotation absent from the 11 program-I/O dtype codes and cast to fp16 before the engine segment; fp32 and int32 annotations are accepted but implemented nowhere below, and a cast to int32 is rejected on M1. int4 has no arithmetic lane (it indexes a 16-entry fp16 codebook) and int8 weights are reconstructed to fp16 before the MAC, so compression cuts streamed bytes, not MAC precision.

## Saturation and edge behaviour

| bound | behaviour | condition | mark |
|---|---|---|---|
| fp16 maximum 65504 | silently +inf, no fp32 margin | any written fp16 value | [M] |
| MAC output port 2^15 = 32768 | 32752 passes, 32768 returns inf; an interior partial overflows even if later cancellation would fit | matrix multiply, linear, any convolution with ≥2 taps, regardless of tap count | [M] |
| full-range paths | elementwise multiply overflows at 65504; single-input-channel pointwise 1×1 passes a fill of 60000 | paths not through the MAC output port | [M] |
| reduce-sum | rounds to 65504 first, then overflows | dedicated reduction path | [M] |
| NaN | coerced to +inf at the input boundary, never produced; relu→inf, sigmoid/tanh→1.0, erf→1.0, max→inf, softmax puts all mass on that lane | exception: layer-norm variance reduction sees NaN as a large finite magnitude | [M] |
| indeterminate forms | inf−inf, 0×inf, sqrt(−1), rsqrt(−1) and log(−1) all return +0; −0 loses its sign before reciprocal/rsqrt | forms IEEE would turn into NaN | [M] |
| output rounding | round half to even (2049 at spacing 2 → 2048) | M5 returns 2050 for the same case; rounding-vs-accumulator difference unresolved | [M] |
| denormals | elementwise preserved down to 2^−24 ≈ 5.96e−8; M1 flushes inside the MAC, M5 preserves | M2–M4 unmeasured, so the generation of the change is unknown | [M] |

Determinism belongs to the lowered program: a fixed graph and input are bit-identical across reruns, an independent recompile and a fresh process, and batch position does not change a row. Only changing the math changes bits — `(a+b)+c` versus `a+(b+c)` differs in ~31% of elements by ~1 fp16 ULP, maximum absolute difference ~7.8e−3, each ordering itself reproducible.

## M1 width-slice ×16 hazard

A width-axis slice with a nonzero begin offset routes through a Q.4 fixed-point crop DMA with a fixed ×16 gain, so |v| ≤4094 passes bit-exact (4094×16 = 65504) while 4095 and above rounds to 4096 on the fp16 grid and becomes +inf (4096×16 = 65536). Only the width axis is affected; nonzero offsets on height, channel or batch stay finite, and a zero begin offset avoids the path entirely. Measured on M1 (H13) and A14 (M2); the clean route is placed at A15 and later — decompile-derived for A15 itself, confirmed on M5. Keep loss scales and offset slices under 4094 on the affected generations.

## Pre-build checklist

In order: rank ≤5 (Linear ≤4); the dtype is fp16 with int32 cast and bf16 I/O excluded; per-axis extents inside 16384 (65536 from A16) and 65536; conv kernel W ≤13, H ≤29 and not taller than the input, groups dividing channels and — for speed — the four M1 cores; arg reductions ≤2048; matmul depth 1 and output channels fitting 64 KB kernel memory; largest live operand ≤2 MB; channel interleave-aligned and width a multiple of the granule; output magnitude below 2^15 on matmul and multi-tap convolutions, width slices with a zero offset or |v| ≤4094, and cancellation-heavy reductions anchored on CPU or GPU. The final gate is a compile-and-run on the target chip; a passing validator is not one.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)

[Shared source catalog](../../data/catalog.json)
