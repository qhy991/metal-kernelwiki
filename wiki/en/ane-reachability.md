# ANE capability surface: advertised, compilable, actually executable

[中文](../ane-reachability.md) · English companion edition

A capability byte, a frontend recognition or a passing validator proves one layer only; the paper's standard is a compile-and-run on the target chip ([ANE paper](https://arxiv.org/abs/2606.22283) §4.4). This page condenses the 187-operation census, its five status marks, the per-family floors, the advertised-but-never-lowered set and the operand envelope. Numbers keep the paper's [M]/[D]/[P] marks; no local measurement is published. Cost and placement live in [roofline](ane-roofline-dispatch.md) and [placement](ane-placement.md); weight forms in [weights](ane-weights.md); the syscall-level view in [execution](ane-execution-measurement.md); shape and numerics in [shape-numerics](ane-shape-numerics.md).

## Marks and census

Five marks decide the consequence: **Native** compiles and runs on that family on the direct path (budget engine throughput); **Family-gated** has no path there but is native from the family named (treat as off-engine); **Bridge** is reachable only through a decompose, fallback or compiler-internal route, never standalone codegen (never budget it as an engine kernel); **No path** is rejected on every family M1–M5 (rewrite or compute on the host); **Compiler-internal** is mapped with no observed standalone codegen and reachable only inside a wrapping construct (treat it as part of that construct).

The printed buckets sum to 182 against the 187 operation rows (≈108 native, 9 to M2, 4 to M3, 37 rejected everywhere, ≈24 compiler-internal), and the printed "37 rejected" exceeds the 35 printed No-path rows; both figures are kept instead of merged. A row-level census comes to 113 native (108 net of five private `ne_`/`pe_` rows), 11 to M2, 6 to M3, 22 Bridge and 35 No-path = 187. M1/M2/M5 cells are measured; the M3 column and the M4 half of the merged M4/M5 column are decompile-derived predictions.

## Per-family floors

| family | floor / unlock | mark |
|---|---|---|
| M1 (H13/A13) | ≈108 native ops; no texture engine; no sin/cos; IL-route arg reductions gated; int4-LUT and sparse weights stream; int8 and blockwise-affine fold | [M] |
| M2 (H14/A14) | texture engine (crop-resize, resample, affine, hardware gather); rank/sort bridge; five resize/upsample native; reflect/symmetric pad native | [M] |
| M3 (H15/A15) | native sin/cos; hardware random; whole-tensor arg reductions on the IL route; reduction-to-transpose 192→384; blockwise and sparse-blockwise stream | [P] |
| M4/M5 (A16/A17) | same op set; tensor and transpose extents 16384→65536; blockwise streams; the upper tier adds cores over A16, no new operations (fp8-bearing templates stay inert below H18) | [M] M5 / [P] M4 |

## The route decides the answer

"Unsupported on M1" is meaningless without naming the route: the conversion route is gated by `MinimumFamily<N>` floors (F0 all families, F2 = A13+, F3 = A14+, F4 = A15+, nothing above F4), while direct netplist authoring is gated by per-chip HAL feature bytes read as `hal[offset] & 1`.

- Attention: Appendix A marks fused SDPA native from M1 but "on the matmul + softmax path, not texture-gated"; the reachable M1 form is the matmul+scale+softmax+matmul composition. The compiler does contain a single fused SDPA descriptor (four or five operands; the only parsed key is `SubtractMax`, false by default and required true for a correct softmax), but the high-level path never emits it — only direct netplist authoring through the tunneled-unit path reaches it, which is research-only.
- Whole-tensor argmin/argmax is rejected by conversion on M1 (F4 floor) yet runs via direct authoring on the same M1, because byte `0x4f2` is set from A13. Sin/cos have no direct-authoring bridge and stay conversion-only; sort/top-K have a bridge that the M1 code generator rejects.

## Advertised, never lowered

| item | attested at | why it does not run |
|---|---|---|
| 3-D convolution | HAL capability byte / kernel-depth attestation at `0x70` (16 from A13) | fails backend lowering on every device mask |
| native state / ring buffer | compiler reserves the op classes; descriptor has the fields | M1 descriptor stubs the counter-and-event DMA engine, compile fails |
| bf16 program input/output | documented capability | absent from the 11 `ANECIRDataType` codes; unsupported output dtype |
| flexible/symbolic shapes, sort, dynamic-slice, top-K | symbolic-shape system present with the gate on; validators callable or passing | a "?" dimension parses then fails to lower; M1 codegen rejects the rest, and K∈{3,4} fails at every width |

## Envelope and verification

Envelope: activation dtype fp16 only (int32 cast and bf16 I/O rejected on M1); rank ≤5 with Linear at input rank ≤4; W/H 16384, 65536 from A16/M5, C 65536; conv kW 13 in the fp16 kernel format (frontend pre-screen ≤15), kH 29 input permitting; arg-min/max reduction axis 2048 for fp16 accuracy; matmul depth 1 on both operands and output channels inside the 64 KB kernel-memory budget; largest single live operand 2 MB (correct but tiled and streamed above it). Walk the ladder byte → frontend → validator (50 `_ANECValidate<Op>Layer`, also the back-end legalizer) → lowering → run, and record chip, ANECompiler 9.509.0, macOS floor and the concrete shape; keep clean rejections separate from compiler crashes (dynamic-weight convolution at batch ≥2 crashes as a lost helper application).

The direct route is private, unsupported, OS-fragile and not App-Store-safe: the kernel loads only a daemon-compiled-and-signed program (self-built programs fail with `0xe00002e2`; a missing entitlement gives `0xe00002c7`). Entitlements do not unlock the four hardware-gated features; they add loader-tier convenience only, and Core ML remains the shipping path.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)

[Shared source catalog](../../data/catalog.json)
