# ANE quick reference: 187-row operation matrix, per-generation limits and budget numbers

[中文](../ane-quick-reference.md) · companion edition

This page indexes the six ANE topics ([capability surface](ane-reachability.md),
[roofline and dispatch](ane-roofline-dispatch.md), [placement](ane-placement.md),
[shape and numeric contract](ane-shape-numerics.md), [weight forms](ane-weights.md),
[execution and measurement](ane-execution-measurement.md)) and points at the machine-readable
[table dump](../../data/ane-tables.json). All data comes from the third-party paper
[arXiv:2606.22283](https://arxiv.org/abs/2606.22283): M1/M5 measured, other generations decompiled
or predicted, nothing re-measured here. Keep the `M` / `D` / `P` mark on every figure you reuse.

## The data file

One JSON object, seven datasets, 187 + 24 + 9 + 28 + 119 + 17 + 313 + 67 rows:

| dataset | rows | content |
|---|---:|---|
| `ops.rows` | 187 | operation × four families (M1/A13, M2/A14, M3/A15, M4M5/A16-A17) status plus lowering note; `ops.derived` holds class counts |
| `chip_limits.limits` | 24 | table A.1 per-generation numeric limits |
| `chip_limits.tiers` | 9 | table 34.2 capability tiers (dimension cap, 3-D conv, texture engine, sin/cos, dropout) |
| `targets.rows` | 28 | table 34.1 compiler targets, silicon class and NE-core count |
| `metrics.rows` | 119 | roofline, energy, placement, weight-form, numeric, pitfall and execution rows, each with condition, locator and mark |
| `hidden_layers.rows` | 17 | appendix B.1 native layer kinds with family gate and the decision trap (SDPA `SubtractMax`, TopK K in {3,4}, Sort/DynamicSlice validator-only) |
| `workload_tables.groups` | 313 | chapters 7-36 per-class tables: weight forms and latency, energy, class map, conv variants and constants, chip scaling, fp16 divergence axes, decode, training, numeric fit, MAC geometry, memory levels, silicon map, family floors, cost-model clamps and the predicted upper tier |
| `glossary.rows` | 67 | appendix D glossary (AFPP, aned, ANEC, CCDMA, DART, DPE, dispatch floor, wide accumulator, ...) with the four core-fact corrections |

```bash
jq '.ops.rows[] | select(.m1=="No path") | .op' data/ane-tables.json
jq '.metrics.rows[] | select(.group=="weight_forms")' data/ane-tables.json
```

`ops.derived`: 113 native on every family, 108 excluding the five private `ne_`/`pe_` unit rows;
10 bridge-on-M1-then-native; 7 family-gated-on-M1-then-native; 22 permanent bridge; 35 no-path.
That sums to 187, while the paper's own bucket totals sum to 182 — the difference is preserved in
`derived.note`, not reconciled away.

## Reading the operation matrix

Read down a column, not across a name: the same row can move from Bridge to Native by generation
(`resize`, `topk`, `crop_resize`) or stay No path everywhere (`scatter`, `gru`/`lstm`/`rnn`, inverse
trigonometry). The 35 no-path operations are a hard boundary — rewrite them before compiling. On M1
there are also soft boundaries: `gather`/`gather_nd` only inside a software envelope (batch 1, depth 1,
three-element index channel), `sort` and `dynamic-slice` pass the validator and fail code generation,
`topk` compiles only outside a forbidden parameter band and K in {3,4} fails at every width.

Family unlocks: texture engine from A14, native `sin`/`cos` and hardware random from A15, tensor and
transpose extents 16384 → 65536 at A16. A17 and A18 add no operation over A16.

## Per-generation limits (table A.1)

| limit | M1/A13 | A14 | A15 | A16/M5 |
|---|---|---|---|---|
| tensor W/H and D | 16384 | 16384 | 16384 | 65536 |
| tensor C | 65536 | 65536 | 65536 | 65536 |
| tensor N (batch) | 65536 | 65536 | 65536 | 65536 |
| conv kernel W (default / fp16, large) | 29 / 13 | 32 / 16 | 32 / 16 | 32 / 16 |
| conv kernel D (large / small) | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 |
| reduction-to-transpose threshold | 192 | 192 | 384 | 384 |
| instruction alignment | 256 B | 16 B | 16 B | 16 B |
| texture engine | no | yes | yes | yes |
| kernel-memory budget | 64 KB | 64 KB | 64 KB | 64 KB |
| context-switch live tensors | 2 | unlimited | unlimited | unlimited |

Pre-A13 has no 3-D depth axis, no texture engine, a 4096 batch cap; `M11` is a mixed profile (A16
feature flags with the A13 16384 dimension limit).

## Budget numbers (M1 basis unless marked)

Compute roof 12 fp16 TFLOP/s (fused-chain slope; 4.8 for one large matmul, 1.8 for one conv
dispatch); bandwidth 85 GB/s at the memory controller (51 GB/s single-row weight stream, 50 GB/s
compiler constant, 10 GB/s single relu); ridge 141 FLOP/byte on that basis (424 on the M5's other
basis — never convert between them); on-chip working set 2 MB (4.72 MB on M5), compared against the
largest single live operand; dispatch floor 0.23 ms (0.11 ms on M5) with a 190 µs small-call budget
and 130 µs of it in one firmware round trip; fusion amortizes 222 → 6.3 µs per layer from 1 to 32
layers and 196 → 1.5 µs per sample at batch 512; energy 0.37 pJ/FLOP at the compute optimum and
22 pJ/FLOP at the floor (a third, 0.5 pJ/FLOP basis exists and must not be merged).

## Class map (table 11.1, M1)

Engine versus GPU GFLOP/s, GFLOP/s per watt and the engine's efficiency advantage: GEMM floor (K=256) 51/17 vs
46/27 = **0.6x, the engine loses**; GEMM bandwidth (K=1024) 2.3x; GEMM compute (K=4096) 4.0x; single conv at
C=64 1.3x; conv-resnet d=16 14.5x; attention at S=197 2.0x; attention at S=512 2.5x. The headline "3.8x faster
and 9x more efficient" is a 256-channel convolution; at 64 channels the advantage is 1.3x, and at the dispatch
floor the engine loses on energy outright. The full 313 rows (including the weight-stream latency table, the M2 class map, MAC geometry, the above-2 MB throughput curve, the per-generation residency-buffer threshold, the silicon map, the F0-F4 family floors and the H18 fp8 upper tier) are in `workload_tables.groups`.

## Weight forms by generation

int4 LUT and structured sparsity stream on M1 (2.37× and 1.55–1.64× at 0.43× dense bytes); int8
affine folds to dense fp16 on M1 and streams from A14 (0.87× / 0.60× / 0.52× at K=N 2048 / 4096 /
8192); blockwise affine folds through M2 and streams from A15 (predicted). Quantization cuts streamed
bytes, not fp16 MAC throughput, and int8 decode stays at 0.99× because the step is dispatch-bound.

## Placement at a glance

Engine for convolution, vision, encoder front ends (256-channel 3×3: 3.8× faster and 9× more
efficient than the same-chip GPU on M1; vision convolution leads at batches 1–256) and short-sequence
attention (S≈197). GPU for large square GEMM (2.8×, 3.0× at saturation), long attention (2.1× at
S≈512 with an 1.8× engine energy edge) and decode at every batch (2.7× faster, 4.6× more efficient at
batch 16). CPU below the dispatch floor. Serving crossovers on M1: encoder block near batch 23,
self-attention near batch 6.

## Limits of this page

It is an index, not the argument: conditions, counterexamples and the paper's internal contradictions
live in the six topic pages and in each metric row's `condition` and `locator`. Generation columns
other than the measured M1/M5 are decompiled or predicted, and no operation status substitutes for a
compile-and-run on the target chip.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)
- [apple-coreml-docs](https://developer.apple.com/documentation/coreml) (official-doc)

[Shared source catalog](../../data/catalog.json)
