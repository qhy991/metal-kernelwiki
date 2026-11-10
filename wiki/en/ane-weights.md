# ANE weight forms: int4 LUT, structured sparsity and int8 stream-or-fold

[中文](../ane-weights.md) · English companion edition

Choose a compressed weight form for a layer already bound for the Neural Engine, and know whether it cuts DRAM bytes on the target or only the file. Evidence: the paper's §7 (M1 measured, per-format gates), §25 (decompiled mechanism and gates), §10.6 (energy). Not covered: per-layer accuracy or tolerance (the paper publishes none) and training-time use (streaming requires an immutable weight); intensity and floor tests are in [roofline and dispatch](ane-roofline-dispatch.md).

## Four forms, one reconstruction op each, and the stream/fold matrix

All four reconstruct to fp16 at the multiplier input, so the multiply stays the ordinary fp16 multiply. int4 LUT: a 4-bit index into a 16-entry fp16 codebook, `w = LUT[g/v][k][c mod v]`, no arithmetic in reconstruction, vector size legal only in {1,2,4,5}, 32 B per 4-bit codebook. Structured sparsity: a 1-bit nonzero mask plus packed fp16 nonzeros, exact apart from the fp16 rounding of kept values, worth it at ≥50% zeros. int8 per-tensor/per-channel affine: one byte plus a fp16 scale, `w = s(q − z)` with `z = 0` on M1, q in [−127,127]. Blockwise affine: int8 bytes with a per-block fp16 scale. Each form must appear as exactly one `constexpr_*` reconstruction op — a count violation is a hard validation error (§7.2). There is **no int4 arithmetic lane**: a 4-bit weight has no underlying scalar type and can only be a palette index (§25.2).

| Form | M1 / A13 | A14 / M2 | A16 / M5 | Selecting gate |
|---|---|---|---|---|
| int4 LUT | stream | stream | stream | master `0x48f` + palette enable `0x529` |
| structured sparsity | stream | stream | stream | master `0x48f`, as a separate mask-and-values operand |
| int8 affine | fold | stream | stream | `0x528`/`0x532`/`0x537`: 0 on M1, 1 from A14 |
| blockwise affine | fold | fold | stream | `0x520`/`0x523`/`0x533`/`0x539`: from A15 (gate-derived, predicted) |

Streaming moves compressed bytes across DRAM and decompresses on chip; folding expands to a dense fp16 constant in DRAM first, so a folded int8 matmul moves the same bytes as fp16 and runs at 1.0× its latency — on M1 the int8 form is a 0.5× stored-size saving only. M1 streams only what its gate bytes admit: `0x48f` is set from A13 and `0x529` on M1, so int4 LUT streams while sparsity streams through the master as its own operand, bypassing the per-format cluster that is clear on M1. The A14 int8 gate is confirmed on M2 silicon; the A15 blockwise path behind that column is predicted.

## Measured gains, with conditions attached

| Measurement | Result | Condition |
|---|---|---|
| M1 int4 LUT stream | 2.37× fp16 | bandwidth-bound 1×1 conv stack; indices are a quarter of the bytes |
| M1 structured sparsity | 1.55–1.64× fp16 at 0.43× dense bytes | six-conv stack, ~63% zeros; §7.3 is the measurement home, §25.5 carries no evidence mark (its bandwidth accounting reads ~29→48 GB/s, 48 being the M1 weight-stream ceiling, and the dense and sparse bundles compile to a byte-identical 2416 B program) |
| A14/M2 int8 stream | 0.87× / 0.60× / 0.52× fp16 | bandwidth-bound matmul at K=N 2048/4096/8192 |
| M2 blockwise (fold) | 0.985× fp16 | near-zero bandwidth gain |
| M5/H17s, all four | 1.6–1.8× fp16 | bandwidth-bound layers |

These are single-workload measurements, not a per-layer model, and the paper offers no extrapolation. Accuracy: int8 tracks an fp32 reference at cosine ≈1.0 with ~0.6% (§3.3) to ~1% (§7.3) relative error — publish the range — against ~0.02% for fp16; sparse is exact apart from fp16 rounding; int4 error is entirely codebook-design error and is unquantified; blockwise "lower error" is qualitative. No form has a published per-layer accuracy or tolerance characterisation, so the gain is a bandwidth effect and must not be read as an accuracy property.

## Bytes, not MAC throughput; energy falls through latency

The datapath is fp16 regardless of storage, so on a compute-bound layer no form changes latency. The decompiled edge: the int8 compile flag quantizes weights but leaves the MAC in fp16, so it halves streamed weight bytes and leaves the eight-channel int8 compute path unused — a 3×3 conv at 256 channels and a square matmul stay within a few percent of fp16, and ~1.5× appears only where the weight must stream from main memory (~4096×4096 at batch ≥256). Do not conflate this with the double-int8 compute mode, which packs two int8 products per MAC and runs ~1.4× measured on conv (trending to 2×): that saves cycles, this saves bytes (§9.1). Decode is a third case — one MAC per weight read, 40–50 small dispatches per step — so int8 weights give 0.99× fp16 because the bottleneck is the dispatch floor.

On an M2 bandwidth-bound 4096-inner matmul power is roughly flat across formats and energy falls through latency (table 10.4): fp16 2.18 W / 0.690 ms / 1.50 mJ (1.00×), int8 2.06 W / 0.431 ms / 0.89 mJ (0.59×), int4 2.29 W / 0.270 ms / 0.62 mJ (0.41×), sparse (~32% dense) 2.59 W / 0.330 ms / 0.86 mJ (0.57×). int4 reaches the same answer at 0.41× the fp16 energy — 2.4× — while also being faster, so there is no efficiency-versus-latency tradeoff; the sparse stream draws the most watts and saves energy by shortening the run. This is one M2 measurement; the paper gives no matching M1/M5 table.

## Engineering limits and how to choose

Two weight budgets decide whether a layer is split: a non-streamable dense weight is sized against `0x200` = **64 KB**, and exceeding it splits the layer into sub-layers — more dispatches, more compile time; a streamable compressed weight is sized against `0x210` = **16 MB** [D]. Making a weight streamable therefore also raises its per-layer budget from 64 KB to 16 MB, a consequence worth as much as the bandwidth gain. Weight sections are also 64-B tiles, `min(8, lanes)` tiles per weight, with a `0xC0` program-image stride for conv weights and `0x40` for matmul [D], so a weight-only edit leaves the descriptor unchanged and needs no recompile; M1 replicates coefficients per core (four copies), A14+ share them; streaming requires an immutable weight, so compressed streaming is strictly inference-time.

1. Confirm the layer is bandwidth-bound (below the ridge, or past the working-set threshold); if not, no form changes MAC or latency. Then pick by generation: M1/A13 streams only int4 LUT and structured sparsity, while int8 and blockwise fold and buy stored size only; from A14 int8 streams and deepens with weight size (0.87× → 0.52× at 2048 → 8192 wide); from A15 blockwise streams too (gate-derived, predicted); on M5 all four stream at 1.6–1.8×.
2. Take structured sparsity first at ≥50% zeros — the one M1 streaming form with no reconstruction arithmetic and exact apart from fp16 rounding; otherwise on M1 take int4 LUT (16-entry fp16 codebook, no reconstruction arithmetic), and fall back to int8 only where 16 levels are too coarse or its stored size is the point.
3. Check both budgets last: does a non-streamable weight cross 64 KB and split the layer, and does a streamable weight stay under 16 MB?

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)

[Shared source catalog](../../data/catalog.json)
