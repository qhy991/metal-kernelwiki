# ANE execution and measurement: compile once, resident state, and the observability boundary

[中文](../ane-execution-measurement.md) · companion edition

The ANE is not driven instruction by instruction. The host hands over a compiled program plus operand
buffers, rings a doorbell, and waits for a completion notification; the engine's firmware walks a static
graph on its own. This page collects what that model means for timing, residency and measurement, from
chapters 2, 5, 6, 19, 29–33 of [the ANE paper](https://arxiv.org/abs/2606.22283). Every number is the
paper's third-party measurement on M1/M5, not a result reproduced here.

## One dispatch is mostly not compute

A tiny graph (3×3 conv 8→8, relu, mean) costs about 190 µs per call in a hot loop, of which about 98% is
software and firmware: about 25 µs user-space binding and host fp16 copies, 16 µs request build, 2–3 µs
doorbell, **130 µs firmware round trip**, 10 µs kernel-side completion (§2.3, table 2.3). The driver keeps at
most one firmware command in flight, so two submitting threads serialize at 1.04× (§2.4).

Consequences: hoist compilation out of the loop; do not expect concurrent host calls to overlap; and treat any
work whose compute is under the floor as a dispatch problem, not a kernel problem. On the supported Core ML
path one `evaluate` shows up as three driver-to-firmware requests (input cast, inference, output cast) (§33.3).

## Compile once, dispatch many — but every concrete shape is its own program

The compiled bundle is cached on disk, keyed by each segment's network structure and weight containers plus
the source URL, options and platform; changing a weight, a shape, an operation, the device mask or the
segmenter recompiles, and the first dispatch of a fresh program pays an additional one-time cost to produce
the loadable hardware form (§2.1, §5.2.1).

The graph is static: there is no instruction stream, no program counter, and **no data-dependent branching** —
a fixed trip count loop unrolls, a data-dependent length must be resolved on the host or restructured.
Symbolic shapes parse but do not lower on the direct path, so the practical options are pad to a fixed maximum,
bucket a small length set, or recompile per length (§2.5, §19.8).

## Resident state is aliasing, not a faster path

A buffer object can be bound to the output port of call N and the input port of call N+1, so a KV cache or
optimizer state stays on device and the host sends only the per-step input (§2.6). The paper is explicit that
this does not raise the engine's rate: the same steps issued one host call at a time run at the same speed, so
the win is host CPU and copies, not throughput (§6.4). On M1 the native in-place state engine is stubbed out,
so aliasing is the only route.

## Measurement traps

| trap | what it means | locator |
|---|---|---|
| 98% overhead on a tiny graph | overhead share falls with graph size; the absolute ~190 µs does not | §2.3 |
| firmware round trip is not splittable | the per-task-descriptor latency profiler is gated; 130 µs stays one block | §2.3 |
| warm-eval cost is firmware-bound | counting IOConnect calls overstates dispatch cost | §5.2.2 |
| burst boundaries | ~3-call warmup ramp (call 1 ≈3.6× steady); sub-second idle 1.2–1.7×; first call after a 5 s gap ≈260 ms | §10.5 |
| rails-off re-arm | power-up cost grows to ≈0.5 ms once the idle gap reaches ≈100 ms, then plateaus | §31.3 |
| no frequency/voltage telemetry | the engine has no DVFS; only clock-state residency and throttle triggers are visible | §31.3, §33.3 |
| contention | a co-tenant adds ≈202 µs (2.33×) and even a trivial contender adds a ≈178 µs step | §32.2 |

What an unentitled process can still read: whole-engine DRAM read/write bytes, energy in mJ, clock-state
residency, throttle-trigger counts, a free-running timestamp, and a per-dispatch signpost stream. What is gated:
the 24 per-task-descriptor counters — the runtime zeroes the stats mask for third-party clients, and forcing it
non-zero fails at load because the compiled program has no stats-descriptor section (§33.3). The paper's own
working-set evidence (426 MB per dispatch, intensity 60 FLOP/byte, 79–90 GB/s, ≈0.48 pJ/FLOP) was obtained from
the readable channels, which is a good template for ANE measurement that does not need the gated counters.

## Compile-service hazards

A failed compile restarts the shared compile service; failures arriving faster than the restart interval stall
unrelated compiles, so pace retries at roughly 15 s (§19.4). Compile time for a trainable convolution scales
with batch on M1 (≈1.9 s at 4, ≈35 s at 64, ≈79 s at 128, never finishes at 1000), and a wide reconvergent
fan-in (≈12+ branches) plateaus near 9 s per cluster (§19.5).

## Boundary statement

Everything below Core ML — loader, daemon, IOKit selectors, program container, firmware command protocol — is
private, unsupported, OS-version-fragile and not App-Store-safe (ch6, ch27–30). It is useful for measurement and
for explaining observed Core ML behaviour; it is not a shipping path. Core ML remains the path for shipped
software.

## Registered sources

- [ane-paper-2606-22283-v2](https://arxiv.org/abs/2606.22283) (research-paper)
- [apple-coreml-docs](https://developer.apple.com/documentation/coreml) (official-doc)

[Shared source catalog](../../data/catalog.json)
