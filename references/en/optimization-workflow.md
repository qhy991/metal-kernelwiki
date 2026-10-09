# Optimization workflow

[中文](../optimization-workflow.md)

Use existing context to establish chip/GPU family, unified memory, OS/toolchain, installed framework, model/quantization, request lengths, concurrency and the target metric. Ask for missing information only when it changes the next useful action.

Separate loading, queueing, prefill, decode and completion. Read [measurement](../../wiki/en/measurement.md), then retrieve the most relevant mechanisms. Preserve the user's runtime choice and reference-access contract.

~~~bash
./mwiki query "long context decode" --engine mlx --lang en
./mwiki get mlx-kv-cache --lang en --follow-sources
./mwiki query "MSL synchronization precision" --lang en
./mwiki validate
~~~

Propose a few structurally different candidates with applicability, expected bottleneck changes, tradeoffs and validation. Start with existing fast primitives and supported framework paths. Write MSL when hotspot evidence supports kernel work.

For deployment, examine capacities, KV policy, quantization, batching, prefix reuse and speculation separately. For MSL, inspect address spaces, physical strides, participation, numerical types and output ownership before tuning thread/tile work.

Validate against an independent oracle before performance acceptance. Use equal workload conditions and repeated timings; obtain profiler evidence to explain mechanisms. A source heuristic, fewer reads or an exported trace is not measured acceleration.

Preserve scientific limits: primitive success does not prove model quality, M4 observations do not establish M5 behavior, and CUDA protocols are not Metal protocols. Respect frozen project revisions and their gates.

Report diagnosis evidence, implemented change, measured outcome, unresolved limits and the next useful check. Cite topic IDs and relevant primary sources. This workflow does not expand task authorization into model downloads, services, formal campaigns or scheduled work.
