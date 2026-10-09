# Retrieval examples

[中文](../examples.md)

Run these CPU-only commands from the repository. For an installed skill, replace `./mwiki` with its absolute `scripts/wiki.py` launcher path.

| Question | Command | Useful next action |
|---|---|---|
| Long-context MLX decode slows down | `./mwiki query "long context decode" --engine mlx --lang en` | Separate KV capacity, memory traffic and execution |
| First token is slow | `./mwiki query "TTFT prefill" --lang en` | Separate loading, queueing and prompt work |
| Should llama.cpp enable FA? | `./mwiki query --engine llama.cpp --tag flash-attention --lang en` | Check support and actual path before comparing |
| Can M4 use an M5 tensor recipe? | `./mwiki get metal-tensors --lang en --follow-sources` | Separate API, SDK, family and hardware |
| GPU work has gaps | `./mwiki query --symptom launch-overhead --lang en` | Trace submission/waiting before fusion |
| Quantized multirow results differ | `./mwiki get quantized-matmul-validation --lang en` | Inspect independent decoded-weight oracle |
| Is a rotating cache safe to batch? | `./mwiki get mlx-cache-lifecycle --lang en` | Validate policy, mask and continuation |
| How do I tune MSL reductions? | `./mwiki get msl-optimization --lang en` | Compare work, scratch, lifetimes and synchronization |
| Does precise division guarantee these bits? | `./mwiki get msl-f32-divide --lang en` | Separate quality, rounding and input provenance |

Retrieve the Chinese detailed record with `--lang zh` when tables or full code are needed. Follow shared source metadata without treating unpublished artifact references as accessible files.

If keyword retrieval misses, narrow terms or search Markdown with rg. Report missing coverage before researching primary sources. Do not invent a page, cross-device result or profiler capability.
