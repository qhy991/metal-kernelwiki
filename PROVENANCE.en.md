# Provenance and reuse scope

[中文 detailed record](PROVENANCE.md)

This library originated as metal-llm-optimization and was renamed metal-kernelwiki. Its repository plus thin installed skill follows the organization of [qhy991/bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki), inspected at commit 52ae9a1. No BW1100 topic text, experiments, retrieval implementation or hardware conclusions were copied.

Primary Apple, MLX, MLX-LM and llama.cpp references are registered in [the catalog](data/catalog.json), including reading dates, version mutability and specific uses. Source reports, documented mechanisms, inference and local measurements remain distinct.

All 44 topics now have English companion guides alongside detailed Chinese records. English guides summarize mechanisms, main observations and limits rather than duplicating every historical table or source listing. Both editions share source IDs; language additions introduce no new measurement evidence.

Twenty-five local-experiment entries refer to external raw records. These include M4 SDPA/quantized-matmul observations, lazy-evaluation boundaries, cache lifecycle/mask/persistence failures, activation/normalization comparisons, limited llama.cpp FA checks, and MSL programming, matrix, tile, GEMV, softmax and division probes.

The detailed Chinese provenance record retains per-run history. Important limits include unknown llama.cpp binary provenance, unpublished numerical arrays, host-completion timings rather than GPU timestamps, failed numerical gates, incomplete first probes and independent successors that preserve earlier failures.

The repository publishes derived interpretation and logical artifact references. It does not publish raw scripts, arrays, logs, trace bundles, models or the source-ID-to-machine-path mapping. Readers cannot independently replay the original runs from this checkout.

The HTML overview is a generated catalog view. It is not another evidence store or scientific qualification. The public repository does not assign licenses to third-party material or interpret citations as permission to redistribute complete upstream text or code.
