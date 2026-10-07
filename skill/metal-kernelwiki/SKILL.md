---
name: metal-kernelwiki
description: Retrieve and maintain evidence-backed Metal LLM deployment and kernel optimization knowledge for Apple Silicon, MLX/MLX-LM and llama.cpp. Use for slow prefill/TTFT or decode, quantization, KV cache, batching, attention/GEMM/MoE, MSL programming and optimization (address spaces, SIMD reductions, vector layout, specialization), unified memory pressure and Metal profiling. Also matches MSL 用法、MSL 优化、Metal 优化、Mac 本地大模型部署、MLX 推理慢. Excludes generic graphics, CUDA-only work and ANE-only conversion.
---

# Metal KernelWiki

The installed `knowledge/` directory exposes linked documents from the canonical
repository; `repository.json` locates its command entry. Begin with
`knowledge/README.md` and `knowledge/references/optimization-workflow.md`, then
retrieve the relevant mechanism pages and their registered sources. When using
the repository template directly, these documents live at the repository root.

```bash
python3 scripts/wiki.py query "MLX 长上下文 decode 慢" --limit 5
python3 scripts/wiki.py query --engine llama.cpp --tag flash-attention
python3 scripts/wiki.py query "MSL 地址空间 packed simd_sum" --limit 5
python3 scripts/wiki.py get msl-optimization
python3 scripts/wiki.py get measurement
python3 scripts/wiki.py get quantized-matmul-validation --follow-sources
python3 scripts/wiki.py get local-mlx-m4
python3 scripts/wiki.py validate
```

Commands are relative to this skill, or use the absolute launcher path. They
require only Python 3.9+ and perform CPU-only retrieval. The repository's
`data/catalog.json` is the single page/source index. If a keyword query misses,
try a narrower keyword or a relevant filter; do not invent missing evidence.

Explain ideas through the observed cost, proposed transformation, applicability,
tradeoffs and validation. Keep prefill, decode and service latency distinct.
Check the installed framework/API and exact GPU target before applying a recipe;
M4 observations and M5 capabilities are not interchangeable.

`documented`, `source-reported`, `inferred`, `experimental` and
`locally-measured` describe the evidence basis. A local primitive oracle passing
does not prove model quality or deployment speed. Trace export does not prove
profiler analysis or a fused dispatch. External `artifact_ref` entries identify
unpublished local records; preserve that limitation and do not claim replay.

For MSL authoring, retrieve `msl-programming` for address-space, alignment and
synchronization contracts, then `msl-optimization` for implementation patterns.
Inspect framework-generated signatures and actual strides; source-level vectors,
thread arrays and specialization do not prove vector instructions, register
residency or a speedup. Match language-version rules to the installed toolchain.

For maintenance, read `knowledge/MAINTENANCE.md`, retain source/version/scope,
update the relevant page and catalog, then validate and exercise retrieval.
Keep raw experiments outside the checkout and respect the task's reference-access,
frozen-revision and acceptance rules. This skill does not itself authorize model
downloads, GPU campaigns, service publication or scheduled work; use the user's
actual task authorization.
