# Metal KernelWiki

[中文](README.md) · [English HTML overview](docs/index.en.html) · [中英文 HTML 总览](docs/index.html)

Evidence-backed LLM deployment and kernel optimization knowledge for Apple Silicon, Metal, MLX/MLX-LM and llama.cpp. Guides connect optimization mechanisms with conditions, costs, numerical behavior and primary sources.

The library has **44 topics and 177 registered sources**, including **25 local-experiment records**. Every topic has a detailed Chinese record and an English companion guide. The English edition covers mechanisms, main observations and limits; it is not a line-by-line translation of every table, code listing or historical record. Shared source metadata keeps its original language.

Local M4 primitive checks do not establish model quality or deployment speed. Exporting a trace does not establish profiler analysis. Raw experiments remain external and unpublished.

## Start here

- [Optimization workflow](references/en/optimization-workflow.md) and [measurement](wiki/en/measurement.md).
- [MLX deployment](wiki/en/mlx-deployment.md), [execution](wiki/en/mlx-execution.md), [KV cache](wiki/en/mlx-kv-cache.md) and [serving](wiki/en/mlx-serving.md).
- [llama.cpp deployment](wiki/en/llamacpp-deployment.md), [FA paths](wiki/en/llamacpp-fa-paths.md), [tuning](wiki/en/llamacpp-metal-tuning.md) and [validation](wiki/en/llamacpp-validation.md).
- [Attention](wiki/en/attention.md), [GEMM/MoE](wiki/en/gemm-moe.md), [quantization](wiki/en/quantization.md) and [fusion](wiki/en/fusion.md).
- [MSL programming](wiki/en/msl-programming.md), [optimization](wiki/en/msl-optimization.md), [matrix operations](wiki/en/msl-matrix.md), [tiles](wiki/en/msl-tiles.md), [GEMV](wiki/en/msl-gemv.md) and [quantized GEMV](wiki/en/msl-quantized-gemv.md).
- [Softmax](wiki/en/msl-softmax.md), [timing](wiki/en/msl-softmax-timing.md), [low precision](wiki/en/msl-softmax-lowp.md), [half arithmetic](wiki/en/msl-half-arithmetic.md) and [F32 division](wiki/en/msl-f32-divide.md).
- Browse all topics in the HTML overview or [single catalog](data/catalog.json).

## Offline retrieval

Only Python 3.9+ standard library is required. Retrieval loads no model, contacts no service and starts no GPU work.

~~~bash
./mwiki query "MSL reduction precision" --lang en --limit 5
./mwiki query --engine llama.cpp --tag flash-attention --lang en
./mwiki get mlx-kv-cache --lang en --follow-sources
./mwiki get wiki/en/msl-f32-divide.md
./mwiki get measurement --lang zh
./mwiki validate
~~~

The default edition is Chinese (`zh`); `--lang en` selects English titles, summaries, bodies and paths. Topics keep the same IDs and sources in both languages. Exact registered paths select their edition unless `--lang` overrides it. Source following prints unchanged metadata rather than fetching experiments.

Search supports Chinese/English aliases and `--engine`, `--type`, `--tag`, `--symptom`, `--limit`, `--json` and `--paths-only`. It is keyword retrieval, not semantic search.

## Install the skill

The skill name is **metal-kernelwiki**. Its thin entry point follows the independent knowledge-repository organization of [BW1100 KernelWiki](https://github.com/qhy991/bw1100-kernelwiki).

~~~bash
git clone https://github.com/qhy991/metal-kernelwiki.git
cd metal-kernelwiki
python3 scripts/install_skill.py
python3 ~/.codex/skills/metal-kernelwiki/scripts/wiki.py query "slow prefill" --lang en
~~~

The installer uses CODEX_HOME/skills or ~/.codex/skills and accepts an explicit `--dest /absolute/path/metal-kernelwiki`. Adjust the example launcher path if CODEX_HOME is set. It refuses existing destinations. The installed skill copies only its thin entry point, links selected knowledge directories/documents and stores a machine-local repository locator.

Wiki updates remain visible through those links. After moving the checkout or upgrading the entry point, move the old installation outside skill scan directories and reinstall. Do not link the whole repository into a skill: its nested template creates duplicate discovery.

## HTML and maintenance

Open `docs/index.html` for the bilingual overview or `docs/index.en.html` for the English overview. GitHub's file viewer shows HTML source; download/clone and open locally to render it. The committed HTML is standalone, with navigation, topic links and light/dark controls.

Rebuild the HTML from the catalog using [the build instructions](docs/README.md). Markdown remains the knowledge authority; HTML is a generated browsing view.

Read [maintenance](MAINTENANCE.en.md), [provenance](PROVENANCE.en.md) and [examples](references/en/examples.md). Validate structure and relevant retrieval before commits; run installation regression checks for launcher/installer changes. No GPU rerun is needed for language or packaging edits.
