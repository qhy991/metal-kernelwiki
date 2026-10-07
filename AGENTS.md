# Metal KernelWiki owner rules

Maintain optimization knowledge through mechanisms, conditions, tradeoffs and scoped observations. Use primary sources and name the version and examined domain. A local primitive test is not a general hardware guarantee or model acceptance; trace export is not profiler analysis.

`wiki/` owns synthesis, `data/catalog.json` owns the page/source index, and external experiment directories own raw facts. Preserve failures and superseded observations. Do not import weights, datasets, credentials, raw model conversations, trace bundles or caches. Do not mutate experiments, Compiler revisions or approvals while maintaining this library.

The discoverable skill is `skill/metal-kernelwiki/`. Its installed `knowledge/` directory links only the canonical data, wiki, references and selected root documents; generated `repository.json` locates the launcher. Do not link the entire repository into the skill: recursive discovery would also find its nested skill template. Keep the locator and knowledge links bound to the same checkout, and keep the machine-local locator out of Git. Avoid duplicating the wiki in the skill package. Querying and validation are CPU-only and do not authorize GPU jobs, publishing or automations.

Before committing, run `./mwiki validate` and a relevant query/get command. When changing launchers or installation, run `python3 scripts/test_install_skill.py`: exercise a temporary installation from another working directory and assert that following symlinks finds only its root SKILL.md. Preserve the evidence boundary described in `MAINTENANCE.md`. No routine digest checks.
