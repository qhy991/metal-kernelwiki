# Metal KernelWiki owner rules

Maintain optimization knowledge through mechanisms, conditions, tradeoffs and scoped observations. Use primary sources and name the version and examined domain. A local primitive test is not a general hardware guarantee or model acceptance; trace export is not profiler analysis.

`wiki/` owns synthesis, `data/catalog.json` owns the page/source index, and external experiment directories own raw facts. Preserve failures and superseded observations. Do not import weights, datasets, credentials, raw model conversations, trace bundles or caches. Do not mutate experiments, Compiler revisions or approvals while maintaining this library.

The discoverable skill is `skill/metal-kernelwiki/`; its installed `knowledge` link points to this canonical repository. Avoid duplicating the wiki in the skill package. Querying and validation are CPU-only and do not authorize GPU jobs, publishing or automations.

Before committing, run `./mwiki validate` and a relevant query/get command. When changing launchers or installation, exercise a temporary installation from another working directory. Preserve the evidence boundary described in `MAINTENANCE.md`. No routine digest checks.
