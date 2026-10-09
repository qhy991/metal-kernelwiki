# Sources, editions and maintenance

[中文](MAINTENANCE.md)

The catalog owns topic/source identity and retrieval metadata. Wiki pages own synthesis; external experiment directories own raw records. The library covers Apple Silicon Metal, MLX and llama.cpp, without claiming comprehensive ANE, Intel/AMD Metal or distributed inference coverage.

## One catalog, two editions

`data/catalog.json` uses schema version 1. Original topic fields include id, title, path, type, engines, tags, symptoms, confidence, sources and summary. `translations.en` registers the companion title, summary, path and `edition=companion`. Translation does not create a second topic or source.

Chinese records preserve detailed tables, code and history. English guides cover mechanisms, main results and limitations with links back to those records. Update both editions when a technical claim changes. Do not infer that omitted experimental detail was newly verified.

Sources record id, title, kind, checked date, revision, mutability and note. Remote sources use URLs. Local experiments use a logical `artifact_ref` of run-ID/relative-file, without URLs or machine-local absolute paths. Original source notes retain their language.

The checked date is the reading date, not publication date. Mutable main/master is not a pinned snapshot. A merged PR or closed issue does not prove installed support or acceptance.

## Evidence boundaries

Evidence labels distinguish documented mechanisms, source reports, inference, proposed experiments and bounded local observations. A page can contain multiple bases; qualify each claim in context.

Retain execution sources, inputs/seeds, predefined oracle/tolerances, every result, failures, versions and timing samples outside the checkout. New runs use new directories. Keeping files does not establish historical custody or formal qualification.

Structural validation checks local fields and links only. It does not open external raw artifacts, verify their existence, confirm measured claims, visit URLs or establish API support. Source following prints metadata only. Public readers cannot independently replay unpublished records.

## Refresh and publish

1. Select a topic from a real question and read applicable primary documentation or implementation.
2. Separate hardware, OS, SDK and framework conditions. Prefer fixed commit links; record mutable references honestly.
3. Update the affected topic, English companion and source metadata without claiming a whole-library refresh.
4. Run `./mwiki validate` and relevant Chinese/English query/get commands.
5. For behavior changes, exercise retrieval and installation from another working directory. Ordinary prose updates need no hardware experiment.
6. Rebuild committed HTML when the catalog or overview content changes.

Keep failures and superseded observations. Do not enlarge numerical gates after results or normalize an environment failure into a successful original run. No routine digest checks are needed.
