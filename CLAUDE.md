# Dewey

GLOSSARY.md holds this project's words for stars, summaries, clusters and the map. Use them in code, flags, file names and prose.

`mise run check` is the project's check.

`data/repos/` is committed on purpose. It holds every fetched repository with its summary, and the summaries cost API money to regenerate, so leave it out of any cleanup and keep its layout as it is.

A third-party library without type information gets a stub under `typings/`, one file per module, declaring only the members this project calls. Add the member you need there; pyright does not honour `# type: ignore` here.

The cluster-name cache key includes the prompt text, so editing `src/dewey/prompts/cluster_name.md` renames clusters on the next run. Editing `src/dewey/prompts/summary.md` changes nothing until the summaries are regenerated with `--overwrite-summaries`.
