# Dewey

Dewey draws a map of one GitHub user's starred repositories. Each star becomes a point. Repositories that do similar things sit close together in named clusters, so a few thousand stars become something you can browse.

For every starred repository it fetches the metadata and README, has an LLM write a one-paragraph technical summary, embeds the summaries, reduces them with UMAP, clusters them with HDBSCAN, has the LLM name each cluster from its most central repositories and distinctive words, and writes an interactive HTML map. Hover a point for the summary, click it to open the repository.

The words used here (star, summary, cluster, unclustered, central repository) are defined in [GLOSSARY.md](GLOSSARY.md).

## Setup

Requires [mise](https://mise.jdx.dev/) and a GitHub token with public read access.

```bash
mise install          # python, uv, ollama at the pinned versions
uv sync               # the virtualenv and every dependency
```

Tokens go in `.env`, which mise loads:

```bash
echo "GITHUB_TOKEN=$(gh auth token)" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

## Run

```bash
uv run dewey joaofnds
```

This writes `map.html`. The first run of a user lists their stars, fetches each repository once, and writes one summary per repository with Claude (`claude-sonnet-5` by default). Later runs reuse everything under `data/`. Pass `--refresh-stars` to list the stars again and `--overwrite-summaries` to rewrite every summary.

To run without an API key, use a local model through Ollama:

```bash
ollama serve
ollama pull gemma3:4b
uv run dewey joaofnds --llm ollama --llm-model gemma3:4b
```

The knobs that change the map most:

| Option | Default | Effect |
| --- | --- | --- |
| `--dimensions {2,3}` | 2 | 2-D maps carry cluster names on the map; 3-D maps rotate |
| `--min-cluster-size N` | 15 | smallest group HDBSCAN will call a cluster |
| `--min-samples N` | 5 | higher values leave more repositories unclustered, but clusters get tighter |
| `--embedding-model NAME` | `Qwen/Qwen3-Embedding-0.6B` | any sentence-transformers model |
| `--seed N` | 42 | UMAP's random state; the same seed and data give the same map |

`uv run dewey --help` lists the rest.

## How the map is built

1. **Stars.** The user's starred repositories come from the GitHub API. Each is stored once under `data/repos/<id>/` as the repository object and the README contents object, exactly as GitHub returned them, and the star list is cached in `data/starred_ids.txt`.
2. **Summaries.** The LLM gets the repository's name, description, language, license, size, year, topics and the first 4,000 characters of its README, and returns a dense paragraph written for clustering (the prompt is `src/dewey/prompts/summary.md`). Summaries are stored beside the repository and never rewritten unless asked. Summaries rather than raw READMEs go into the embedding because READMEs vary from a badge wall to a book, and the summary normalizes them to the same register and length.
3. **Embeddings.** Summaries are embedded with a sentence-transformers model and L2-normalized. Vectors are cached in `data/embeddings/` keyed by model name and text, so changing either re-embeds.
4. **Two reductions.** UMAP reduces the vectors twice: to five dimensions with `min_dist=0` for clustering, and to two or three dimensions for the map. Clustering on the plotted coordinates, which is what the first version did, throws away structure that the plot does not need but the clusterer does.
5. **Clusters.** scikit-learn's HDBSCAN clusters the five-dimensional points and gives each repository a membership probability. Repositories it cannot place are unclustered and drawn in grey.
6. **Names.** For each cluster the LLM sees the repositories with the highest membership probability and the cluster's top TF-IDF terms, computed with one document per cluster, and answers with a two-to-four-word category (the prompt is `src/dewey/prompts/cluster_name.md`). Names are cached in `data/names/` keyed by model, prompt, clustering and texts.
7. **Map.** Plotly draws one trace per cluster and, on 2-D maps, each cluster's name at its median position.

### Why these models

Measured on this user's 2,872 summaries on an Apple M5 Pro on 2026-09-21, with UMAP to five dimensions and HDBSCAN at `min_cluster_size=15, min_samples=5`:

| Embedding model | Embed time | Clusters | Unclustered | Silhouette (5-D) |
| --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` (the first version's model) | 10 s | 49 | 18.6% | 0.491 |
| `Qwen/Qwen3-Embedding-0.6B` | 60 s | 57 | 21.4% | 0.571 |

Qwen3 separated clusters better at every setting tried, at six times the embedding cost, and became the default. Both models run locally. The first run downloads the weights (about 1.2 GB for Qwen3).

The LLM writes the summaries and the cluster names, so pick it by cost and writing quality. Regenerating all summaries with `claude-sonnet-5` is roughly seven million input tokens; the [Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing) halves that price but is not wired in.

## Data

`data/repos/` is committed because it holds the fetched repositories and their summaries, and the summaries are the expensive part. `data/embeddings/`, `data/names/` and `data/starred_ids.txt` are caches and are ignored by git. Delete a cache to force that stage to run again.

## Development

```bash
mise run check    # ruff format, ruff lint (all rules), pyright strict, pytest
mise run fix      # format and auto-fix
```

The test suite runs the whole pipeline through in-memory fakes of GitHub, the LLM and the embedder, so it needs no network and no model weights. Libraries that ship no type information (umap, scikit-learn, plotly, sentence-transformers) have minimal stubs under `typings/` declaring only what this project calls.

## Troubleshooting

- **`GITHUB_TOKEN is not set`**: put it in `.env` or export it. `gh auth token` prints one if you use the GitHub CLI.
- **Ollama connection errors**: `ollama serve` must be running and the model pulled (`ollama list`).
- **A gated embedding model** (for example `google/embeddinggemma-300m`) needs a Hugging Face login that has accepted its terms; Qwen3-Embedding is Apache-2.0 and needs none.
- **HDBSCAN `cluster_selection_epsilon`** is not exposed because scikit-learn 1.9.1 raises a NumPy scalar-conversion error on that path whenever it would merge clusters ([scikit-learn#34243](https://github.com/scikit-learn/scikit-learn/pull/34243)).
