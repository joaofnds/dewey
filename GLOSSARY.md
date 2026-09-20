# Glossary

Terms of the work this tool does: keeping a large collection of starred GitHub repositories findable. Terms of the implementation (embeddings, UMAP, HDBSCAN, caches) stay in the code.

- **Star**: a GitHub user's bookmark on a repository. Dewey maps one user's stars.
- **Starred repository**: a repository the user starred, with the facts GitHub publishes about it (name, description, language, topics, license, size, creation year) and its README when it has one.
- **Summary**: the one-paragraph technical description of a starred repository that an LLM writes from those facts and the README. Every later stage works from summaries, never from READMEs directly. The prompt asks the model for an "abstract"; the stored text is the summary.
- **Map**: the HTML page Dewey produces. Each starred repository is a point; repositories with similar summaries sit close together.
- **Cluster**: a group of starred repositories the map draws in one color and names as one category.
- **Cluster name**: the short category an LLM gives a cluster.
- **Central repository**: one of the repositories that belong most strongly to a cluster. The LLM sees the central repositories when it names the cluster.
- **Distinctive words**: the words that appear in one cluster's summaries far more than in the others'. They accompany the central repositories in the naming prompt.
- **Unclustered**: a starred repository that belongs to no cluster. The map draws these in grey under the legend entry "Unclustered" and writes no name for them on the map.
