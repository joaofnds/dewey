import argparse
import logging
import sys
from pathlib import Path

from sklearn.metrics import silhouette_score

from dewey.clusterer import NOISE, Clusterer, HdbscanSettings
from dewey.embeddings import EmbeddingCache, SentenceTransformerEmbedder
from dewey.pipeline import CLUSTER_SPACE_DIMENSIONS, CLUSTER_SPACE_MIN_DIST
from dewey.reducer import Reducer, UmapSettings
from dewey.repos import RepoStore


def main() -> None:
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("measure")
    store = RepoStore(args.data_dir)
    summaries = [store.summary(repo_id) for repo_id in stored_ids(store)]
    print(f"{len(summaries)} summaries from {args.data_dir}")
    print("| Embedding model | Clusters | Unclustered | Silhouette (5-D) |")
    print("| --- | --- | --- | --- |")

    for model in args.models:
        embedder = SentenceTransformerEmbedder(model, batch_size=32, logger=logger)
        vectors = EmbeddingCache(store.embeddings_dir, embedder, logger).embed(summaries)
        umap = UmapSettings(n_neighbors=args.neighbors, metric="cosine", n_epochs=200)
        points = Reducer(umap, args.seed, logger).reduce(vectors, CLUSTER_SPACE_DIMENSIONS, CLUSTER_SPACE_MIN_DIST)
        hdbscan = HdbscanSettings(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
        clustering = Clusterer(hdbscan, logger).run(points)

        clustered = clustering.labels != NOISE
        silhouette = silhouette_score(points[clustered], clustering.labels[clustered])
        unclustered = 100 * clustering.noise_count() / len(summaries)
        print(f"| `{model}` | {len(clustering.cluster_ids())} | {unclustered:.1f}% | {silhouette:.3f} |")


def stored_ids(store: RepoStore) -> list[int]:
    return sorted(int(folder.name) for folder in store.repos_dir.iterdir() if store.has_summary(int(folder.name)))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare embedding models on the stored summaries.")
    parser.add_argument("models", nargs="+", help="sentence-transformers model names")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)
