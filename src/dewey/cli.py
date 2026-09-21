import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dewey.claude import Claude
from dewey.clusterer import HdbscanSettings
from dewey.embeddings import SentenceTransformerEmbedder
from dewey.ollama import Ollama
from dewey.pipeline import PipelineSettings, Services, run
from dewey.reducer import UmapSettings
from dewey.repos import RepoStore
from dewey.stars import GitHubRestClient

if TYPE_CHECKING:
    from dewey.llm import LLM

CLAUDE_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_TOKENS = 400
GITHUB_TIMEOUT_SECONDS = 30


def main() -> None:
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("dewey")

    settings = PipelineSettings(
        username=args.username,
        output=args.output,
        title=f"{args.username}'s starred repositories",
        plot_dimensions=args.dimensions,
        refresh_stars=args.refresh_stars,
        overwrite_summaries=args.overwrite_summaries,
        summary_workers=args.summary_workers,
        central_repos=args.central_repos,
        umap=UmapSettings(n_neighbors=args.neighbors, metric="cosine", n_epochs=200),
        hdbscan=HdbscanSettings(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples),
        seed=args.seed,
    )
    llm = build_llm(args.llm, args.llm_model, args.ollama_url)
    services = Services(
        store=RepoStore(args.data_dir),
        github=GitHubRestClient(github_token(), GITHUB_TIMEOUT_SECONDS),
        summarizer=llm,
        namer=llm,
        embedder=SentenceTransformerEmbedder(args.embedding_model, batch_size=32, logger=logger),
        logger=logger,
    )

    result = run(settings, services)
    logger.info(
        "%d repos in %d clusters, %d unclustered: %s",
        result.repos,
        result.clusters,
        result.unclustered,
        result.output,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dewey", description="Cluster and map a GitHub user's starred repositories.")
    parser.add_argument("username", help="GitHub user whose stars to map")
    parser.add_argument("--output", type=Path, default=Path("map.html"), help="HTML file to write")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="where repos, summaries and caches live")
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=2, help="map dimensions (default: 2)")
    parser.add_argument("--llm", choices=("claude", "ollama"), default="claude", help="who writes summaries and names")
    parser.add_argument("--llm-model", default=None, help=f"model name (default: {CLAUDE_MODEL} or {OLLAMA_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"Ollama server (default: {OLLAMA_URL})")
    parser.add_argument(
        "--embedding-model", default=EMBEDDING_MODEL, help=f"sentence-transformers model (default: {EMBEDDING_MODEL})"
    )
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min_cluster_size (default: 15)")
    parser.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples (default: 5)")
    parser.add_argument("--central-repos", type=int, default=8, help="repos shown to the LLM per cluster (default: 8)")
    parser.add_argument("--summary-workers", type=int, default=4, help="parallel summary requests (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random state (default: 42)")
    parser.add_argument(
        "--refresh-stars", action="store_true", help="list the stars again instead of using the cached list"
    )
    parser.add_argument("--overwrite-summaries", action="store_true", help="regenerate every summary")

    return parser.parse_args(argv)


def build_llm(provider: str, model: str | None, ollama_url: str) -> LLM:
    if provider == "ollama":
        return Ollama(model or OLLAMA_MODEL, ollama_url, LLM_TIMEOUT_SECONDS)

    return Claude(model or CLAUDE_MODEL, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS)


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    message = "GITHUB_TOKEN is not set (try: export GITHUB_TOKEN=$(gh auth token))"
    raise SystemExit(message)
