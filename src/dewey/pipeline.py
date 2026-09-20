from dataclasses import dataclass
from typing import TYPE_CHECKING

from dewey.cluster_namer import ClusterNamer
from dewey.clusterer import Clusterer, HdbscanSettings
from dewey.embeddings import Embedder, EmbeddingCache
from dewey.map_writer import MapPoint, write_map
from dewey.reducer import Reducer, UmapSettings
from dewey.stars import GitHubClient, StarFetcher
from dewey.summaries import SummaryWriter

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from dewey.llm import LLM
    from dewey.repos import RepoStore

CLUSTER_SPACE_DIMENSIONS = 5
CLUSTER_SPACE_MIN_DIST = 0.0
MAP_MIN_DIST = 0.1
FETCH_WORKERS = 10


@dataclass(frozen=True)
class PipelineSettings:
    username: str
    output: Path
    title: str
    plot_dimensions: int
    refresh_stars: bool
    overwrite_summaries: bool
    summary_workers: int
    representatives: int
    umap: UmapSettings
    hdbscan: HdbscanSettings
    seed: int


@dataclass(frozen=True)
class Services:
    store: RepoStore
    github: GitHubClient
    summarizer: LLM
    namer: LLM
    embedder: Embedder
    logger: Logger


@dataclass(frozen=True)
class MapResult:
    output: Path
    repos: int
    clusters: int
    unclustered: int


def run(settings: PipelineSettings, services: Services) -> MapResult:
    store, logger = services.store, services.logger

    repos = StarFetcher(store, services.github, FETCH_WORKERS, logger).run(
        settings.username, refresh=settings.refresh_stars
    )
    SummaryWriter(
        store, services.summarizer, settings.summary_workers, logger, overwrite=settings.overwrite_summaries
    ).run(repos)
    summaries = [store.summary(repo.id) for repo in repos]

    vectors = EmbeddingCache(store.embeddings_dir, services.embedder, logger).embed(summaries)
    reducer = Reducer(settings.umap, settings.seed, logger)
    cluster_space = reducer.reduce(vectors, CLUSTER_SPACE_DIMENSIONS, CLUSTER_SPACE_MIN_DIST)
    map_space = reducer.reduce(vectors, settings.plot_dimensions, MAP_MIN_DIST)

    clustering = Clusterer(settings.hdbscan, logger).run(cluster_space)
    names = ClusterNamer(store.labels_dir, services.namer, settings.representatives, logger).run(
        clustering, repos, summaries
    )

    points = [
        MapPoint(
            name=repo.full_name,
            url=repo.url,
            cluster=names[int(label)],
            summary=summary,
            coordinates=tuple(float(value) for value in coordinates),
        )
        for repo, summary, label, coordinates in zip(repos, summaries, clustering.labels, map_space, strict=True)
    ]
    write_map(points, settings.output, settings.title)
    logger.info("wrote %s", settings.output)

    return MapResult(
        output=settings.output,
        repos=len(repos),
        clusters=len(clustering.cluster_ids()),
        unclustered=clustering.noise_count(),
    )
