from dataclasses import dataclass
from typing import TYPE_CHECKING

from dewey.cluster_namer import ClusterNamer
from dewey.clusterer import Clusterer, Clustering, HdbscanSettings
from dewey.embeddings import Embedder, EmbeddingCache
from dewey.map_writer import MapPoint, write_map
from dewey.prompts import load_prompt
from dewey.reducer import Reducer, UmapSettings
from dewey.stars import GitHubClient, StarFetcher
from dewey.summaries import SummaryWriter

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from dewey.llm import LLM
    from dewey.repos import RepoStore, StarredRepo

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
    central_repos: int
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


class NoStarsError(ValueError):
    pass


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
    if not repos:
        message = f"{settings.username} has no starred repositories"
        raise NoStarsError(message)

    SummaryWriter(
        store, services.summarizer, settings.summary_workers, logger, overwrite=settings.overwrite_summaries
    ).run(repos)
    summaries = [store.summary(repo.id) for repo in repos]

    vectors = EmbeddingCache(store.embeddings_dir, services.embedder, logger).embed(summaries)
    reducer = Reducer(settings.umap, settings.seed, logger)
    cluster_space = reducer.reduce(vectors, CLUSTER_SPACE_DIMENSIONS, CLUSTER_SPACE_MIN_DIST)
    map_space = reducer.reduce(vectors, settings.plot_dimensions, MAP_MIN_DIST)

    clustering = Clusterer(settings.hdbscan, logger).run(cluster_space)
    namer = ClusterNamer(store.names_dir, services.namer, settings.central_repos, load_prompt("cluster_name"), logger)
    names = namer.run(clustering, repos, summaries)

    write_map(map_points(repos, summaries, clustering, names, map_space), settings.output, settings.title)
    logger.info("wrote %s", settings.output)

    return MapResult(
        output=settings.output,
        repos=len(repos),
        clusters=len(clustering.cluster_ids()),
        unclustered=clustering.noise_count(),
    )


def map_points(
    repos: list[StarredRepo],
    summaries: list[str],
    clustering: Clustering,
    names: dict[int, str],
    map_space: NDArray[np.float32],
) -> list[MapPoint]:
    labels: list[int] = clustering.labels.tolist()

    return [
        MapPoint(
            name=repo.full_name,
            url=repo.url,
            cluster_id=label,
            cluster=names[label],
            summary=summary,
            coordinates=tuple(float(value) for value in coordinates),
        )
        for repo, summary, label, coordinates in zip(repos, summaries, labels, map_space, strict=True)
    ]
