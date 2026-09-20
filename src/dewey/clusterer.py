from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import HDBSCAN

if TYPE_CHECKING:
    from logging import Logger

    from numpy.typing import NDArray

NOISE = -1


@dataclass(frozen=True)
class HdbscanSettings:
    min_cluster_size: int
    min_samples: int


@dataclass(frozen=True)
class Clustering:
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]

    def cluster_ids(self) -> list[int]:
        labels: list[int] = np.unique(self.labels).tolist()

        return [label for label in labels if label != NOISE]

    def members(self, cluster_id: int) -> list[int]:
        members: list[int] = np.flatnonzero(self.labels == cluster_id).tolist()

        return members

    def noise_count(self) -> int:
        return int(np.count_nonzero(self.labels == NOISE))


class Clusterer:
    def __init__(self, settings: HdbscanSettings, logger: Logger) -> None:
        self.settings = settings
        self.logger = logger

    def run(self, points: NDArray[np.float32]) -> Clustering:
        self.logger.info("clustering %s with HDBSCAN", points.shape)
        model = HDBSCAN(
            min_cluster_size=self.settings.min_cluster_size,
            min_samples=self.settings.min_samples,
            copy=True,
        )
        model.fit(points)

        clustering = Clustering(
            labels=np.asarray(model.labels_, dtype=np.int64),
            probabilities=np.asarray(model.probabilities_, dtype=np.float64),
        )
        self.logger.info(
            "found %d clusters and %d noise points", len(clustering.cluster_ids()), clustering.noise_count()
        )

        return clustering
