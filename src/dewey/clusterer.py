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
    epsilon: float


@dataclass(frozen=True)
class Clustering:
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]

    def cluster_ids(self) -> list[int]:
        return [int(label) for label in np.unique(self.labels) if label != NOISE]

    def members(self, cluster_id: int) -> NDArray[np.intp]:
        return np.flatnonzero(self.labels == cluster_id)

    def noise_count(self) -> int:
        return int(np.sum(self.labels == NOISE))


class Clusterer:
    def __init__(self, settings: HdbscanSettings, logger: Logger) -> None:
        self.settings = settings
        self.logger = logger

    def run(self, points: NDArray[np.float32]) -> Clustering:
        self.logger.info("clustering %s with HDBSCAN", points.shape)
        model = HDBSCAN(
            min_cluster_size=self.settings.min_cluster_size,
            min_samples=self.settings.min_samples,
            cluster_selection_epsilon=self.settings.epsilon,
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
