from logging import Logger

import hdbscan
import numpy as np


class Clusterer:
    def __init__(
        self,
        logger: Logger,
        min_cluster_size: int,
        min_samples: int,
        epsilon: float,
        max_cluster_size: int,
        metric: str,
    ):
        self.logger = logger

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            max_cluster_size=max_cluster_size,
            metric=metric,
        )

    def run(self, embeddings: np.ndarray) -> np.ndarray:
        self.logger.info("clustering with HDBSCAN...")
        labels = self.clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(np.array(labels) == -1)
        self.logger.info(f"Found {n_clusters} clusters and {n_noise} noise points.")

        return labels
