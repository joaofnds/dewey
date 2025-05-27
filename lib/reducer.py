from logging import Logger

import numpy as np
from umap import UMAP


class Reducer:
    def __init__(
        self,
        logger: Logger,
        random_state: int,
        umap_components: int,
        umap_n_neighbors: int,
        umap_metric: str,
        umap_n_epochs: int,
        umap_min_dist: float,
        umap_spread: float,
        umap_learning_rate: float,
    ):
        self.logger = logger

        self.random_state = random_state

        self.reducer = UMAP(
            random_state=self.random_state,
            n_components=umap_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            spread=umap_spread,
            metric=umap_metric,
            n_epochs=umap_n_epochs,
            learning_rate=umap_learning_rate,
        )

    def run(self, embeddings: np.ndarray) -> np.ndarray:
        self.logger.info("reducing dimensions with UMAP...")

        reduced_embeddings = self.reducer.fit_transform(embeddings)
        assert isinstance(reduced_embeddings, np.ndarray)

        return reduced_embeddings
