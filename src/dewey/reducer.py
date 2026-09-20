from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from umap import UMAP

if TYPE_CHECKING:
    from logging import Logger

    from numpy.typing import NDArray


@dataclass(frozen=True)
class UmapSettings:
    n_neighbors: int
    metric: str
    n_epochs: int


class Reducer:
    def __init__(self, settings: UmapSettings, seed: int, logger: Logger) -> None:
        self.settings = settings
        self.seed = seed
        self.logger = logger

    def reduce(self, vectors: NDArray[np.float32], components: int, min_dist: float) -> NDArray[np.float32]:
        self.logger.info("reducing %s to %d dimensions with UMAP", vectors.shape, components)
        umap = UMAP(
            n_components=components,
            n_neighbors=self.settings.n_neighbors,
            min_dist=min_dist,
            metric=self.settings.metric,
            n_epochs=self.settings.n_epochs,
            random_state=self.seed,
        )

        return np.asarray(umap.fit_transform(vectors), dtype=np.float32)
