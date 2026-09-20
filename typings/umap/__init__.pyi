import numpy as np
from numpy.typing import NDArray

class UMAP:
    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "euclidean",
        n_epochs: int | None = None,
        min_dist: float = 0.1,
        random_state: int | None = None,
    ) -> None: ...
    def fit_transform(self, x: NDArray[np.floating], /) -> NDArray[np.float32]: ...
