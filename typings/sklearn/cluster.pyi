import numpy as np
from numpy.typing import NDArray

class HDBSCAN:
    labels_: NDArray[np.int64]
    probabilities_: NDArray[np.float64]

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int | None = None,
        metric: str = "euclidean",
    ) -> None: ...
    def fit(self, X: NDArray[np.floating]) -> HDBSCAN: ...
