import numpy as np
from numpy.typing import NDArray

class HDBSCAN:
    labels_: NDArray[np.int64]
    probabilities_: NDArray[np.float64]

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        max_cluster_size: int | None = None,
        metric: str = "euclidean",
        copy: bool = False,
    ) -> None: ...
    def fit(self, X: NDArray[np.floating]) -> HDBSCAN: ...
