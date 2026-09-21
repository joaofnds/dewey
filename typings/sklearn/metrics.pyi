import numpy as np
from numpy.typing import NDArray

def silhouette_score(
    x: NDArray[np.floating], labels: NDArray[np.integer], /, *, metric: str = "euclidean"
) -> float: ...
