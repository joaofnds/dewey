import numpy as np
from numpy.typing import NDArray

class SentenceTransformer:
    device: object

    def __init__(self, model_name_or_path: str) -> None: ...
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool | None = None,
    ) -> NDArray[np.float32]: ...
