from typing import TYPE_CHECKING, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from dewey.fingerprint import fingerprint

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from numpy.typing import NDArray


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...
    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int, logger: Logger) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger = logger

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.logger.info("loading embedding model %s", self.model_name)
        model = SentenceTransformer(self.model_name)
        self.logger.info("embedding %d texts on %s", len(texts), model.device)

        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.asarray(vectors, dtype=np.float32)


class EmbeddingCache:
    def __init__(self, directory: Path, embedder: Embedder, logger: Logger) -> None:
        self.directory = directory
        self.embedder = embedder
        self.logger = logger

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        path = self.directory / f"{self.key(texts)}.npy"
        if path.exists():
            self.logger.info("loading embeddings from %s", path)
            return np.load(path)

        vectors = self.embedder.embed(texts)

        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)

        return vectors

    def key(self, texts: list[str]) -> str:
        return fingerprint([self.embedder.model_name.encode()], texts)
