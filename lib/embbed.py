import logging
from hashlib import blake2b
from os import path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from lib.repo_data import RepoData


class Embedder:
    def __init__(
        self,
        logger: logging.Logger,
        model_name: str,
    ):
        self.logger = logger
        self.model_name = model_name

    def run(self, repos: list[RepoData]):
        texts = [repo.summary() for repo in repos]

        texts_hash = blake2b(self.model_name.encode() + "".join(texts).encode()).hexdigest()
        embeddings_cache_path = f"data/embeddings/{texts_hash}.npy"

        if path.exists(embeddings_cache_path):
            self.logger.info(f"Loading cached embeddings from {embeddings_cache_path}")
            return np.load(embeddings_cache_path)

        logging.info("generating embeddings...")

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        normalized = normalize(embeddings)
        np.save(embeddings_cache_path, normalized)

        logging.info(f"embeddings shape: {normalized.shape}")

        return normalized
