import logging
from typing import TYPE_CHECKING

import numpy as np

from dewey.embeddings import EmbeddingCache
from tests.fakes import FakeEmbedder

if TYPE_CHECKING:
    from pathlib import Path


def a_cache(root: Path, embedder: FakeEmbedder) -> EmbeddingCache:
    return EmbeddingCache(root, embedder, logging.getLogger("test"))


class TestEmbeddingCache:
    def test_returns_the_embedder_vectors(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder()

        vectors = a_cache(tmp_path, embedder).embed(["a", "b"])

        np.testing.assert_array_equal(vectors, embedder.embed(["a", "b"]))

    def test_reuses_stored_vectors_for_the_same_texts(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder()
        first = a_cache(tmp_path, embedder).embed(["a", "b"])

        second = a_cache(tmp_path, embedder).embed(["a", "b"])

        np.testing.assert_array_equal(second, first)
        assert len(embedder.calls) == 1

    def test_embeds_again_for_a_different_model(self, tmp_path: Path) -> None:
        a_cache(tmp_path, FakeEmbedder(model_name="one")).embed(["a"])
        other = FakeEmbedder(model_name="two")

        a_cache(tmp_path, other).embed(["a"])

        assert len(other.calls) == 1

    def test_embeds_again_when_a_text_changes(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder()
        a_cache(tmp_path, embedder).embed(["a", "b"])

        a_cache(tmp_path, embedder).embed(["a", "c"])

        assert len(embedder.calls) == 2
