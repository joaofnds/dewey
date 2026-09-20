import logging

import numpy as np
import pytest

from dewey.reducer import Reducer, UmapSettings


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((60, 16)).astype(np.float32)


def a_reducer(seed: int = 1) -> Reducer:
    return Reducer(
        UmapSettings(n_neighbors=10, metric="cosine", n_epochs=50), seed=seed, logger=logging.getLogger("test")
    )


class TestReducer:
    @pytest.mark.parametrize("components", [2, 5])
    def test_reduces_to_the_requested_dimensions(self, vectors: np.ndarray, components: int) -> None:
        points = a_reducer().reduce(vectors, components=components, min_dist=0.0)

        assert points.shape == (60, components)

    def test_is_deterministic_for_a_seed(self, vectors: np.ndarray) -> None:
        first = a_reducer(seed=7).reduce(vectors, components=2, min_dist=0.1)

        second = a_reducer(seed=7).reduce(vectors, components=2, min_dist=0.1)

        np.testing.assert_array_equal(second, first)
