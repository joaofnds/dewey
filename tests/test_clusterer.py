import logging

import numpy as np
import pytest

from dewey.clusterer import Clusterer, HdbscanSettings


@pytest.fixture
def three_blobs_and_an_outlier() -> np.ndarray:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    blobs = np.concatenate([center + rng.normal(scale=0.3, size=(20, 2)) for center in centers])

    return np.vstack([blobs, [[50.0, -50.0]]])


def a_clusterer() -> Clusterer:
    return Clusterer(HdbscanSettings(min_cluster_size=5, min_samples=3), logger=logging.getLogger("test"))


class TestClusterer:
    def test_finds_each_blob(self, three_blobs_and_an_outlier: np.ndarray) -> None:
        clustering = a_clusterer().run(three_blobs_and_an_outlier)

        assert clustering.cluster_ids() == [0, 1, 2]

    def test_marks_the_outlier_as_noise(self, three_blobs_and_an_outlier: np.ndarray) -> None:
        clustering = a_clusterer().run(three_blobs_and_an_outlier)

        assert clustering.labels[-1] == -1
        assert clustering.noise_count() == 1

    def test_lists_the_members_of_a_cluster(self, three_blobs_and_an_outlier: np.ndarray) -> None:
        clustering = a_clusterer().run(three_blobs_and_an_outlier)

        members = clustering.members(clustering.labels[0])

        assert sorted(members) == list(range(20))

    def test_scores_membership_between_zero_and_one(self, three_blobs_and_an_outlier: np.ndarray) -> None:
        clustering = a_clusterer().run(three_blobs_and_an_outlier)

        assert clustering.probabilities.min() >= 0.0
        assert clustering.probabilities.max() <= 1.0
