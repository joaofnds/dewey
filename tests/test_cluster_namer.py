import logging
from typing import TYPE_CHECKING

import numpy as np

from dewey.cluster_namer import UNCLUSTERED, ClusterNamer
from dewey.clusterer import Clustering
from tests.builders import a_repo
from tests.fakes import FakeLLM

if TYPE_CHECKING:
    from pathlib import Path


def a_namer(root: Path, llm: FakeLLM, central_repos: int = 2) -> ClusterNamer:
    return ClusterNamer(directory=root, llm=llm, central_repos=central_repos, logger=logging.getLogger("test"))


def two_clusters_and_noise() -> Clustering:
    return Clustering(
        labels=np.array([0, 0, 0, 1, 1, -1]),
        probabilities=np.array([0.9, 0.2, 0.7, 1.0, 0.5, 0.0]),
    )


REPOS = [a_repo(i, full_name=f"org/repo{i}") for i in range(6)]
SUMMARIES = [
    "Kubernetes operator for postgres clusters.",
    "Kubernetes controller for backups.",
    "Kubernetes dashboard.",
    "Emacs configuration in org mode.",
    "Emacs package manager.",
    "A rock collection catalog.",
]


class TestClusterNamer:
    def test_names_each_cluster_with_the_llm_answer(self, tmp_path: Path) -> None:
        llm = FakeLLM()
        llm.queue("Kubernetes Operators", "Emacs Tooling")

        names = a_namer(tmp_path, llm).run(two_clusters_and_noise(), REPOS, SUMMARIES)

        assert names == {0: "Kubernetes Operators", 1: "Emacs Tooling", -1: UNCLUSTERED}

    def test_prompts_with_the_most_central_repos(self, tmp_path: Path) -> None:
        llm = FakeLLM()

        a_namer(tmp_path, llm, central_repos=2).run(two_clusters_and_noise(), REPOS, SUMMARIES)

        assert "org/repo0" in llm.prompts[0]
        assert "org/repo2" in llm.prompts[0]
        assert "org/repo1" not in llm.prompts[0]

    def test_prompts_with_the_distinctive_words(self, tmp_path: Path) -> None:
        llm = FakeLLM()

        a_namer(tmp_path, llm).run(two_clusters_and_noise(), REPOS, SUMMARIES)

        assert "kubernetes" in llm.prompts[0].lower()
        assert "emacs" in llm.prompts[1].lower()

    def test_reuses_stored_names_for_the_same_clustering(self, tmp_path: Path) -> None:
        first_llm = FakeLLM()
        first_llm.queue("Kubernetes Operators", "Emacs Tooling")
        first = a_namer(tmp_path, first_llm).run(two_clusters_and_noise(), REPOS, SUMMARIES)
        second_llm = FakeLLM()

        second = a_namer(tmp_path, second_llm).run(two_clusters_and_noise(), REPOS, SUMMARIES)

        assert second == first
        assert second_llm.prompts == []

    def test_stores_names_as_json(self, tmp_path: Path) -> None:
        a_namer(tmp_path, FakeLLM()).run(two_clusters_and_noise(), REPOS, SUMMARIES)

        assert [path.suffix for path in tmp_path.iterdir()] == [".json"]
