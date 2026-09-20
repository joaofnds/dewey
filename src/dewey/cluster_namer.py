import json
from hashlib import blake2b
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from dewey.clusterer import NOISE, Clustering
from dewey.prompts import load_prompt

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from numpy.typing import NDArray

    from dewey.llm import LLM
    from dewey.repos import StarredRepo

UNCLUSTERED = "Unclustered"
DISTINCTIVE_WORDS_PER_CLUSTER = 10


class ClusterNamer:
    def __init__(self, directory: Path, llm: LLM, central_repos: int, logger: Logger) -> None:
        self.directory = directory
        self.llm = llm
        self.central_repos = central_repos
        self.logger = logger
        self.template = load_prompt("cluster_name")

    def run(self, clustering: Clustering, repos: list[StarredRepo], summaries: list[str]) -> dict[int, str]:
        path = self.directory / f"{self.key(clustering, summaries)}.json"
        if path.exists():
            self.logger.info("loading cluster names from %s", path)
            return {int(cluster_id): name for cluster_id, name in json.loads(path.read_text(encoding="utf-8")).items()}

        cluster_ids = clustering.cluster_ids()
        words = distinctive_words(clustering, cluster_ids, summaries)
        self.logger.info("naming %d clusters with %s", len(cluster_ids), self.llm.model)

        names = {NOISE: UNCLUSTERED}
        for cluster_id in cluster_ids:
            prompt = self.prompt(clustering, cluster_id, words[cluster_id], repos, summaries)
            names[cluster_id] = self.llm.generate(prompt).strip().strip('"')
            self.logger.info("cluster %d: %s", cluster_id, names[cluster_id])

        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(names, indent=2), encoding="utf-8")

        return names

    def prompt(
        self,
        clustering: Clustering,
        cluster_id: int,
        words: list[str],
        repos: list[StarredRepo],
        summaries: list[str],
    ) -> str:
        members = clustering.members(cluster_id)
        order: list[int] = np.argsort(-clustering.probabilities[members], kind="stable").tolist()
        central = [members[position] for position in order[: self.central_repos]]
        lines = [f"- {repos[index].full_name}: {summaries[index]}" for index in central]

        return self.template.format(distinctive_words=", ".join(words), central_repos="\n".join(lines))

    def key(self, clustering: Clustering, summaries: list[str]) -> str:
        digest = blake2b(self.llm.model.encode())
        digest.update(self.template.encode())
        digest.update(str(self.central_repos).encode())
        digest.update(clustering.labels.tobytes())
        for text in summaries:
            digest.update(text.encode())
            digest.update(b"\x00")

        return digest.hexdigest()


def distinctive_words(clustering: Clustering, cluster_ids: list[int], summaries: list[str]) -> dict[int, list[str]]:
    documents = [" ".join(summaries[index] for index in clustering.members(cluster_id)) for cluster_id in cluster_ids]
    if not documents:
        return {}

    vectorizer = TfidfVectorizer(stop_words="english")
    weights = vectorizer.fit_transform(documents).toarray()
    vocabulary: list[str] = vectorizer.get_feature_names_out().tolist()

    return {cluster_id: top_words(weights[row], vocabulary) for row, cluster_id in enumerate(cluster_ids)}


def top_words(weights: NDArray[np.float64], vocabulary: list[str]) -> list[str]:
    columns: list[int] = np.argsort(-weights, kind="stable").tolist()

    return [vocabulary[column] for column in columns[:DISTINCTIVE_WORDS_PER_CLUSTER]]
