import json
import logging
import pickle
from hashlib import blake2b
from os import path

import numpy as np

from lib.ollama import Ollama


class LabelNamer:
    def __init__(
        self,
        logger: logging.Logger,
        llm: Ollama,
        samples_per_cluster: int,
    ):
        self.logger = logger
        self.samples_per_cluster = samples_per_cluster
        self.llm = llm
        self.prompt_template = """You are analyzing clusters of GitHub repository summaries. Based on the following sample texts from a cluster, provide a single concise category name (2-4 words) that best describes the common theme or domain of these repositories.

    Sample texts from the cluster:
    <samples>
    {samples_text}
    </samples>

    Respond with *ONLY* a *SINGLE* category name (e.g., "Web Development", "Data Science", "Mobile Apps", "DevOps Tools", etc.). Do not include explanations or additional text."""

    def run(self, labels: np.ndarray, texts: list[str]) -> dict[int, str]:
        assert len(labels) == len(texts), "labels and texts must have the same length"
        logging.info(f"generating label names using {self.llm.model} for {len(set(labels))} clusters...")

        cache_key = blake2b(
            self.llm.model.encode()
            + self.prompt_template.encode()
            + str(self.samples_per_cluster).encode()
            + str(labels).encode()
            + "".join(texts).encode()
        )
        cache_file_path = f"data/labels/{cache_key.hexdigest()}.pkl"
        if path.exists(cache_file_path):
            logging.info(f"Loading cached labels from {cache_file_path}")
            with open(cache_file_path, "rb") as cache_file:
                return pickle.load(cache_file)

        logging.info("Generating labels for clusters...")

        cluster_texts = {}
        for i, label in enumerate(labels):
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append(texts[i])

        label_mapping = {}

        for cluster_id, cluster_data in cluster_texts.items():
            if cluster_id == -1:
                label_mapping[cluster_id] = "Noise/Outliers"
                continue

            n_samples = min(self.samples_per_cluster, len(cluster_data))
            samples = np.random.choice(cluster_data, n_samples, replace=False)
            prompt = self.prompt_template.format(
                samples_text="\n".join([f"<sample>\n{text}\n</sample>" for text in samples])
            )
            label = self.llm.generate(prompt)
            label_mapping[cluster_id] = label
            logging.info(f"Cluster {cluster_id}: {label}")

        with open(cache_file_path, "wb") as cache_file:
            pickle.dump(label_mapping, cache_file)

        return label_mapping
