import logging

import numpy as np

from lib.clusterer import Clusterer
from lib.embbed import Embedder
from lib.env import must_get_env
from lib.generate_summaries import GenerateSummaries
from lib.labe_namer import LabelNamer
from lib.ollama import Ollama
from lib.reducer import Reducer
from lib.repo_fetcher import RepoFetcher
from lib.viz import plot

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

repo_fetcher = RepoFetcher(
    logger=logger,
    token=must_get_env("GITHUB_TOKEN"),
    workers=10,
)
summary_generator = GenerateSummaries(
    logger=logger,
    overwrite=False,
    workers=1,
    llm=Ollama(
        model="mistral",
        base_url="http://localhost:11434",
        timeout=60,
    ),
)
embedder = Embedder(
    logger=logger,
    model_name="all-MiniLM-L6-v2",
)
reducer = Reducer(
    logger=logger,
    random_state=np.random.randint(100),
    umap_components=3,
    umap_n_neighbors=9,
    umap_metric="cosine",
    umap_n_epochs=200,
    umap_min_dist=0.04,
    umap_spread=1.0,
    umap_learning_rate=1.0,
)
clusterer = Clusterer(
    logger=logger,
    min_cluster_size=15,
    min_samples=2,
    epsilon=0.25,
    max_cluster_size=0,
    metric="euclidean",
)
label_namer = LabelNamer(
    logger=logger,
    samples_per_cluster=10,
    llm=Ollama(
        model="mistral",
        base_url="http://localhost:11434",
        timeout=20,
    ),
)

repos = repo_fetcher.run("joaofnds")
summary_generator.run(repos)
embeddings = embedder.run(repos)
reduced_embeddings = reducer.run(embeddings)
labels = clusterer.run(reduced_embeddings)
summaries = [repo.summary() for repo in repos]
label_to_name = label_namer.run(labels, summaries)

plot(
    embeddings=reduced_embeddings,
    labels=[label_to_name.get(label, f"Cluster {label}") for label in labels],
    texts=summaries,
    ids=[repo.full_name() for repo in repos],
    label_cutoff=50,
    text_cutoff=50,
    output_file="cluster_visualization.html",
)
