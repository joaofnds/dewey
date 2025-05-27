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

USERNAME = "joaofnds"

SUMMARY_LLM_MODEL = "mistral"
SUMMARY_LLM_BASE_URL = "http://localhost:11434"
SUMMARY_LLM_TIMEOUT = 60

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RANDOM_STATE = 42

UMAP_COMPONENTS = 3
UMAP_N_NEIGHBORS = 9
UMAP_METRIC = "cosine"
UMAP_N_EPOCHS = 200
UMAP_LEARNING_RATE = 1.0
UMAP_INIT = "spectral"
UMAP_MIN_DIST = 0.04
UMAP_SPREAD = 1.0

HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_EPSILON = 0.25
HDBSCAN_MAX_CLUSTER_SIZE = 0
HDBSCAN_METRIC = "euclidean"

LABLER_LLM_MODEL = "mistral"
LABLER_LLM_BASE_URL = "http://localhost:11434"
LABLER_LLM_TIMEOUT = 20

PLOT_OUTPUT_FILE = "cluster_visualization.html"

np.random.seed(RANDOM_STATE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

repo_fetcher = RepoFetcher(
    logger=logger,
    token=must_get_env("GITHUB_TOKEN"),
    workers=10,
)
summary_generator = GenerateSummaries(
    logger=logger,
    llm=Ollama(
        model=SUMMARY_LLM_MODEL,
        base_url=SUMMARY_LLM_BASE_URL,
        timeout=SUMMARY_LLM_TIMEOUT,
    ),
)
embedder = Embedder(
    logger=logger,
    model_name=EMBEDDING_MODEL_NAME,
)
reducer = Reducer(
    logger=logger,
    random_state=RANDOM_STATE,
    umap_components=UMAP_COMPONENTS,
    umap_n_neighbors=UMAP_N_NEIGHBORS,
    umap_metric=UMAP_METRIC,
    umap_n_epochs=UMAP_N_EPOCHS,
    umap_min_dist=UMAP_MIN_DIST,
    umap_spread=UMAP_SPREAD,
    umap_learning_rate=UMAP_LEARNING_RATE,
)
clusterer = Clusterer(
    logger=logger,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    epsilon=HDBSCAN_EPSILON,
    max_cluster_size=HDBSCAN_MAX_CLUSTER_SIZE,
    metric=HDBSCAN_METRIC,
)
label_namer = LabelNamer(
    logger=logger,
    samples_per_cluster=10,
    llm=Ollama(
        model=LABLER_LLM_MODEL,
        base_url=LABLER_LLM_BASE_URL,
        timeout=LABLER_LLM_TIMEOUT,
    ),
)

repos = repo_fetcher.run(USERNAME)
texts = [repo.summary() for repo in repos]
summary_generator.run(repos)
embeddings = embedder.run(repos)
reduced_embeddings = reducer.run(embeddings)
labels = clusterer.run(reduced_embeddings)
label_to_name = label_namer.run(labels, texts)

plot(
    dimensions=UMAP_COMPONENTS,
    embeddings=reduced_embeddings,
    labels=[label_to_name.get(label, f"Cluster {label}") for label in labels],
    texts=texts,
    ids=[repo.full_name() for repo in repos],
    label_cutoff=50,
    text_cutoff=50,
    output_file=PLOT_OUTPUT_FILE,
)
