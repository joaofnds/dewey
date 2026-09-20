import logging

import numpy as np
import pandas as pd
import plotly.express as px

OUTPUT_HTML_FILE = "cluster_visualization.html"


def plot(
    embeddings: np.ndarray,
    labels: list[str],
    texts: list[str],
    ids: list[str],
    label_cutoff: int,
    text_cutoff: int,
    output_file: str,
):
    labels = [label[:label_cutoff] if len(label) > label_cutoff else label for label in labels]
    texts = [text[:text_cutoff] if len(text) > text_cutoff else text for text in texts]

    dimensions = embeddings.shape[1]

    if dimensions == 2:
        fig = visualize_clusters_2d(embeddings, labels, texts, ids)
    elif dimensions == 3:
        fig = visualize_clusters_3d(embeddings, labels, texts, ids)
    else:
        raise ValueError("Dimensions must be either 2 or 3.")

    fig.write_html(output_file)


def visualize_clusters_2d(
    embeddings: np.ndarray,
    labels: list[str],
    texts: list[str],
    ids: list[str],
):
    logging.info(f"Generating visualization... saving to {OUTPUT_HTML_FILE}")
    df = pd.DataFrame(embeddings, columns=["x", "y"])

    df["cluster"] = labels
    df["text"] = texts
    df["id"] = ids

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data={"x": False, "y": False, "cluster": True, "id": True, "text": True},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title_text="Cluster",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig


def visualize_clusters_3d(
    embeddings: np.ndarray,
    labels: list[str],
    texts: list[str],
    ids: list[str],
):
    logging.info(f"Generating 3D visualization... saving to {OUTPUT_HTML_FILE}")
    df = pd.DataFrame(embeddings, columns=["x", "y", "z"])

    df["cluster"] = labels
    df["text"] = texts
    df["id"] = ids

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_data={
            "x": False,
            "y": False,
            "z": False,
            "cluster": True,
            "id": True,
            "text": False,
        },
        title=f"3D Cluster Visualization of {len(texts)} GitHub Repo Summaries (HDBSCAN clustering)",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
        ),
        legend_title_text="Cluster",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=20),
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    return fig
