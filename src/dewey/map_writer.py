from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from dewey.cluster_namer import UNCLUSTERED

if TYPE_CHECKING:
    from pathlib import Path

SUMMARY_PREVIEW = 160
PLANE = 2
OPEN_ON_CLICK = """
document.getElementById('{plot_id}').on('plotly_click', function (event) {
  var url = event.points[0].customdata;
  if (url) { window.open(url, '_blank'); }
});
"""


@dataclass(frozen=True)
class MapPoint:
    name: str
    url: str
    cluster: str
    summary: str
    coordinates: tuple[float, ...]


def write_map(points: list[MapPoint], output: Path, title: str) -> None:
    dimensions = {len(point.coordinates) for point in points}
    if dimensions not in ({2}, {3}):
        message = f"every point needs 2 or 3 coordinates, got dimensions {sorted(dimensions)}"
        raise ValueError(message)

    clusters: defaultdict[str, list[MapPoint]] = defaultdict(list)
    for point in points:
        clusters[point.cluster].append(point)

    traces = [cluster_trace(name, members) for name, members in clusters.items() if name != UNCLUSTERED]
    if UNCLUSTERED in clusters:
        traces.insert(0, cluster_trace(UNCLUSTERED, clusters[UNCLUSTERED]))
    if dimensions == {2}:
        traces.append(label_trace(clusters))

    figure = go.Figure(data=traces)
    figure.update_layout(title=title, legend_title_text="Cluster", hovermode="closest", template="plotly_white")
    figure.write_html(str(output), include_plotlyjs="cdn", post_script=OPEN_ON_CLICK)


def cluster_trace(name: str, members: list[MapPoint]) -> go.BaseTraceType:
    unclustered = name == UNCLUSTERED
    hover = [f"<b>{point.name}</b><br>{point.cluster}<br>{preview(point.summary)}" for point in members]
    marker: dict[str, object] = {"size": 4 if unclustered else 7, "opacity": 0.35 if unclustered else 0.85}
    if unclustered:
        marker["color"] = "#9e9e9e"

    return trace(
        members,
        name=name,
        mode="markers",
        text=hover,
        customdata=[point.url for point in members],
        hovertemplate="%{text}<extra></extra>",
        marker=marker,
    )


def label_trace(clusters: dict[str, list[MapPoint]]) -> go.BaseTraceType:
    named = {name: members for name, members in clusters.items() if name != UNCLUSTERED}
    centers = [np.median([point.coordinates for point in members], axis=0) for members in named.values()]

    return go.Scatter(
        x=[center[0] for center in centers],
        y=[center[1] for center in centers],
        mode="text",
        text=list(named),
        textfont={"size": 11, "color": "#222"},
        hoverinfo="skip",
        showlegend=False,
    )


def trace(members: list[MapPoint], **style: object) -> go.BaseTraceType:
    coordinates = np.array([point.coordinates for point in members])
    if coordinates.shape[1] == PLANE:
        return go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], **style)

    return go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2], **style)


def preview(summary: str) -> str:
    if len(summary) <= SUMMARY_PREVIEW:
        return summary

    return summary[:SUMMARY_PREVIEW].rsplit(" ", 1)[0] + "…"
