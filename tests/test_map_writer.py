from typing import TYPE_CHECKING

import pytest

from dewey.clusterer import NOISE
from dewey.map_writer import MapPoint, write_map
from tests.maps import embedded_text, text_traces, trace_names

if TYPE_CHECKING:
    from pathlib import Path


CLUSTER_IDS = {"Rust Tools": 0, "Unclustered": NOISE}


def a_point(name: str, cluster: str, *coordinates: float, summary: str | None = None) -> MapPoint:
    return MapPoint(
        name=name,
        url=f"https://github.com/{name}",
        cluster_id=CLUSTER_IDS.get(cluster, 1),
        cluster=cluster,
        summary=summary or f"About {name}.",
        coordinates=coordinates,
    )


class TestWriteMap:
    def test_writes_an_html_map_naming_every_cluster_and_repo(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        points = [
            a_point("a/one", "Rust Tools", 0.0, 0.0),
            a_point("b/two", "Rust Tools", 0.1, 0.1),
            a_point("c/three", "Unclustered", 5.0, 5.0),
        ]

        write_map(points, output=output, title="stars")

        html = embedded_text(output)
        assert "Rust Tools" in html
        assert "Unclustered" in html
        assert "a/one" in html
        assert "c/three" in html
        assert "https://github.com/a/one" in html

    def test_draws_each_cluster_name_on_the_map(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        points = [a_point("a/one", "Rust Tools", 0.0, 0.0), a_point("b/two", "Rust Tools", 0.1, 0.1)]

        write_map(points, output=output, title="stars")

        assert text_traces(embedded_text(output)) == ['"Rust Tools"']

    def test_draws_the_unclustered_points_first_and_grey(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        points = [a_point("a/one", "Rust Tools", 0.0, 0.0), a_point("c/three", "Unclustered", 5.0, 5.0)]

        write_map(points, output=output, title="stars")

        html = embedded_text(output)
        assert trace_names(html)[0] == "Unclustered"
        assert "#9e9e9e" in html

    def test_previews_a_long_summary_in_the_hover(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        summary = "word " * 100
        points = [a_point("a/one", "Rust Tools", 0.0, 0.0, summary=summary), a_point("b/two", "Rust Tools", 0.1, 0.1)]

        write_map(points, output=output, title="stars")

        html = embedded_text(output)
        assert "word word word…" in html
        assert summary.strip() not in html

    def test_tells_clusters_apart_by_id_not_name(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        real = MapPoint(
            name="a/one",
            url="https://github.com/a/one",
            cluster_id=0,
            cluster="Unclustered",
            summary="s",
            coordinates=(0.0, 0.0),
        )
        noise = MapPoint(
            name="b/two",
            url="https://github.com/b/two",
            cluster_id=NOISE,
            cluster="Unclustered",
            summary="s",
            coordinates=(1.0, 1.0),
        )

        write_map([real, noise], output=output, title="stars")

        html = embedded_text(output)
        assert len(trace_names(html)) == 2
        assert text_traces(html) == ['"Unclustered"']

    def test_writes_a_three_dimensional_map(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        points = [a_point("a/one", "Rust Tools", 0.0, 0.0, 0.0), a_point("b/two", "Rust Tools", 0.1, 0.1, 0.1)]

        write_map(points, output=output, title="stars")

        html = embedded_text(output)
        assert "scatter3d" in html
        assert text_traces(html) == []

    class TestWhenPointsMixDimensions:
        def test_rejects_them(self, tmp_path: Path) -> None:
            points = [a_point("a/one", "Rust Tools", 0.0, 0.0), a_point("b/two", "Rust Tools", 0.1, 0.1, 0.1)]

            with pytest.raises(ValueError, match="dimensions"):
                write_map(points, output=tmp_path / "map.html", title="stars")

    class TestWhenTextContainsHtml:
        def test_escapes_it_in_the_hover(self, tmp_path: Path) -> None:
            output = tmp_path / "map.html"
            points = [
                a_point("a/one", "Rust <script>alert(1)</script>", 0.0, 0.0),
                a_point("b/two", "Rust <T>", 0.1, 0.1),
            ]

            write_map(points, output=output, title="stars")

            html = embedded_text(output)
            assert "Rust &lt;script&gt;alert(1)&lt;/script&gt;" in html
            assert "Rust <script>alert(1)</script>" not in html
