import re
from pathlib import Path

import pytest

from dewey.map_writer import MapPoint, write_map


def embedded_text(output: Path) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), output.read_text(encoding="utf-8")
    )


def a_point(name: str, cluster: str, *coordinates: float) -> MapPoint:
    return MapPoint(
        name=name, url=f"https://github.com/{name}", cluster=cluster, summary=f"About {name}.", coordinates=coordinates
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

    def test_writes_a_three_dimensional_map(self, tmp_path: Path) -> None:
        output = tmp_path / "map.html"
        points = [a_point("a/one", "Rust Tools", 0.0, 0.0, 0.0), a_point("b/two", "Rust Tools", 0.1, 0.1, 0.1)]

        write_map(points, output=output, title="stars")

        assert "scatter3d" in embedded_text(output)

    class TestWhenPointsMixDimensions:
        def test_rejects_them(self, tmp_path: Path) -> None:
            points = [a_point("a/one", "Rust Tools", 0.0, 0.0), a_point("b/two", "Rust Tools", 0.1, 0.1, 0.1)]

            with pytest.raises(ValueError, match="dimensions"):
                write_map(points, output=tmp_path / "map.html", title="stars")
