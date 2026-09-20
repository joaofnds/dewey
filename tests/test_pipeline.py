import logging
from typing import TYPE_CHECKING

from dewey.clusterer import HdbscanSettings
from dewey.pipeline import PipelineSettings, Services, run
from dewey.reducer import UmapSettings
from dewey.repos import RepoStore
from tests.builders import a_snapshot
from tests.fakes import EchoLLM, FakeGitHub, FakeLLM, KeywordEmbedder
from tests.test_map_writer import embedded_text

if TYPE_CHECKING:
    from pathlib import Path

LANGUAGES = ("Rust", "Python", "JavaScript")


def a_github_with_thirty_stars() -> FakeGitHub:
    github = FakeGitHub()
    for index in range(30):
        language = LANGUAGES[index % 3]
        snapshot = a_snapshot(
            index, full_name=f"{language.lower()}/repo{index}", language=language, readme=f"# {language} {index}"
        )
        github.seed_star("ada", snapshot.repo, snapshot.readme)

    return github


def settings(tmp_path: Path, dimensions: int = 2) -> PipelineSettings:
    return PipelineSettings(
        username="ada",
        output=tmp_path / "map.html",
        title="ada's stars",
        plot_dimensions=dimensions,
        refresh_stars=False,
        overwrite_summaries=False,
        summary_workers=2,
        representatives=3,
        umap=UmapSettings(n_neighbors=8, metric="cosine", n_epochs=100),
        hdbscan=HdbscanSettings(min_cluster_size=5, min_samples=2),
        seed=3,
    )


def services(tmp_path: Path, namer: FakeLLM) -> Services:
    embedder = KeywordEmbedder()
    for axis, language in enumerate(LANGUAGES):
        embedder.anchor(language, axis)

    return Services(
        store=RepoStore(tmp_path / "data"),
        github=a_github_with_thirty_stars(),
        summarizer=EchoLLM(),
        namer=namer,
        embedder=embedder,
        logger=logging.getLogger("test"),
    )


class TestRun:
    def test_writes_a_map_naming_every_language_cluster(self, tmp_path: Path) -> None:
        namer = FakeLLM()
        namer.queue("Rust Systems", "Python Tools", "JavaScript Apps")

        result = run(settings(tmp_path), services(tmp_path, namer))

        html = embedded_text(result.output)
        assert result.clusters == 3
        assert "Rust Systems" in html
        assert "Python Tools" in html
        assert "JavaScript Apps" in html
        assert "rust/repo0" in html

    def test_keeps_every_repo_on_the_map(self, tmp_path: Path) -> None:
        result = run(settings(tmp_path), services(tmp_path, FakeLLM()))

        assert result.repos == 30

    def test_writes_a_three_dimensional_map_when_asked(self, tmp_path: Path) -> None:
        result = run(settings(tmp_path, dimensions=3), services(tmp_path, FakeLLM()))

        assert "scatter3d" in embedded_text(result.output)
