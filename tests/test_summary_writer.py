import logging
from typing import TYPE_CHECKING

from dewey.repos import RepoStore
from dewey.summaries import README_LIMIT, SummaryWriter
from tests.builders import a_snapshot
from tests.fakes import FakeLLM

if TYPE_CHECKING:
    from pathlib import Path


def a_writer(store: RepoStore, llm: FakeLLM, *, overwrite: bool = False) -> SummaryWriter:
    return SummaryWriter(store=store, llm=llm, workers=1, overwrite=overwrite, logger=logging.getLogger("test"))


class TestSummaryWriter:
    def test_stores_the_llm_answer_as_the_repo_summary(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(a_snapshot(1))
        llm = FakeLLM()
        llm.queue("A tiny library.")

        a_writer(store, llm).run([store.load(1)])

        assert store.summary(1) == "A tiny library."

    def test_prompts_with_the_repo_facts(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(
            a_snapshot(
                1,
                full_name="acme/rocket",
                description="Launch things",
                language="Rust",
                topics=("space", "sim"),
                readme="# Rocket\nIt flies.",
            ),
        )
        llm = FakeLLM()

        a_writer(store, llm).run([store.load(1)])

        prompt = llm.prompts[0]
        assert "acme/rocket" in prompt
        assert "Launch things" in prompt
        assert "Rust" in prompt
        assert "space, sim" in prompt
        assert "It flies." in prompt

    def test_truncates_a_long_readme(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(a_snapshot(1, readme="x" * (README_LIMIT + 100)))
        llm = FakeLLM()

        a_writer(store, llm).run([store.load(1)])

        assert "x" * README_LIMIT + "\n\n[README truncated...]" in llm.prompts[0]
        assert "x" * (README_LIMIT + 1) not in llm.prompts[0]

    class TestWhenTheSummaryExists:
        def test_keeps_it(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(1))
            store.save_summary(1, "Kept.")
            llm = FakeLLM()

            a_writer(store, llm).run([store.load(1)])

            assert store.summary(1) == "Kept."
            assert llm.prompts == []

        def test_overwrites_when_asked(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(1))
            store.save_summary(1, "Old.")
            llm = FakeLLM()
            llm.queue("New.")

            a_writer(store, llm, overwrite=True).run([store.load(1)])

            assert store.summary(1) == "New."

    class TestWhenTheRepoHasNoReadme:
        def test_says_so_in_the_prompt(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(1, readme=None))
            llm = FakeLLM()

            a_writer(store, llm).run([store.load(1)])

            assert "README:** Not available" in llm.prompts[0]
