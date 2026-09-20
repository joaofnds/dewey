import logging
from typing import TYPE_CHECKING

from dewey.repos import RepoStore
from dewey.stars import StarFetcher
from tests.builders import a_snapshot
from tests.fakes import FakeGitHub

if TYPE_CHECKING:
    from pathlib import Path


def a_fetcher(store: RepoStore, github: FakeGitHub) -> StarFetcher:
    return StarFetcher(store=store, github=github, workers=2, logger=logging.getLogger("test"))


class TestStarFetcher:
    def test_stores_every_starred_repo_with_its_readme(self, tmp_path: Path) -> None:
        github = FakeGitHub()
        first = a_snapshot(1, full_name="a/one", readme="# one")
        second = a_snapshot(2, full_name="b/two", readme="# two")
        github.seed_star("ada", first.repo, first.readme)
        github.seed_star("ada", second.repo, second.readme)
        store = RepoStore(tmp_path)

        repos = a_fetcher(store, github).run("ada", refresh=False)

        assert [(repo.full_name, repo.readme) for repo in repos] == [("a/one", "# one"), ("b/two", "# two")]
        assert store.starred_ids() == [1, 2]

    def test_reuses_the_cached_star_list(self, tmp_path: Path) -> None:
        github = FakeGitHub()
        store = RepoStore(tmp_path)
        store.save_snapshot(a_snapshot(1, full_name="a/one"))
        store.save_starred_ids([1])

        repos = a_fetcher(store, github).run("ada", refresh=False)

        assert [repo.full_name for repo in repos] == ["a/one"]
        assert github.star_requests == []

    def test_does_not_refetch_a_repo_already_stored(self, tmp_path: Path) -> None:
        github = FakeGitHub()
        known = a_snapshot(1, full_name="a/one")
        new = a_snapshot(2, full_name="b/two")
        github.seed_star("ada", known.repo, known.readme)
        github.seed_star("ada", new.repo, new.readme)
        store = RepoStore(tmp_path)
        store.save_snapshot(known)

        a_fetcher(store, github).run("ada", refresh=False)

        assert github.readme_requests == ["b/two"]

    class TestWhenRefreshIsRequested:
        def test_lists_the_stars_again(self, tmp_path: Path) -> None:
            github = FakeGitHub()
            fresh = a_snapshot(2, full_name="b/two")
            github.seed_star("ada", fresh.repo, fresh.readme)
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(1, full_name="a/one"))
            store.save_starred_ids([1])

            repos = a_fetcher(store, github).run("ada", refresh=True)

            assert [repo.full_name for repo in repos] == ["b/two"]

    class TestWhenARepoHasNoReadme:
        def test_stores_it_without_one(self, tmp_path: Path) -> None:
            github = FakeGitHub()
            github.seed_star("ada", a_snapshot(1, readme=None).repo)
            store = RepoStore(tmp_path)

            repos = a_fetcher(store, github).run("ada", refresh=False)

            assert repos[0].readme is None
