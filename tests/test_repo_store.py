from typing import TYPE_CHECKING

import pytest

from dewey.repos import MalformedSnapshotError, RepoStore, StarredRepo
from tests.builders import a_snapshot

if TYPE_CHECKING:
    from pathlib import Path


class TestRepoStore:
    def test_loads_a_saved_snapshot_as_a_starred_repo(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(
            a_snapshot(7, full_name="acme/rocket", language="Rust", topics=("space",), readme="# Rocket")
        )

        repo = store.load(7)

        assert repo == StarredRepo(
            id=7,
            full_name="acme/rocket",
            description="a repo",
            language="Rust",
            topics=("space",),
            license="MIT License",
            size_kb=42,
            created_year="2021",
            url="https://github.com/acme/rocket",
            readme="# Rocket",
        )

    def test_knows_which_repos_it_holds(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(a_snapshot(7))

        assert store.has_repo(7)
        assert not store.has_repo(8)

    def test_round_trips_the_starred_ids(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)

        store.save_starred_ids([3, 1, 2])

        assert store.starred_ids() == [3, 1, 2]

    def test_round_trips_a_summary(self, tmp_path: Path) -> None:
        store = RepoStore(tmp_path)
        store.save_snapshot(a_snapshot(7))

        store.save_summary(7, "A Rust rocket simulator.")

        assert store.summary(7) == "A Rust rocket simulator."

    class TestWhenTheSnapshotHasNoReadme:
        def test_loads_with_no_readme(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7, readme=None))

            assert store.load(7).readme is None

    class TestWhenFieldsAreNull:
        def test_fills_the_placeholders_the_summary_prompt_expects(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7, description=None, language=None))

            repo = store.load(7)

            assert (repo.description, repo.language) == ("No description provided", "Unknown")

    class TestWhenNothingWasFetchedYet:
        def test_has_no_starred_ids(self, tmp_path: Path) -> None:
            assert not RepoStore(tmp_path).has_starred_ids()

        def test_has_no_summary(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7))

            assert not store.has_summary(7)

    class TestWhenTheRepoIsMissing:
        def test_raises_on_load(self, tmp_path: Path) -> None:
            with pytest.raises(FileNotFoundError):
                RepoStore(tmp_path).load(404)

    class TestWhenTheReadmeIsNotUtf8:
        def test_replaces_the_undecodable_bytes(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7, readme_bytes=b"caf\xe9"))

            assert store.load(7).readme == "caf\ufffd"

    class TestWhenTheReadmeIsTooLargeForTheApi:
        def test_loads_with_no_readme(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7, readme_encoding="none"))

            assert store.load(7).readme is None

    class TestWhenTheSnapshotIsMalformed:
        def test_refuses_a_url_off_github(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            store.save_snapshot(a_snapshot(7, url="javascript:alert(1)"))

            with pytest.raises(MalformedSnapshotError):
                store.load(7)

        def test_refuses_a_missing_field(self, tmp_path: Path) -> None:
            store = RepoStore(tmp_path)
            snapshot = a_snapshot(7)
            del snapshot.repo["created_at"]
            store.save_snapshot(snapshot)

            with pytest.raises(MalformedSnapshotError):
                store.load(7)
