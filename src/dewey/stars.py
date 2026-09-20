from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol

import github
import github.Auth

from dewey.repos import RepoSnapshot, RepoStore, StarredRepo

if TYPE_CHECKING:
    from logging import Logger

HTTP_NOT_FOUND = 404


class GitHubClient(Protocol):
    def starred(self, username: str) -> list[dict[str, Any]]: ...
    def readme(self, full_name: str) -> dict[str, Any] | None: ...


class PyGithubClient:
    def __init__(self, token: str, timeout_seconds: int) -> None:
        self.client = github.Github(auth=github.Auth.Token(token), per_page=100, timeout=timeout_seconds)

    def starred(self, username: str) -> list[dict[str, Any]]:
        return [repo.raw_data for repo in self.client.get_user(username).get_starred()]

    def readme(self, full_name: str) -> dict[str, Any] | None:
        try:
            return self.client.get_repo(full_name, lazy=True).get_readme().raw_data
        except github.GithubException as error:
            if error.status == HTTP_NOT_FOUND:
                return None
            raise


class StarFetcher:
    def __init__(self, store: RepoStore, github: GitHubClient, workers: int, logger: Logger) -> None:
        self.store = store
        self.github = github
        self.workers = workers
        self.logger = logger

    def run(self, username: str, *, refresh: bool) -> list[StarredRepo]:
        if self.store.has_starred_ids() and not refresh:
            ids = self.store.starred_ids()
            self.logger.info("using the %d cached stars for %s", len(ids), username)
            return [self.store.load(repo_id) for repo_id in ids]

        self.logger.info("listing stars for %s", username)
        starred = self.github.starred(username)
        self.logger.info("found %d stars, fetching the ones not yet stored", len(starred))

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            list(executor.map(self._store_if_new, starred))

        ids = [int(repo["id"]) for repo in starred]
        self.store.save_starred_ids(ids)

        return [self.store.load(repo_id) for repo_id in ids]

    def _store_if_new(self, repo: dict[str, Any]) -> None:
        if self.store.has_repo(int(repo["id"])):
            return

        readme = self.github.readme(str(repo["full_name"]))
        self.store.save_snapshot(RepoSnapshot(repo=repo, readme=readme))
