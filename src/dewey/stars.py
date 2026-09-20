from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from dewey.repos import RepoSnapshot, RepoStore, StarredRepo

if TYPE_CHECKING:
    from logging import Logger

HTTP_NOT_FOUND = 404
PAGE_SIZE = 100


class GitHubClient(Protocol):
    def starred(self, username: str) -> list[dict[str, Any]]: ...
    def readme(self, full_name: str) -> dict[str, Any] | None: ...


class GitHubRestClient:
    def __init__(self, token: str, timeout_seconds: int, base_url: str = "https://api.github.com") -> None:
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def starred(self, username: str) -> list[dict[str, Any]]:
        repos: list[dict[str, Any]] = []
        url: str | None = f"/users/{username}/starred?per_page={PAGE_SIZE}"
        while url is not None:
            response = self.client.get(url)
            response.raise_for_status()
            repos.extend(response.json())
            url = response.links.get("next", {}).get("url")

        return repos

    def readme(self, full_name: str) -> dict[str, Any] | None:
        response = self.client.get(f"/repos/{full_name}/readme")
        if response.status_code == HTTP_NOT_FOUND:
            return None
        response.raise_for_status()

        return response.json()


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
