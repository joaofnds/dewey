from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger
from os import path

import github
import github.Repository

from lib.repo_data import RepoData


class RepoFetcher:
    def __init__(
        self,
        logger: Logger,
        token: str,
        workers: int,
    ):
        self.logger = logger
        self.token = token
        self.workers = workers
        self.github_client = github.Github(
            auth=github.Auth.Token(self.token),
            per_page=100,
        )
        self.ids_file = "ids.txt"

    def run(self, username: str) -> list[RepoData]:
        if path.exists(self.ids_file):
            self.logger.info(f"File '{self.ids_file}' already exists. Skipping fetching.")
            return [RepoData(int(line.strip())) for line in open(self.ids_file) if line.strip().isdigit()]

        self.logger.info(f"Fetching starred repositories for '{username}'...")
        starred_repos = list(self.github_client.get_user(username).get_starred())
        self.logger.info(f"Found {len(starred_repos)} starred repositories.")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_repo = {executor.submit(self.__process_repo, repo): repo for repo in starred_repos}

            for future in as_completed(future_to_repo):
                result = future.result()
                self.logger.debug(f"Result: {result}")

        with open(self.ids_file, "w") as f:
            for repo in starred_repos:
                f.write(f"{repo.id}\n")

        return [RepoData(repo.id) for repo in starred_repos]

    def __process_repo(self, repo: github.Repository.Repository):
        repo_data = RepoData(repo.id)

        if repo_data.folder_exists():
            return f"Skipped {repo.full_name}: already exists"

        repo_data.create_folder()
        repo_data.write_repo_json(repo.raw_data)

        try:
            readme = repo.get_readme()
            repo_data.write_readme_json(readme.raw_data)
            return f"Completed {repo.full_name}"
        except github.GithubException as e:
            if e.status == 404:
                return f"Completed {repo.full_name} (no README)"
            else:
                raise e
