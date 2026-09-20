from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tqdm import tqdm

from dewey.prompts import load_prompt

if TYPE_CHECKING:
    from logging import Logger

    from dewey.llm import LLM
    from dewey.repos import RepoStore, StarredRepo

README_LIMIT = 4000


class SummaryWriter:
    def __init__(self, store: RepoStore, llm: LLM, workers: int, logger: Logger, *, overwrite: bool) -> None:
        self.store = store
        self.llm = llm
        self.workers = workers
        self.overwrite = overwrite
        self.logger = logger
        self.template = load_prompt("summary")

    def run(self, repos: list[StarredRepo]) -> None:
        pending = [repo for repo in repos if self.overwrite or not self.store.has_summary(repo.id)]
        self.logger.info("summarizing %d of %d repos with %s", len(pending), len(repos), self.llm.model)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(self.summarize, pending)
            for _ in tqdm(results, total=len(pending), desc="Summarizing", disable=not pending):
                pass

    def summarize(self, repo: StarredRepo) -> None:
        summary = self.llm.generate(self.prompt(repo))
        self.store.save_summary(repo.id, summary)

    def prompt(self, repo: StarredRepo) -> str:
        return self.template.format(
            full_name=repo.full_name,
            description=repo.description,
            language=repo.language,
            license=repo.license,
            size=repo.size_kb,
            created_at=repo.created_year,
            topics=", ".join(repo.topics) or "None",
            readme_section=readme_section(repo.readme),
        )


def readme_section(readme: str | None) -> str:
    if readme is None:
        return "- **README:** Not available"

    if len(readme) > README_LIMIT:
        readme = readme[:README_LIMIT] + "\n\n[README truncated...]"

    return f"- **README:**\n{readme}"
