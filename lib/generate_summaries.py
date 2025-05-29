from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger

from lib.llm import LLM
from lib.repo_data import RepoData


class GenerateSummaries:
    def __init__(
        self,
        logger: Logger,
        workers: int,
        overwrite: bool,
        llm: LLM,
    ):
        self.logger = logger
        self.workers = workers
        self.overwrite = overwrite
        self.llm = llm
        self.max_readme_length = 4000
        self.prompt_template = """Generate a technical abstract for this GitHub repository. This abstract will be used for vector embedding and clustering, so focus on distinctive technical characteristics.

**Repository Data:**
- **Name:** {full_name}
- **Description:** {description}
- **Language:** {language}
- **License:** {license}
- **Size:** {size} KB
- **Created:** {created_at}
- **Topics:** {topics}
{readme_section}

**Requirements:**
Write a single, dense paragraph (3-5 sentences) that captures:

1. **Primary function and domain** - What problem does it solve? What technical domain (web dev, ML, systems, etc.)?
2. **Core features and capabilities** - What can users do with it?
3. **Technology stack** - Key languages, frameworks, libraries, architectural patterns
4. **Project type** - Library, application, framework, tool, etc.
5. **Target users** - Who uses this? (only if clearly evident)

**Style:**
- Start directly with technical details, no introductory phrases
- Use precise technical language
- Focus on distinguishing characteristics for clustering
- Keep it factual and information-dense

**Examples:**
• "A TypeScript HTTP client library providing promise-based request handling with automatic retries, request/response interceptors, and built-in timeout management for Node.js and browser environments."
• "A Python command-line tool for automated code formatting and linting, integrating Black, isort, and flake8 to enforce consistent style across Python projects."
• "A React component library implementing Google's Material Design system with TypeScript support, featuring customizable themes, accessibility compliance, and tree-shaking optimization."

Generate the abstract:"""

    def run(self, repos: list[RepoData]):
        self.logger.info(f"Processing {len(repos)} repositories...")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_repo = {executor.submit(self.process_repo, repo): repo for repo in repos}
            total = len(repos)

            for i, future in enumerate(as_completed(future_to_repo), 1):
                future.result()
                self.logger.info(f"Processed {i}/{total} repositories")

        self.logger.info("Processing complete!")

    def process_repo(self, repo: RepoData):
        if repo.summary_exists() and not self.overwrite:
            self.logger.info(f"Skipping {repo.full_name()} - summary already exists.")
            return

        prompt = self.summary_prompt(repo)
        summary = self.llm.generate(prompt)
        repo.write_summary(summary)

    def summary_prompt(self, repo_data: RepoData) -> str:
        repo = repo_data.repo_json()

        return self.prompt_template.format(
            full_name=repo["full_name"],
            description=repo.get("description", "No description provided"),
            topics=", ".join(repo.get("topics", [])) or "None",
            language=repo.get("language", "Unknown"),
            license=(repo.get("license") or {}).get("name", "No license provided"),
            size=repo.get("size", 0),
            created_at=repo["created_at"][:4],  # year only
            readme_section=self.format_readme(repo_data),
        )

    def format_readme(self, repo_data: RepoData) -> str:
        if not repo_data.readme_exists():
            return "- **README:** Not available"

        readme = repo_data.readme_content()

        if len(readme) > self.max_readme_length:
            readme = readme[: self.max_readme_length] + "\n\n[README truncated...]"

        return f"- **README:**\n{readme}"
