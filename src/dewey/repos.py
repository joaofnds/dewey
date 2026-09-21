import json
from base64 import b64decode
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

GITHUB = "https://github.com/"
NO_DESCRIPTION = "No description provided"
NO_LICENSE = "No license provided"
UNKNOWN_LANGUAGE = "Unknown"


class MalformedSnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class RepoSnapshot:
    repo: dict[str, Any]
    readme: dict[str, Any] | None

    @property
    def id(self) -> int:
        return int(self.repo["id"])


@dataclass(frozen=True)
class StarredRepo:
    id: int
    full_name: str
    description: str
    language: str
    topics: tuple[str, ...]
    license: str
    size_kb: int
    created_year: str
    url: str
    readme: str | None

    @classmethod
    def from_snapshot(cls, snapshot: RepoSnapshot) -> StarredRepo:
        repo = snapshot.repo
        license_field: dict[str, Any] = repo.get("license") or {}
        topics: list[Any] = repo.get("topics") or []

        try:
            url = str(repo["html_url"])
            if not url.startswith(GITHUB):
                message = f"repo {snapshot.id} has a url off GitHub: {url!r}"
                raise MalformedSnapshotError(message)

            return cls(
                id=snapshot.id,
                full_name=str(repo["full_name"]),
                description=str(repo.get("description") or NO_DESCRIPTION),
                language=str(repo.get("language") or UNKNOWN_LANGUAGE),
                topics=tuple(str(topic) for topic in topics),
                license=str(license_field.get("name") or NO_LICENSE),
                size_kb=int(repo.get("size") or 0),
                created_year=str(repo["created_at"])[:4],
                url=url,
                readme=decode_readme(snapshot.readme),
            )
        except KeyError as missing:
            message = f"repo {snapshot.id} is missing {missing}"
            raise MalformedSnapshotError(message) from missing


def decode_readme(readme: dict[str, Any] | None) -> str | None:
    if readme is None or readme.get("encoding") != "base64":
        return None

    return b64decode(str(readme.get("content", ""))).decode("utf-8", errors="replace").strip()


class RepoStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repos_dir = root / "repos"
        self.embeddings_dir = root / "embeddings"
        self.names_dir = root / "names"
        self.starred_ids_path = root / "starred_ids.txt"

    def has_starred_ids(self) -> bool:
        return self.starred_ids_path.exists()

    def starred_ids(self) -> list[int]:
        lines = self.starred_ids_path.read_text(encoding="utf-8").splitlines()

        return [int(line) for line in lines if line.strip()]

    def save_starred_ids(self, ids: list[int]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.starred_ids_path.write_text("".join(f"{repo_id}\n" for repo_id in ids), encoding="utf-8")

    def has_repo(self, repo_id: int) -> bool:
        return self._repo_path(repo_id).exists()

    def save_snapshot(self, snapshot: RepoSnapshot) -> None:
        folder = self._folder(snapshot.id)
        folder.mkdir(parents=True, exist_ok=True)

        self._repo_path(snapshot.id).write_text(json.dumps(snapshot.repo, indent=2), encoding="utf-8")
        if snapshot.readme is not None:
            self._readme_path(snapshot.id).write_text(json.dumps(snapshot.readme, indent=2), encoding="utf-8")

    def load(self, repo_id: int) -> StarredRepo:
        repo = json.loads(self._repo_path(repo_id).read_text(encoding="utf-8"))
        readme_path = self._readme_path(repo_id)
        readme = json.loads(readme_path.read_text(encoding="utf-8")) if readme_path.exists() else None

        return StarredRepo.from_snapshot(RepoSnapshot(repo=repo, readme=readme))

    def has_summary(self, repo_id: int) -> bool:
        return self._summary_path(repo_id).exists()

    def summary(self, repo_id: int) -> str:
        return self._summary_path(repo_id).read_text(encoding="utf-8").strip()

    def save_summary(self, repo_id: int, text: str) -> None:
        self._summary_path(repo_id).write_text(text, encoding="utf-8")

    def _folder(self, repo_id: int) -> Path:
        return self.repos_dir / str(repo_id)

    def _repo_path(self, repo_id: int) -> Path:
        return self._folder(repo_id) / "repo.json"

    def _readme_path(self, repo_id: int) -> Path:
        return self._folder(repo_id) / "readme.json"

    def _summary_path(self, repo_id: int) -> Path:
        return self._folder(repo_id) / "summary.txt"
