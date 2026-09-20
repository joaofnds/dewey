from base64 import b64encode

from dewey.repos import RepoSnapshot


def a_snapshot(
    repo_id: int,
    *,
    full_name: str = "octo/cat",
    description: str | None = "a repo",
    language: str | None = "Python",
    topics: tuple[str, ...] = (),
    readme: str | None = "# hello",
) -> RepoSnapshot:
    repo = {
        "id": repo_id,
        "full_name": full_name,
        "description": description,
        "language": language,
        "topics": list(topics),
        "license": {"name": "MIT License"},
        "size": 42,
        "created_at": "2021-05-01T00:00:00Z",
        "html_url": f"https://github.com/{full_name}",
    }
    readme_json = None if readme is None else {"content": b64encode(readme.encode()).decode(), "encoding": "base64"}

    return RepoSnapshot(repo=repo, readme=readme_json)
