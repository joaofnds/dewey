from base64 import b64encode

from dewey.repos import RepoSnapshot, StarredRepo


def a_snapshot(
    repo_id: int,
    *,
    full_name: str = "octo/cat",
    description: str | None = "a repo",
    language: str | None = "Python",
    topics: tuple[str, ...] = (),
    readme: str | None = "# hello",
    readme_bytes: bytes | None = None,
    readme_encoding: str = "base64",
    url: str | None = None,
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
        "html_url": url or f"https://github.com/{full_name}",
    }
    content = readme_bytes if readme_bytes is not None else (None if readme is None else readme.encode())
    readme_json = None if content is None else {"content": b64encode(content).decode(), "encoding": readme_encoding}

    return RepoSnapshot(repo=repo, readme=readme_json)


def a_repo(repo_id: int, full_name: str = "octo/cat") -> StarredRepo:
    return StarredRepo(
        id=repo_id,
        full_name=full_name,
        description="a repo",
        language="Python",
        topics=(),
        license="MIT License",
        size_kb=1,
        created_year="2021",
        url=f"https://github.com/{full_name}",
        readme=None,
    )
