import json

import httpx
import pytest

from dewey.stars import GitHubRestClient, MissingTokenError


def a_repo(repo_id: int) -> dict[str, object]:
    return {"id": repo_id, "full_name": f"org/repo{repo_id}"}


def github_api(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/users/ada/starred" and request.url.params.get("page") != "2":
        return httpx.Response(
            200,
            json=[a_repo(index) for index in range(100)],
            headers={"Link": '<https://api.github.com/users/ada/starred?per_page=100&page=2>; rel="next"'},
        )
    if request.url.path == "/users/ada/starred":
        return httpx.Response(200, json=[a_repo(index) for index in range(100, 150)])
    if request.url.path == "/repos/org/repo0/readme":
        return httpx.Response(200, json={"encoding": "base64", "content": "IyBoaQ=="})

    return httpx.Response(404, json={"message": "Not Found"})


def a_client() -> GitHubRestClient:
    return GitHubRestClient(a_transport(), "a-token")


def a_client_without_token() -> GitHubRestClient:
    return GitHubRestClient(a_transport(), "")


def a_transport() -> httpx.Client:
    return httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(github_api))


class TestGitHubRestClient:
    def test_follows_the_link_header_across_pages(self) -> None:
        repos = a_client().starred("ada")

        assert [repo["id"] for repo in repos] == list(range(150))

    def test_returns_the_readme_object(self) -> None:
        readme = a_client().readme("org/repo0")

        assert readme == {"encoding": "base64", "content": "IyBoaQ=="}

    class TestWhenTheRepoHasNoReadme:
        def test_returns_none(self) -> None:
            assert a_client().readme("org/repo1") is None

    class TestWhenTheTokenIsMissing:
        def test_refuses_before_any_request(self) -> None:
            with pytest.raises(MissingTokenError):
                a_client_without_token().starred("ada")


def test_the_fake_api_serves_json() -> None:
    assert json.loads(github_api(httpx.Request("GET", "https://api.github.com/repos/org/repo0/readme")).content)
