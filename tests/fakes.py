from collections import deque
from typing import Any


class FakeGitHub:
    def __init__(self) -> None:
        self.stars: dict[str, list[dict[str, Any]]] = {}
        self.readmes: dict[str, dict[str, Any]] = {}
        self.readme_requests: list[str] = []
        self.star_requests: list[str] = []

    def reset(self) -> None:
        self.__init__()

    def seed_star(self, username: str, repo: dict[str, Any], readme: dict[str, Any] | None = None) -> None:
        self.stars.setdefault(username, []).append(repo)
        if readme is not None:
            self.readmes[str(repo["full_name"])] = readme

    def starred(self, username: str) -> list[dict[str, Any]]:
        self.star_requests.append(username)

        return list(self.stars.get(username, []))

    def readme(self, full_name: str) -> dict[str, Any] | None:
        self.readme_requests.append(full_name)

        return self.readmes.get(full_name)


class FakeLLM:
    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.prompts: list[str] = []
        self.responses: deque[str] = deque()

    def reset(self) -> None:
        self.prompts.clear()
        self.responses.clear()

    def queue(self, *responses: str) -> None:
        self.responses.extend(responses)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.popleft()

        return f"response {len(self.prompts)}"
