import re
from collections import deque
from hashlib import blake2b
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FakeGitHub:
    def __init__(self) -> None:
        self.stars: dict[str, list[dict[str, Any]]] = {}
        self.readmes: dict[str, dict[str, Any]] = {}
        self.readme_requests: list[str] = []
        self.star_requests: list[str] = []

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

    def queue(self, *responses: str) -> None:
        self.responses.extend(responses)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.popleft()

        return f"response {len(self.prompts)}"


class FakeEmbedder:
    def __init__(self, model_name: str = "fake-embedder", dimensions: int = 8) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))

        return np.stack([self.vector(text) for text in texts]).astype(np.float32)

    def vector(self, text: str) -> NDArray[np.float32]:
        seed = int.from_bytes(blake2b(text.encode(), digest_size=4).digest(), "big")
        vector = np.random.default_rng(seed).standard_normal(self.dimensions).astype(np.float32)

        return vector / np.linalg.norm(vector)


class SummaryStub:
    model = "summary-stub"

    def generate(self, prompt: str) -> str:
        name = re.search(r"\*\*Name:\*\* (\S+)", prompt)
        language = re.search(r"\*\*Language:\*\* (\S+)", prompt)
        assert name is not None
        assert language is not None

        return f"{language.group(1)} project {name.group(1)}"


class KeywordEmbedder:
    def __init__(self, dimensions: int = 8) -> None:
        self.model_name = "keyword-embedder"
        self.dimensions = dimensions
        self.axes: dict[str, int] = {}

    def anchor(self, keyword: str, axis: int) -> None:
        self.axes[keyword.lower()] = axis

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self.vector(text) for text in texts]).astype(np.float32)

    def vector(self, text: str) -> NDArray[np.float32]:
        seed = int.from_bytes(blake2b(text.encode(), digest_size=4).digest(), "big")
        vector = np.random.default_rng(seed).standard_normal(self.dimensions).astype(np.float32) * 0.05
        for keyword, axis in self.axes.items():
            if keyword in text.lower():
                vector[axis] += 1.0

        return vector / np.linalg.norm(vector)
