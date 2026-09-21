from typing import Protocol


class TruncatedReplyError(RuntimeError):
    pass


class LLM(Protocol):
    @property
    def model(self) -> str: ...
    def generate(self, prompt: str) -> str: ...
