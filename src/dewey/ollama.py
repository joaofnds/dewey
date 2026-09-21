import httpx

from dewey.llm import TruncatedReplyError


class Ollama:
    def __init__(self, model: str, base_url: str, timeout_seconds: int) -> None:
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def generate(self, prompt: str) -> str:
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9},
            },
        )
        response.raise_for_status()
        reply = response.json()
        if reply.get("done_reason") == "length":
            message = f"{self.model} hit its output length limit"
            raise TruncatedReplyError(message)

        return str(reply["response"]).strip()
