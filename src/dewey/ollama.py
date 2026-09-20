import httpx


class Ollama:
    def __init__(self, model: str, base_url: str, timeout: int):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
            timeout=self.timeout,
        )
        assert response.status_code == 200
        return response.json()["response"].strip()
