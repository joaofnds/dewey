import httpx


class Copilot:
    def __init__(self, token: str, model: str, timeout: int):
        self.token = token
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            "https://api.githubcopilot.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.80.1",
            },
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 250,
                "top_p": 0.9,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
