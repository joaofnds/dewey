import httpx


class Gemini:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/v1beta/models/{self.model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Goog-Api-Key": self.api_key,
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 250,
                    "topP": 0.9,
                    "topK": 40,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
