import anthropic
from anthropic.types import TextBlock


class Claude:
    def __init__(self, model: str, max_tokens: int, timeout_seconds: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(timeout=timeout_seconds, max_retries=2)

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return "".join(block.text for block in message.content if isinstance(block, TextBlock)).strip()
