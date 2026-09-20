from importlib import resources


def load_prompt(name: str) -> str:
    return resources.files("dewey.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
