import re
from pathlib import Path  # noqa: TC003


def embedded_text(output: Path) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), output.read_text(encoding="utf-8")
    )


def trace_names(html: str) -> list[str]:
    return re.findall(r'"name":"([^"]*)"', html)


def text_traces(html: str) -> list[str]:
    return re.findall(r'"mode":"text"[^}]*"text":\[([^\]]*)\]', html)
