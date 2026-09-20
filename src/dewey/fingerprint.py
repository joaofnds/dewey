from hashlib import blake2b


def fingerprint(parts: list[bytes], texts: list[str]) -> str:
    digest = blake2b()
    for part in parts:
        digest.update(part)
        digest.update(b"\x00")
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\x00")

    return digest.hexdigest()
