import os


def must_get_env(key: str) -> str:
    value = os.getenv(key)

    if value in (None, "", "None", "null"):
        raise ValueError(f"Environment variable '{key}' is not set.")

    return value
