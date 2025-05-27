import json
import os
from base64 import b64decode
from os import path


class RepoData:
    def __init__(self, id: int):
        self.id = id
        self.folder = f"data/repos/{id}"
        self.repo_path = f"{self.folder}/repo.json"
        self.readme_path = f"{self.folder}/readme.json"
        self.summary_path = f"{self.folder}/summary.txt"

        self._cached_repo_json = None
        self._cached_readme_json = None
        self._cached_readme_content = None
        self._cached_summary = None

    def create_folder(self):
        os.makedirs(self.folder)

    def folder_exists(self) -> bool:
        return path.exists(self.folder)

    def repo_exists(self) -> bool:
        return path.exists(self.repo_path)

    def readme_exists(self) -> bool:
        return path.exists(self.readme_path)

    def summary_exists(self) -> bool:
        return path.exists(self.summary_path)

    def write_repo_json(self, data: dict):
        with open(self.repo_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._cached_repo_json = data

    def write_readme_json(self, data: dict):
        with open(self.readme_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._cached_readme_json = data
        self._cached_readme_content = None  # Invalidate cached content

    def write_summary(self, data: str):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(data)
        self._cached_summary = data

    def repo_json(self) -> dict:
        if self._cached_repo_json is None:
            with open(self.repo_path, "r", encoding="utf-8") as f:
                self._cached_repo_json = json.load(f)

        return self._cached_repo_json

    def readme_json(self) -> dict:
        if self._cached_readme_json is None:
            with open(self.readme_path, "r", encoding="utf-8") as f:
                self._cached_readme_json = json.load(f)

        return self._cached_readme_json

    def readme_content(self) -> str:
        if self._cached_readme_content is None:
            readme_json = self.readme_json()
            content = readme_json.get("content", "")
            return b64decode(content).decode("utf-8").strip()

        return self._cached_readme_content

    def summary(self) -> str:
        if self._cached_summary is None:
            with open(self.summary_path, "r", encoding="utf-8") as f:
                self._cached_summary = f.read().strip()

        return self._cached_summary

    def full_name(self) -> str:
        repo_json = self.repo_json()
        assert "full_name" in repo_json
        return repo_json["full_name"]
