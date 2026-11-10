"""Fetching pinned third-party sources into data/raw, verified by SHA-256.

Every source in configs/data/relations.yaml names an immutable location (a git commit, a Hub
revision, a wiki oldid) or, for unversioned files, pins the hash itself. A file
already on disk is reused only if its hash matches; any mismatch fails the build
(failure F14: hash the files actually used). Nothing fetched here is committed.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saeselect.provenance import EngineeringFailed, sha256_file

USER_AGENT = "saeselect-data-build/0.1 (research; contact via repository owner)"


@dataclass(frozen=True)
class FetchedFile:
    key: str
    path: Path
    sha256: str
    origin: str
    license: str

    def card_entry(self) -> dict[str, str]:
        return {"file": self.key, "origin": self.origin, "sha256": self.sha256, "license": self.license}


def _verify(path: Path, expected: str, origin: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise EngineeringFailed(
            "source_hash", f"{origin}: sha256 {actual} does not match the pinned {expected}; the source changed"
        )
    return actual


def fetch_url(url: str, dest: Path, *, sha256: str) -> str:
    """Download ``url`` to ``dest`` unless a file with the right hash is already there."""
    if dest.is_file() and sha256_file(dest) == sha256:
        return sha256
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    tmp.replace(dest)
    return _verify(dest, sha256, url)


def fetch_hub_file(repo_id: str, filename: str, *, revision: str, sha256: str, repo_type: str = "dataset") -> Path:
    """Download one file from the Hugging Face Hub at a pinned revision (uses the HF cache)."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, filename, revision=revision, repo_type=repo_type))
    _verify(path, sha256, f"{repo_id}@{revision}/{filename}")
    return path


class SourceStore:
    """Resolves the ``sources`` section of configs/data/relations.yaml to verified local files."""

    def __init__(self, sources: dict[str, Any], raw_dir: str | Path):
        self.sources = sources
        self.raw_dir = Path(raw_dir)
        self.fetched: dict[str, FetchedFile] = {}

    # A source may name a local copy instead (``path`` for a single file,
    # ``local_dir`` for a repository or Hub dataset): used by the CPU tests and
    # for offline rebuilds. The pinned hash is verified either way.

    def url_file(self, name: str) -> FetchedFile:
        spec = self.sources[name]
        if "path" in spec:
            path = Path(spec["path"])
            _verify(path, spec["sha256"], str(path))
            return self._note(name, path, spec["sha256"], f"local:{path.as_posix()} (pinned {spec['url']})", spec["license"])
        dest = self.raw_dir / name / Path(spec["url"].split("?")[0]).name
        if "?" in spec["url"]:
            dest = dest.with_name(dest.name + ".raw")
        fetch_url(spec["url"], dest, sha256=spec["sha256"])
        return self._note(name, dest, spec["sha256"], spec["url"], spec["license"])

    def repo_file(self, name: str, relpath: str) -> FetchedFile:
        """A file from a pinned git repository served over raw.githubusercontent.com."""
        spec = self.sources[name]
        sha = spec["files"][relpath]
        if "local_dir" in spec:
            path = Path(spec["local_dir"]) / relpath
            _verify(path, sha, str(path))
            return self._note(f"{name}:{relpath}", path, sha, f"local:{path.as_posix()}", spec["license"])
        dest = self.raw_dir / name / relpath
        url = f"{spec['base_url']}/{relpath}"
        fetch_url(url, dest, sha256=sha)
        return self._note(f"{name}:{relpath}", dest, sha, url, spec["license"])

    def hub_file(self, name: str, filename: str) -> FetchedFile:
        spec = self.sources[name]
        sha = spec["files"][filename]
        if "local_dir" in spec:
            path = Path(spec["local_dir"]) / filename
            _verify(path, sha, str(path))
            return self._note(f"{name}:{filename}", path, sha, f"local:{path.as_posix()}", spec["license"])
        path = fetch_hub_file(spec["repo_id"], filename, revision=spec["revision"], sha256=sha)
        origin = f"hf://datasets/{spec['repo_id']}@{spec['revision']}/{filename}"
        return self._note(f"{name}:{filename}", path, sha, origin, spec["license"])

    def _note(self, key: str, path: Path, sha: str, origin: str, license_: str) -> FetchedFile:
        fetched = FetchedFile(key=key, path=Path(path), sha256=sha, origin=origin, license=license_)
        self.fetched[key] = fetched
        return fetched

    def card_entries(self, prefix: str) -> list[dict[str, str]]:
        return [f.card_entry() for key, f in sorted(self.fetched.items()) if key == prefix or key.startswith(prefix + ":")]
