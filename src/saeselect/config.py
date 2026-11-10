"""Loading configs/*.yaml.

Configs hold only (declared) values and paths. Each
declared value carries a one-line comment with its reason and the section that
declares it; the loader cannot check comments, so review does. What the loader
does enforce: a derived value never lives in a config (keys under ``derived``
are refused), and a missing declared value is an error at the point of use,
never a silent default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from saeselect.provenance import sha256_file


class ConfigError(Exception):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config {path.as_posix()} not found")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path.as_posix()} must be a mapping")
    if "derived" in data:
        raise ConfigError(
            f"config {path.as_posix()} has a 'derived' section; derived values live in "
            "derived.json and frozen_protocol.json, never in configs"
        )
    data["_source"] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    return data


def require(config: dict[str, Any], dotted: str) -> Any:
    """Return a declared value, raising if it is absent (no silent defaults)."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            source = config.get("_source", {}).get("path", "<config>")
            raise ConfigError(f"declared value '{dotted}' missing from {source}")
        node = node[part]
    return node
