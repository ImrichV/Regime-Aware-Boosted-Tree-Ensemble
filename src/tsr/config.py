from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


_PATH_KEY_SUFFIXES = ("_path", "_root")


def _project_root_for(config_path: Path) -> Path:
    """Return the repository root used for relative paths in a config file."""
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _resolve_config_paths(value: Any, project_root: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _resolve_config_paths(item_value, project_root, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_config_paths(item, project_root, key) for item in value]
    if isinstance(value, str) and key and key.endswith(_PATH_KEY_SUFFIXES):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if "${" in expanded:
            raise ConfigError(f"Unresolved environment variable in {key}: {value}")
        path = Path(expanded)
        return str(path.resolve() if path.is_absolute() else (project_root / path).resolve())
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML value must be a mapping.")
    project_root = _project_root_for(config_path)
    data = _resolve_config_paths(data, project_root)
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(project_root)
    return data


def canonical_json(data: dict[str, Any]) -> str:
    filtered = {key: value for key, value in data.items() if not key.startswith("_")}
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(data: dict[str, Any], length: int = 12) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]
