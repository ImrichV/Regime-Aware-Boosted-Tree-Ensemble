from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, load_yaml

_REQUIRED_PROJECT_FILES = (
    "AGENTS.md",
    "README.md",
    "REPRODUCTION.md",
    "MODULE_CONTRACTS.md",
    "PROJECT_STATE.md",
    "ROADMAP.md",
    "pyproject.toml",
    "requirements-lock.txt",
)
_REQUIRED_PACKAGES = ("numpy", "pandas", "yaml", "scipy", "sklearn")


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def inspect_repository(project_root: str | Path, configs: list[str] | None = None) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    checks.append(_check("project_root_exists", root.is_dir(), str(root)))
    for relative in _REQUIRED_PROJECT_FILES:
        path = root / relative
        checks.append(_check(f"required_file:{relative}", path.is_file(), str(path)))

    source_root = root / "src" / "tsr"
    checks.append(_check("active_source_tree", source_root.is_dir(), str(source_root)))
    legacy_batches = sorted(source_root.glob("batch*.py")) if source_root.exists() else []
    checks.append(
        _check(
            "legacy_batches_excluded_from_active_package",
            not legacy_batches,
            "none" if not legacy_batches else ", ".join(str(path) for path in legacy_batches),
        )
    )

    for package_name in _REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            checks.append(_check(f"dependency:{package_name}", True, str(version)))
        except Exception as error:  # pragma: no cover - environment-specific
            checks.append(_check(f"dependency:{package_name}", False, str(error)))

    config_results: list[dict[str, Any]] = []
    for config_value in configs or []:
        config_path = Path(config_value)
        if not config_path.is_absolute():
            config_path = root / config_path
        try:
            config = load_yaml(config_path)
            path_checks: list[dict[str, Any]] = []

            def walk(
                value: Any,
                prefix: str = "",
                path_key: str | None = None,
            ) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if str(key).startswith("_"):
                            continue
                        key_path = f"{prefix}.{key}" if prefix else str(key)
                        walk(item, key_path, str(key))
                    return
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        walk(item, f"{prefix}[{index}]", path_key)
                    return
                if isinstance(value, str) and path_key and path_key.endswith(("_path", "_root", "_roots")):
                    path = Path(value)
                    output_path = path_key.endswith("runs_root") or path_key.endswith("artifact_root")
                    exists = path.parent.exists() if output_path else path.exists()
                    path_checks.append(
                        {
                            "key": prefix,
                            "path": str(path),
                            "passed": bool(exists),
                            "existence_rule": "parent" if output_path else "path",
                        }
                    )

            walk(config)
            config_results.append(
                {
                    "config": str(config_path.resolve()),
                    "loaded": True,
                    "path_checks": path_checks,
                    "passed": all(item["passed"] for item in path_checks),
                }
            )
        except (OSError, ConfigError, ValueError) as error:
            config_results.append(
                {
                    "config": str(config_path.resolve()),
                    "loaded": False,
                    "path_checks": [],
                    "passed": False,
                    "error": str(error),
                }
            )

    checks.extend(
        _check(f"config:{Path(result['config']).name}", result["passed"], result.get("error", "paths valid"))
        for result in config_results
    )
    passed = all(item["passed"] for item in checks)
    return {
        "tool": "tsr doctor",
        "tsr_version": __version__,
        "python": sys.version,
        "project_root": str(root),
        "passed": passed,
        "checks": checks,
        "configs": config_results,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Check repository structure, dependencies, and config paths"
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    result = inspect_repository(args.project_root, args.config)
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"TSR repository doctor: {'PASS' if result['passed'] else 'FAIL'}")
        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"[{status}] {check['name']}: {check['detail']}")
    return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsr doctor")
    configure_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
