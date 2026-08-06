from __future__ import annotations

import json
import os
import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import config_hash


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def source_tree_fingerprint(root: Path) -> str | None:
    source_root = root / "src"
    if not source_root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(root: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


@dataclass
class RunContext:
    module_name: str
    module_version: str
    config: dict[str, Any]
    runs_root: Path
    run_dir: Path | None = None
    run_id: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        module_name: str,
        module_version: str,
        config: dict[str, Any],
        runs_root: str | Path,
        resume_dir: str | Path | None = None,
    ) -> "RunContext":
        root = Path(runs_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        cfg_hash = config_hash(config)
        if resume_dir:
            run_dir = Path(resume_dir).expanduser().resolve()
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Resume manifest not found: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("module_name") != module_name:
                raise ValueError("Resume directory belongs to a different module.")
            if manifest.get("config_hash") != cfg_hash:
                raise ValueError("Configuration hash differs from the run being resumed.")
            context = cls(
                module_name=module_name,
                module_version=module_version,
                config=config,
                runs_root=root,
                run_dir=run_dir,
                run_id=manifest["run_id"],
                manifest=manifest,
            )
            context.update(status="running", resumed_at=utc_now())
            return context

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{module_name}_{module_version}_{timestamp}_{cfg_hash}"
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        working_directory = Path.cwd().resolve()
        manifest = {
            "run_id": run_id,
            "module_name": module_name,
            "module_version": module_version,
            "config_hash": cfg_hash,
            "config_path": config.get("_config_path"),
            "status": "created",
            "created_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "working_directory": str(working_directory),
            "source_tree_sha256": source_tree_fingerprint(working_directory),
            "package_version": __version__,
            "git_commit": git_commit(working_directory),
            "git_dirty": git_dirty(working_directory),
            "outputs": {},
            "progress": {},
        }
        context = cls(
            module_name=module_name,
            module_version=module_version,
            config=config,
            runs_root=root,
            run_dir=run_dir,
            run_id=run_id,
            manifest=manifest,
        )
        context.update(status="running", started_at=utc_now())
        return context

    @property
    def manifest_path(self) -> Path:
        assert self.run_dir is not None
        return self.run_dir / "run_manifest.json"

    @property
    def log_path(self) -> Path:
        assert self.run_dir is not None
        return self.run_dir / "events.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        assert self.run_dir is not None
        return self.run_dir / "checkpoint.json"

    def update(self, **changes: Any) -> None:
        self.manifest.update(changes)
        atomic_json(self.manifest_path, self.manifest)

    def set_progress(self, **progress: Any) -> None:
        self.manifest.setdefault("progress", {}).update(progress)
        self.manifest["updated_at"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)

    def record_output(self, name: str, path: str | Path) -> None:
        self.manifest.setdefault("outputs", {})[name] = str(Path(path).resolve())
        atomic_json(self.manifest_path, self.manifest)

    def log(self, event: str, level: str = "INFO", **fields: Any) -> None:
        payload = {"timestamp": utc_now(), "level": level, "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def save_checkpoint(self, payload: dict[str, Any]) -> None:
        atomic_json(self.checkpoint_path, {"saved_at": utc_now(), **payload})

    def load_checkpoint(self) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def complete(self, **fields: Any) -> None:
        self.update(status="completed", completed_at=utc_now(), **fields)
        self.log("run_completed", **fields)


    def interrupt(self, reason: str) -> None:
        self.update(status="interrupted", interrupted_at=utc_now(), interruption_reason=reason)
        self.log("run_interrupted", level="WARNING", reason=reason)

    def fail(self, error: BaseException) -> None:
        self.update(
            status="failed",
            failed_at=utc_now(),
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self.log(
            "run_failed",
            level="ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
        )
