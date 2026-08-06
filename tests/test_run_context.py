from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsr.run import RunContext


def test_run_context_checkpoint_and_resume(tmp_path: Path) -> None:
    config = {"module": {"name": "x"}, "value": 1}
    context = RunContext.create("module_x", "v1", config, tmp_path / "runs")
    context.save_checkpoint({"next_index": 12})
    context.set_progress(processed=12)

    resumed = RunContext.create(
        "module_x",
        "v1",
        config,
        tmp_path / "runs",
        resume_dir=context.run_dir,
    )
    assert resumed.load_checkpoint() is not None
    assert resumed.load_checkpoint()["next_index"] == 12
    assert resumed.manifest["status"] == "running"


def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    context = RunContext.create(
        "module_x",
        "v1",
        {"value": 1},
        tmp_path / "runs",
    )
    with pytest.raises(ValueError, match="Configuration hash"):
        RunContext.create(
            "module_x",
            "v1",
            {"value": 2},
            tmp_path / "runs",
            resume_dir=context.run_dir,
        )


def test_run_context_failure_is_recorded(tmp_path: Path) -> None:
    context = RunContext.create("module_x", "v1", {"value": 1}, tmp_path / "runs")
    context.fail(RuntimeError("boom"))
    manifest = json.loads(context.manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert "boom" in manifest["error_message"]


def test_run_context_interruption_is_recorded(tmp_path: Path) -> None:
    context = RunContext.create("module_x", "v1", {"value": 1}, tmp_path / "runs")
    context.interrupt("stopped safely")
    manifest = json.loads(context.manifest_path.read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["interruption_reason"] == "stopped safely"
