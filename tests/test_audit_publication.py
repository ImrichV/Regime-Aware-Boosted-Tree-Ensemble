from __future__ import annotations

import zipfile
from pathlib import Path

from tsr.data.audit import audit_stooq_archive

HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def make_archive(path: Path) -> Path:
    member = "data/daily/us/nasdaq stocks/1/abc.us.txt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            member,
            HEADER + "ABC.US,D,20200102,000000,10,11,9,10.5,1000.5,0\n",
        )
    return path


def config_for(tmp_path: Path, archive: Path) -> dict:
    return {
        "module": {"name": "module_01_test", "version": "v1"},
        "data": {
            "archive_path": str(archive),
            "compute_archive_sha256": True,
            "survivor_window_days": 10,
        },
        "execution": {
            "runs_root": str(tmp_path / "runs"),
            "checkpoint_every_files": 1,
            "progress_every_files": 100,
            "dry_run_max_files": 1,
        },
        "publication": {"artifact_root": str(tmp_path / "accepted_artifacts")},
    }


def test_dry_run_cannot_overwrite_accepted_artifact_root(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "sample.zip")
    config = config_for(tmp_path, archive)
    run_dir = audit_stooq_archive(config, dry_run=True)
    assert not (tmp_path / "accepted_artifacts").exists()
    assert (run_dir / "dry_run_artifacts" / "summary.json").exists()


def test_full_run_publishes_to_configured_artifact_root(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "sample.zip")
    config = config_for(tmp_path, archive)
    audit_stooq_archive(config, dry_run=False)
    assert (tmp_path / "accepted_artifacts" / "summary.json").exists()
