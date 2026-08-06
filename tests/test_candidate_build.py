from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsr.candidates.build import build, publish, verify
from tsr.features.engine import compute_features
from tsr.features.schema import payload as feature_schema_payload
from tsr.features.storage import shard_rel, write_npz


def _write_stooq_member(archive: zipfile.ZipFile, member: str, ticker: str, rows: int) -> None:
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    dates = pd.bdate_range("2020-01-01", periods=rows)
    lines = [header]
    for index, date in enumerate(dates):
        close = 100.0 + index * 0.25
        lines.append(
            f"{ticker}.US,D,{date:%Y%m%d},000000,{close-0.1:.4f},{close+0.5:.4f},"
            f"{close-0.5:.4f},{close:.4f},{1000000+index},0\n"
        )
    archive.writestr(member, "".join(lines))


def _make_upstream(tmp_path: Path, instrument_count: int = 3) -> dict:
    archive_path = tmp_path / "stooq.zip"
    symbol_rows = []
    feature_rows = []
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    (feature_root / "feature_schema.json").write_text(json.dumps(feature_schema_payload()))
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index in range(instrument_count):
            ticker = f"T{index:02d}"
            member = f"data/daily/us/nasdaq stocks/1/{ticker.lower()}.us.txt"
            _write_stooq_member(archive, member, ticker, 220 + index)
            instrument_id = f"US:NASDAQ:STOCKS:{ticker}"
            symbol_rows.append(
                {
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "exchange": "NASDAQ",
                    "instrument_class": "stocks",
                    "source_member": member,
                    "valid_row_count": 220 + index,
                    "invalid_row_count": 0,
                    "first_date": "2020-01-01",
                    "last_date": str(pd.bdate_range("2020-01-01", periods=220 + index)[-1].date()),
                }
            )
    symbol_manifest = tmp_path / "symbol_manifest.csv"
    pd.DataFrame(symbol_rows).to_csv(symbol_manifest, index=False)

    # Build accepted-looking Module 02 shards from the same mini archive through the real parser.
    from tsr.data.stooq import StooqCatalog

    catalog = StooqCatalog(archive_path, symbol_manifest)
    for identity in pd.DataFrame(symbol_rows).itertuples(index=False):
        bars = catalog.load_instrument(identity.instrument_id)
        arrays = compute_features(bars).arrays
        relative = shard_rel(identity.instrument_id)
        digest = write_npz(feature_root / relative, arrays)
        feature_rows.append(
            {
                "instrument_id": identity.instrument_id,
                "ticker": identity.ticker,
                "exchange": identity.exchange,
                "instrument_class": identity.instrument_class,
                "row_count": len(bars),
                "shard_relative_path": str(relative.as_posix()),
                "shard_sha256": digest,
            }
        )
    pd.DataFrame(feature_rows).to_csv(feature_root / "feature_manifest.csv", index=False)
    data_fp = tmp_path / "data_fingerprint.json"
    data_fp.write_text(json.dumps({"dataset_fingerprint_sha256": "d" * 64}))
    feature_fp = tmp_path / "feature_fingerprint.json"
    feature_fp.write_text(
        json.dumps(
            {
                "feature_dataset_fingerprint_sha256": "f" * 64,
                "schema_sha256": feature_schema_payload()["schema_sha256"],
            }
        )
    )
    return {
        "archive_path": str(archive_path),
        "symbol_manifest_path": str(symbol_manifest),
        "data_dataset_fingerprint_path": str(data_fp),
        "feature_root": str(feature_root),
        "feature_dataset_fingerprint_path": str(feature_fp),
    }


def _config(tmp_path: Path, upstream: dict) -> dict:
    return {
        "module": {"name": "module_03_candidate_framework", "version": "v1.0.0"},
        "upstream": upstream,
        "generators": [
            {
                "class": "tsr.candidates.testing:FrameworkProbeGenerator",
                "config": {"modulus": 50, "offset": 0},
            }
        ],
        "selection": {"require_rows": True},
        "framework": {"allow_test_generators": True},
        "execution": {"runs_root": str(tmp_path / "runs"), "dry_run_max_files": 2},
    }


def test_candidate_build_real_interfaces_resume_and_verify(tmp_path: Path):
    upstream = _make_upstream(tmp_path)
    config = _config(tmp_path, upstream)
    interrupted = build(config, max_new=2)
    interrupted_manifest = json.loads((interrupted / "run_manifest.json").read_text())
    assert interrupted_manifest["status"] == "interrupted"
    resumed = build(config, resume=interrupted)
    assert resumed == interrupted
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["all_pairs_accounted_for"]
    assert summary["completed_pairs"] == 3
    assert summary["candidate_count"] > 0
    assert summary["operational_errors"] == 0
    verification = verify(resumed, tmp_path / "verification.json")
    result = json.loads(verification.read_text())
    assert result["passed"]
    assert result["shards_verified"] == 3


def test_candidate_build_dry_run_limits_instruments(tmp_path: Path):
    upstream = _make_upstream(tmp_path, instrument_count=4)
    config = _config(tmp_path, upstream)
    run = build(config, dry_run=True)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["dry_run"]
    assert summary["instrument_count"] == 2
    assert summary["completed_pairs"] == 2


def test_probe_run_cannot_be_published(tmp_path: Path):
    upstream = _make_upstream(tmp_path, instrument_count=2)
    config = _config(tmp_path, upstream)
    run = build(config)
    with pytest.raises(ValueError, match="test-generator"):
        publish(run, tmp_path / "published")
