from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import CANDIDATE_COLUMNS, validate_candidate_mapping


def shard_relative_path(generator_key: str, instrument_id: str) -> Path:
    digest = hashlib.sha256(f"{generator_key}|{instrument_id}".encode("utf-8")).hexdigest()
    return Path("candidate_shards") / digest[:2] / f"{digest}.csv"


def deterministic_csv_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != CANDIDATE_COLUMNS:
        frame = frame.reindex(columns=CANDIDATE_COLUMNS)
    output = io.StringIO(newline="")
    frame.to_csv(
        output,
        index=False,
        columns=CANDIDATE_COLUMNS,
        lineterminator="\n",
        float_format="%.12g",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )
    return output.getvalue().encode("utf-8")


def write_candidate_shard(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_csv_bytes(frame)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidate_shard(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if tuple(frame.columns) != CANDIDATE_COLUMNS:
        raise ValueError("candidate shard schema mismatch")
    for column in ("signal_date", "signal_source_row_number", "signal_segment_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["earliest_entry_date"] = pd.to_numeric(
        frame["earliest_entry_date"].replace("", pd.NA), errors="coerce"
    ).astype("Int64")
    frame["raw_setup_strength"] = pd.to_numeric(
        frame["raw_setup_strength"].replace("", pd.NA), errors="coerce"
    ).astype("Float64")
    for record in frame.to_dict("records"):
        record["earliest_entry_date"] = (
            None if pd.isna(record["earliest_entry_date"]) else int(record["earliest_entry_date"])
        )
        record["raw_setup_strength"] = (
            None if pd.isna(record["raw_setup_strength"]) else float(record["raw_setup_strength"])
        )
        validate_candidate_mapping(record)
    return frame


class CandidateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.schema = json.loads((self.root / "candidate_schema.json").read_text())
        self.manifest = pd.read_csv(self.root / "candidate_manifest.csv")
        required = {
            "generator_key",
            "instrument_id",
            "candidate_count",
            "shard_relative_path",
            "shard_sha256",
        }
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"candidate manifest missing columns: {sorted(missing)}")
        if self.manifest.duplicated(["generator_key", "instrument_id"]).any():
            raise ValueError("duplicate generator/instrument rows in candidate manifest")

    def list_shards(self, *, generator_key: str | None = None) -> pd.DataFrame:
        result = self.manifest
        if generator_key is not None:
            result = result[result["generator_key"] == generator_key]
        return result.copy().reset_index(drop=True)

    def load_shard(
        self,
        generator_key: str,
        instrument_id: str,
        *,
        verify: bool = False,
    ) -> pd.DataFrame:
        matches = self.manifest[
            (self.manifest["generator_key"] == generator_key)
            & (self.manifest["instrument_id"] == instrument_id)
        ]
        if len(matches) != 1:
            raise KeyError(f"candidate shard not found: {generator_key} / {instrument_id}")
        row = matches.iloc[0]
        path = self.root / row["shard_relative_path"]
        if verify and file_sha256(path) != row["shard_sha256"]:
            raise ValueError("candidate shard hash mismatch")
        frame = read_candidate_shard(path)
        if len(frame) != int(row["candidate_count"]):
            raise ValueError("candidate shard row count mismatch")
        return frame

    def iter_candidates(self, *, generator_key: str | None = None, verify: bool = False):
        for row in self.list_shards(generator_key=generator_key).itertuples(index=False):
            frame = self.load_shard(row.generator_key, row.instrument_id, verify=verify)
            for record in frame.to_dict("records"):
                yield record
