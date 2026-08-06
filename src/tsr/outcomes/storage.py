from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

from .schema import OUTCOME_COLUMNS, validate_outcome_mapping


def shard_relative_path(instrument_id: str) -> Path:
    digest = hashlib.sha256(instrument_id.encode("utf-8")).hexdigest()
    return Path("outcome_shards") / digest[:2] / f"{digest}.csv"


def deterministic_csv_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != OUTCOME_COLUMNS:
        frame = frame.reindex(columns=OUTCOME_COLUMNS)
    output = io.StringIO(newline="")
    frame.to_csv(
        output,
        index=False,
        columns=OUTCOME_COLUMNS,
        lineterminator="\n",
        float_format="%.12g",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )
    return output.getvalue().encode("utf-8")


def write_outcome_shard(path: Path, frame: pd.DataFrame) -> str:
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


def read_outcome_shard(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if tuple(frame.columns) != OUTCOME_COLUMNS:
        raise ValueError("outcome shard schema mismatch")
    nullable_ints = [
        column
        for column in frame.columns
        if column.endswith("_date")
        or column.endswith("_bar")
        or column.endswith("_bars_to_event")
        or column in {"entry_source_row_number", "entry_segment_id"}
    ]
    for column in nullable_ints:
        frame[column] = pd.to_numeric(frame[column].replace("", pd.NA), errors="coerce").astype("Int64")
    frame["signal_date"] = pd.to_numeric(frame["signal_date"], errors="raise").astype("int64")
    frame["same_segment_bars_available"] = pd.to_numeric(
        frame["same_segment_bars_available"], errors="raise"
    ).astype("int64")
    float_columns = [
        column
        for column in frame.columns
        if column in {"signal_close", "signal_atr_14_abs", "entry_open", "entry_gap_return_raw"}
        or "directional_" in column
    ]
    for column in float_columns:
        frame[column] = pd.to_numeric(frame[column].replace("", pd.NA), errors="coerce").astype("Float64")
    for record in frame.to_dict("records"):
        for column, value in list(record.items()):
            if pd.isna(value):
                record[column] = None
        validate_outcome_mapping(record)
    return frame


class OutcomeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.schema = json.loads((self.root / "outcome_schema.json").read_text())
        self.manifest = pd.read_csv(self.root / "outcome_manifest.csv")
        required = {
            "instrument_id",
            "candidate_count",
            "outcome_count",
            "shard_relative_path",
            "shard_sha256",
        }
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"outcome manifest missing columns: {sorted(missing)}")
        if self.manifest["instrument_id"].duplicated().any():
            raise ValueError("duplicate instrument rows in outcome manifest")

    def load_instrument(self, instrument_id: str, *, verify: bool = False) -> pd.DataFrame:
        matches = self.manifest[self.manifest["instrument_id"] == instrument_id]
        if len(matches) != 1:
            raise KeyError(f"outcome shard not found: {instrument_id}")
        row = matches.iloc[0]
        path = self.root / row["shard_relative_path"]
        if verify and file_sha256(path) != row["shard_sha256"]:
            raise ValueError("outcome shard hash mismatch")
        frame = read_outcome_shard(path)
        if len(frame) != int(row["outcome_count"]):
            raise ValueError("outcome shard row-count mismatch")
        return frame

    def iter_outcomes(self, *, verify: bool = False):
        for row in self.manifest.itertuples(index=False):
            frame = self.load_instrument(row.instrument_id, verify=verify)
            for record in frame.to_dict("records"):
                yield record
