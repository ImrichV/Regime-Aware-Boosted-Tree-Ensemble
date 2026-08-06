from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import RESEARCH_COLUMNS


def shard_relative_path(instrument_id: str) -> Path:
    digest = hashlib.sha256(instrument_id.encode("utf-8")).hexdigest()
    return Path("research_shards") / digest[:2] / f"{digest}.csv.gz"


def deterministic_csv_bytes(frame: pd.DataFrame) -> bytes:
    ordered = frame.reindex(columns=RESEARCH_COLUMNS)
    text = io.StringIO(newline="")
    ordered.to_csv(
        text,
        index=False,
        columns=RESEARCH_COLUMNS,
        lineterminator="\n",
        float_format="%.12g",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )
    return gzip.compress(text.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def write_research_shard(path: Path, frame: pd.DataFrame) -> str:
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


def read_research_shard(path: Path, *, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", usecols=usecols, low_memory=False)
    return frame


class ResearchStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = pd.read_csv(self.root / "research_manifest.csv")
        required = {
            "instrument_id",
            "research_count",
            "shard_relative_path",
            "shard_sha256",
        }
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"research manifest missing columns: {sorted(missing)}")
        if self.manifest["instrument_id"].duplicated().any():
            raise ValueError("duplicate instrument rows in research manifest")

    def load_instrument(
        self,
        instrument_id: str,
        *,
        verify: bool = False,
        usecols: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        matches = self.manifest[self.manifest["instrument_id"] == instrument_id]
        if len(matches) != 1:
            raise KeyError(f"research shard not found: {instrument_id}")
        row = matches.iloc[0]
        path = self.root / row["shard_relative_path"]
        if verify and file_sha256(path) != row["shard_sha256"]:
            raise ValueError("research shard hash mismatch")
        frame = read_research_shard(path, usecols=usecols)
        if len(frame) != int(row["research_count"]):
            raise ValueError("research shard row-count mismatch")
        return frame

    def iter_frames(
        self,
        *,
        verify: bool = False,
        usecols: Iterable[str] | None = None,
    ):
        for row in self.manifest.itertuples(index=False):
            yield self.load_instrument(row.instrument_id, verify=verify, usecols=usecols)
