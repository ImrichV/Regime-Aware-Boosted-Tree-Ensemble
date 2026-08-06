from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .schema import CANONICAL_BAR_COLUMNS, EXPECTED_HEADER, InstrumentIdentity

_MEMBER_RE = re.compile(
    r"^data/daily/us/(?P<exchange>nasdaq|nyse|nysemkt) (?P<class_name>stocks|etfs)"
    r"(?:/[123])?/(?P<filename>[^/]+\.us\.txt)$",
    re.IGNORECASE,
)


class StooqDataError(ValueError):
    """Raised for a malformed Stooq member or row."""


@dataclass
class ParsedMember:
    identity: InstrumentIdentity
    rows: list[dict[str, Any]]
    invalid_rows: list[dict[str, Any]]
    summary: dict[str, Any]


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _canonical_filename_ticker(filename: str) -> str:
    ticker = filename[: -len(".us.txt")].upper()
    # Stooq prefixes Windows-reserved filenames (for example PRN) with an underscore.
    if ticker.startswith("_") and ticker[1:] in _WINDOWS_RESERVED_NAMES:
        ticker = ticker[1:]
    return ticker


def classify_member_path(member: str) -> InstrumentIdentity | None:
    match = _MEMBER_RE.match(member)
    if not match:
        return None
    filename = match.group("filename")
    ticker = _canonical_filename_ticker(filename)
    exchange = match.group("exchange").upper()
    instrument_class = match.group("class_name").lower()
    instrument_id = f"US:{exchange}:{instrument_class.upper()}:{ticker}"
    return InstrumentIdentity(
        instrument_id=instrument_id,
        ticker=ticker,
        exchange=exchange,
        instrument_class=instrument_class,
        source_member=member,
    )


def _valid_yyyymmdd(value: int) -> bool:
    year = value // 10000
    month = (value // 100) % 100
    day = value % 100
    if year < 1800 or year > 2200 or month < 1 or month > 12 or day < 1:
        return False
    month_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    max_day = month_days[month - 1]
    if month == 2 and (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)):
        max_day = 29
    return day <= max_day


def yyyymmdd_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value // 10000:04d}-{(value // 100) % 100:02d}-{value % 100:02d}"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class StooqArchive:
    """Read and validate the supplied Stooq daily-data ZIP without extracting it."""

    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path).expanduser().resolve()
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Stooq archive does not exist: {self.archive_path}")
        try:
            # Keep one validated read-only ZIP handle for the catalog lifetime. Reopening
            # a 477 MB central directory for every symbol is both slow and vulnerable to
            # transient mounted-path replacement errors. ZipExtFile objects are still
            # opened and closed per member, so no member stream leaks across calls.
            self._archive = zipfile.ZipFile(self.archive_path, mode="r")
            self._archive.infolist()  # force central-directory validation now
        except (OSError, zipfile.BadZipFile) as exc:
            raise StooqDataError(f"Not a valid ZIP archive: {self.archive_path}") from exc

    def close(self) -> None:
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()
            self._archive = None

    def __enter__(self) -> "StooqArchive":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass

    def _require_archive(self) -> zipfile.ZipFile:
        archive = getattr(self, "_archive", None)
        if archive is None:
            raise RuntimeError("StooqArchive is closed")
        return archive

    def members(self) -> list[str]:
        archive = self._require_archive()
        return sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".txt")
        )

    def classified_members(self) -> list[tuple[str, InstrumentIdentity]]:
        result: list[tuple[str, InstrumentIdentity]] = []
        for member in self.members():
            identity = classify_member_path(member)
            if identity is not None:
                result.append((member, identity))
        return result

    def parse_member(
        self,
        member: str,
        *,
        keep_rows: bool = True,
        max_invalid_raw_chars: int = 300,
    ) -> ParsedMember:
        identity = classify_member_path(member)
        if identity is None:
            raise StooqDataError(f"Unrecognized Stooq member path: {member}")

        valid_rows: list[dict[str, Any]] = []
        invalid_rows: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        first_date: int | None = None
        last_date: int | None = None
        previous_date: int | None = None
        previous_close: float | None = None
        row_count = 0
        valid_count = 0
        zero_volume_count = 0
        extreme_return_50_count = 0
        extreme_return_100_count = 0
        max_abs_return = 0.0
        min_close = math.inf
        max_close = -math.inf
        header_valid = False

        archive = self._require_archive()
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise FileNotFoundError(f"ZIP member not found: {member}") from exc

        with archive.open(info, "r") as raw_handle:
                header_raw = raw_handle.readline()
                if not header_raw:
                    return ParsedMember(
                        identity=identity,
                        rows=[],
                        invalid_rows=[],
                        summary={
                            "source_member": member,
                            "instrument_id": identity.instrument_id,
                            "ticker": identity.ticker,
                            "exchange": identity.exchange,
                            "instrument_class": identity.instrument_class,
                            "compressed_size": info.compress_size,
                            "uncompressed_size": info.file_size,
                            "crc32": f"{info.CRC:08x}",
                            "header_valid": False,
                            "empty_file": True,
                            "row_count": 0,
                            "valid_row_count": 0,
                            "invalid_row_count": 0,
                            "invalid_reason_counts": {},
                            "first_date": None,
                            "last_date": None,
                            "zero_volume_count": 0,
                            "extreme_return_50_count": 0,
                            "extreme_return_100_count": 0,
                            "max_abs_return": None,
                            "min_close": None,
                            "max_close": None,
                        },
                    )

                try:
                    header_text = header_raw.decode("utf-8-sig").strip()
                except UnicodeDecodeError:
                    header_text = ""
                header = tuple(part.strip() for part in header_text.split(","))
                header_valid = header == EXPECTED_HEADER
                if not header_valid:
                    reasons["invalid_header"] += 1

                for source_row_number, raw_line in enumerate(raw_handle, start=2):
                    if not raw_line.strip():
                        continue
                    row_count += 1
                    reason: str | None = None
                    date_value: int | None = None
                    raw_preview = raw_line[:max_invalid_raw_chars].decode("utf-8", "replace").strip()
                    columns = raw_line.rstrip(b"\r\n").split(b",")
                    if not header_valid:
                        reason = "invalid_header"
                    elif len(columns) != 10:
                        reason = "wrong_column_count"
                    else:
                        try:
                            row_ticker = columns[0].decode("ascii").upper()
                            period = columns[1].decode("ascii").upper()
                            date_value = int(columns[2])
                            time_value = columns[3].decode("ascii")
                            open_value = float(columns[4])
                            high_value = float(columns[5])
                            low_value = float(columns[6])
                            close_value = float(columns[7])
                            volume_value = float(columns[8])
                            int(columns[9])  # parsed only to verify the field is numeric
                        except (UnicodeDecodeError, ValueError, OverflowError):
                            reason = "parse_error"
                        else:
                            expected_ticker = f"{identity.ticker}.US"
                            if row_ticker != expected_ticker:
                                reason = "ticker_mismatch"
                            elif period != "D":
                                reason = "non_daily_period"
                            elif time_value != "000000":
                                reason = "unexpected_time"
                            elif not _valid_yyyymmdd(date_value):
                                reason = "invalid_date"
                            elif not all(
                                math.isfinite(value)
                                for value in (open_value, high_value, low_value, close_value)
                            ):
                                reason = "non_finite_ohlc"
                            elif min(open_value, high_value, low_value, close_value) <= 0:
                                reason = "non_positive_ohlc"
                            elif not math.isfinite(volume_value):
                                reason = "non_finite_volume"
                            elif volume_value < 0:
                                reason = "negative_volume"
                            elif high_value < max(open_value, low_value, close_value):
                                reason = "high_below_ohlc"
                            elif low_value > min(open_value, high_value, close_value):
                                reason = "low_above_ohlc"
                            elif previous_date is not None and date_value == previous_date:
                                reason = "duplicate_date"
                            elif previous_date is not None and date_value < previous_date:
                                reason = "non_increasing_date"

                    if reason is not None:
                        reasons[reason] += 1
                        invalid_rows.append(
                            {
                                "source_member": member,
                                "instrument_id": identity.instrument_id,
                                "ticker": identity.ticker,
                                "source_row_number": source_row_number,
                                "date_raw": date_value,
                                "reason": reason,
                                "raw_preview": raw_preview,
                            }
                        )
                        continue

                    valid_count += 1
                    if first_date is None:
                        first_date = date_value
                    last_date = date_value
                    previous_date = date_value
                    if volume_value == 0:
                        zero_volume_count += 1
                    min_close = min(min_close, close_value)
                    max_close = max(max_close, close_value)
                    if previous_close is not None:
                        abs_return = abs(close_value / previous_close - 1.0)
                        max_abs_return = max(max_abs_return, abs_return)
                        if abs_return > 0.50:
                            extreme_return_50_count += 1
                        if abs_return > 1.00:
                            extreme_return_100_count += 1
                    previous_close = close_value

                    if keep_rows:
                        valid_rows.append(
                            {
                                "instrument_id": identity.instrument_id,
                                "ticker": identity.ticker,
                                "exchange": identity.exchange,
                                "instrument_class": identity.instrument_class,
                                "date": yyyymmdd_to_iso(date_value),
                                "open": open_value,
                                "high": high_value,
                                "low": low_value,
                                "close": close_value,
                                "volume": volume_value,
                                "source_member": member,
                                "source_row_number": source_row_number,
                            }
                        )

        summary = {
            "source_member": member,
            "instrument_id": identity.instrument_id,
            "ticker": identity.ticker,
            "exchange": identity.exchange,
            "instrument_class": identity.instrument_class,
            "compressed_size": info.compress_size,
            "uncompressed_size": info.file_size,
            "crc32": f"{info.CRC:08x}",
            "header_valid": header_valid,
            "empty_file": row_count == 0,
            "row_count": row_count,
            "valid_row_count": valid_count,
            "invalid_row_count": len(invalid_rows),
            "invalid_reason_counts": dict(sorted(reasons.items())),
            "first_date": yyyymmdd_to_iso(first_date),
            "last_date": yyyymmdd_to_iso(last_date),
            "zero_volume_count": zero_volume_count,
            "extreme_return_50_count": extreme_return_50_count,
            "extreme_return_100_count": extreme_return_100_count,
            "max_abs_return": max_abs_return if valid_count > 1 else None,
            "min_close": min_close if valid_count else None,
            "max_close": max_close if valid_count else None,
        }
        return ParsedMember(identity, valid_rows, invalid_rows, summary)

    def load_member(self, member: str, invalid_policy: str = "drop") -> pd.DataFrame:
        parsed = self.parse_member(member, keep_rows=True)
        if invalid_policy not in {"drop", "raise"}:
            raise ValueError("invalid_policy must be 'drop' or 'raise'.")
        if invalid_policy == "raise" and parsed.invalid_rows:
            first = parsed.invalid_rows[0]
            raise StooqDataError(
                f"Invalid row in {member} at source row {first['source_row_number']}: "
                f"{first['reason']}"
            )
        frame = pd.DataFrame(parsed.rows, columns=CANONICAL_BAR_COLUMNS)
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
            frame = frame.astype(
                {
                    "instrument_id": "string",
                    "ticker": "string",
                    "exchange": "category",
                    "instrument_class": "category",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                    "source_member": "string",
                    "source_row_number": "int64",
                }
            )
        return frame


class StooqCatalog:
    """Stable downstream interface backed by an accepted Module 01 manifest."""

    def __init__(
        self,
        archive_path: str | Path,
        symbol_manifest_path: str | Path,
        *,
        invalid_policy: str = "drop",
    ) -> None:
        if invalid_policy not in {"drop", "raise"}:
            raise ValueError("invalid_policy must be 'drop' or 'raise'.")
        self.archive = StooqArchive(archive_path)
        self.symbol_manifest_path = Path(symbol_manifest_path).expanduser().resolve()
        if not self.symbol_manifest_path.exists():
            raise FileNotFoundError(f"Symbol manifest not found: {self.symbol_manifest_path}")
        self.invalid_policy = invalid_policy
        self.manifest = pd.read_csv(self.symbol_manifest_path)
        required = {
            "instrument_id",
            "ticker",
            "exchange",
            "instrument_class",
            "source_member",
            "valid_row_count",
            "invalid_row_count",
            "first_date",
            "last_date",
        }
        missing = required.difference(self.manifest.columns)
        if missing:
            raise StooqDataError(f"Symbol manifest is missing columns: {sorted(missing)}")
        if self.manifest["instrument_id"].duplicated().any():
            duplicated = self.manifest.loc[
                self.manifest["instrument_id"].duplicated(keep=False), "instrument_id"
            ].tolist()
            raise StooqDataError(f"Duplicate instrument_id values in manifest: {duplicated[:10]}")
        if self.manifest["source_member"].duplicated().any():
            raise StooqDataError("Duplicate source_member values in symbol manifest.")
        self._member_by_instrument = dict(
            zip(self.manifest["instrument_id"], self.manifest["source_member"], strict=True)
        )

    def list_instruments(
        self,
        *,
        exchange: str | None = None,
        instrument_class: str | None = None,
        require_rows: bool = True,
    ) -> pd.DataFrame:
        result = self.manifest
        if exchange is not None:
            result = result[result["exchange"].str.upper() == exchange.upper()]
        if instrument_class is not None:
            result = result[
                result["instrument_class"].str.lower() == instrument_class.lower()
            ]
        if require_rows:
            result = result[result["valid_row_count"] > 0]
        return result.copy().reset_index(drop=True)

    def member_for(self, instrument_id: str) -> str:
        try:
            return self._member_by_instrument[instrument_id]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument_id: {instrument_id}") from exc

    def load_instrument(self, instrument_id: str) -> pd.DataFrame:
        return self.archive.load_member(
            self.member_for(instrument_id), invalid_policy=self.invalid_policy
        )

    def iter_instruments(
        self,
        *,
        exchange: str | None = None,
        instrument_class: str | None = None,
        require_rows: bool = True,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        universe = self.list_instruments(
            exchange=exchange,
            instrument_class=instrument_class,
            require_rows=require_rows,
        )
        for instrument_id in universe["instrument_id"]:
            yield instrument_id, self.load_instrument(instrument_id)
