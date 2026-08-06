from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tsr.data.stooq import StooqArchive, StooqDataError, classify_member_path

HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_classify_nested_stock_path() -> None:
    identity = classify_member_path("data/daily/us/nasdaq stocks/1/aapl.us.txt")
    assert identity is not None
    assert identity.instrument_id == "US:NASDAQ:STOCKS:AAPL"
    assert identity.ticker == "AAPL"
    assert identity.instrument_class == "stocks"


def test_classify_direct_etf_path() -> None:
    identity = classify_member_path("data/daily/us/nasdaq etfs/qqq.us.txt")
    assert identity is not None
    assert identity.instrument_id == "US:NASDAQ:ETFS:QQQ"


def test_unrecognized_path_returns_none() -> None:
    assert classify_member_path("notes/readme.txt") is None


def test_parse_valid_member(tmp_path: Path) -> None:
    member = "data/daily/us/nasdaq stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "valid.zip",
        {
            member: HEADER
            + "ABC.US,D,20200102,000000,10,11,9,10.5,1000,0\n"
            + "ABC.US,D,20200103,000000,10.5,12,10,11.5,2000,0\n"
        },
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["valid_row_count"] == 2
    assert parsed.summary["invalid_row_count"] == 0
    assert parsed.summary["first_date"] == "2020-01-02"
    assert parsed.summary["last_date"] == "2020-01-03"
    assert parsed.rows[0]["instrument_id"] == "US:NASDAQ:STOCKS:ABC"


def test_invalid_ohlc_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "bad_ohlc.zip",
        {member: HEADER + "ABC.US,D,20200102,000000,10,9,8,10,1000,0\n"},
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["valid_row_count"] == 0
    assert parsed.invalid_rows[0]["reason"] == "high_below_ohlc"


def test_duplicate_date_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "duplicate.zip",
        {
            member: HEADER
            + "ABC.US,D,20200102,000000,10,11,9,10,1000,0\n"
            + "ABC.US,D,20200102,000000,10,11,9,10,1000,0\n"
        },
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["valid_row_count"] == 1
    assert parsed.invalid_rows[0]["reason"] == "duplicate_date"


def test_non_increasing_date_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "descending.zip",
        {
            member: HEADER
            + "ABC.US,D,20200103,000000,10,11,9,10,1000,0\n"
            + "ABC.US,D,20200102,000000,10,11,9,10,1000,0\n"
        },
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["valid_row_count"] == 1
    assert parsed.invalid_rows[0]["reason"] == "non_increasing_date"


def test_ticker_mismatch_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "mismatch.zip",
        {member: HEADER + "XYZ.US,D,20200102,000000,10,11,9,10,1000,0\n"},
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.invalid_rows[0]["reason"] == "ticker_mismatch"


def test_load_member_has_canonical_types(tmp_path: Path) -> None:
    member = "data/daily/us/nasdaq etfs/qqq.us.txt"
    archive_path = make_zip(
        tmp_path / "load.zip",
        {member: HEADER + "QQQ.US,D,20200102,000000,10,11,9,10.5,1000,0\n"},
    )
    frame = StooqArchive(archive_path).load_member(member)
    assert list(frame.columns)[0:5] == [
        "instrument_id",
        "ticker",
        "exchange",
        "instrument_class",
        "date",
    ]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert str(frame["volume"].dtype) == "float64"


def test_strict_load_raises_on_invalid_row(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "strict.zip",
        {member: HEADER + "ABC.US,D,20200102,000000,10,9,8,10,1000,0\n"},
    )
    with pytest.raises(StooqDataError):
        StooqArchive(archive_path).load_member(member, invalid_policy="raise")


def test_empty_member_is_recorded(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/empty.us.txt"
    archive_path = make_zip(tmp_path / "empty.zip", {member: ""})
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["empty_file"] is True
    assert parsed.summary["row_count"] == 0


def test_invalid_calendar_date_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "date.zip",
        {member: HEADER + "ABC.US,D,20210229,000000,10,11,9,10,1000,0\n"},
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.invalid_rows[0]["reason"] == "invalid_date"


def test_classify_nysemkt_path() -> None:
    identity = classify_member_path("data/daily/us/nysemkt stocks/acu.us.txt")
    assert identity is not None
    assert identity.instrument_id == "US:NYSEMKT:STOCKS:ACU"


def test_windows_reserved_filename_alias() -> None:
    identity = classify_member_path("data/daily/us/nasdaq etfs/_prn.us.txt")
    assert identity is not None
    assert identity.ticker == "PRN"
    assert identity.instrument_id == "US:NASDAQ:ETFS:PRN"


def test_adjusted_fractional_volume_is_valid(tmp_path: Path) -> None:
    member = "data/daily/us/nasdaq etfs/aadr.us.txt"
    archive_path = make_zip(
        tmp_path / "fractional_volume.zip",
        {member: HEADER + "AADR.US,D,20200102,000000,10,11,9,10.5,8153.6931763407,0\n"},
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.summary["valid_row_count"] == 1
    assert parsed.rows[0]["volume"] == pytest.approx(8153.6931763407)


def test_catalog_loads_by_instrument_id(tmp_path: Path) -> None:
    from tsr.data.stooq import StooqCatalog

    member = "data/daily/us/nasdaq stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "catalog.zip",
        {member: HEADER + "ABC.US,D,20200102,000000,10,11,9,10.5,1000.5,0\n"},
    )
    manifest = tmp_path / "symbol_manifest.csv"
    pd.DataFrame(
        [
            {
                "instrument_id": "US:NASDAQ:STOCKS:ABC",
                "ticker": "ABC",
                "exchange": "NASDAQ",
                "instrument_class": "stocks",
                "source_member": member,
                "valid_row_count": 1,
                "invalid_row_count": 0,
                "first_date": "2020-01-02",
                "last_date": "2020-01-02",
            }
        ]
    ).to_csv(manifest, index=False)
    catalog = StooqCatalog(archive_path, manifest)
    frame = catalog.load_instrument("US:NASDAQ:STOCKS:ABC")
    assert len(frame) == 1
    assert frame.iloc[0]["volume"] == pytest.approx(1000.5)


def test_catalog_rejects_duplicate_instrument_ids(tmp_path: Path) -> None:
    from tsr.data.stooq import StooqCatalog, StooqDataError

    member = "data/daily/us/nasdaq stocks/1/abc.us.txt"
    archive_path = make_zip(tmp_path / "catalog_dup.zip", {member: HEADER})
    manifest = tmp_path / "manifest_dup.csv"
    row = {
        "instrument_id": "US:NASDAQ:STOCKS:ABC",
        "ticker": "ABC",
        "exchange": "NASDAQ",
        "instrument_class": "stocks",
        "source_member": member,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "first_date": None,
        "last_date": None,
    }
    pd.DataFrame([row, {**row, "source_member": "other"}]).to_csv(manifest, index=False)
    with pytest.raises(StooqDataError, match="Duplicate instrument_id"):
        StooqCatalog(archive_path, manifest)


def test_non_finite_volume_is_quarantined(tmp_path: Path) -> None:
    member = "data/daily/us/nyse stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "nan_volume.zip",
        {member: HEADER + "ABC.US,D,20200102,000000,10,11,9,10,nan,0\n"},
    )
    parsed = StooqArchive(archive_path).parse_member(member)
    assert parsed.invalid_rows[0]["reason"] == "non_finite_volume"


def test_open_archive_handle_survives_source_path_replacement(tmp_path: Path) -> None:
    member = "data/daily/us/nasdaq stocks/1/abc.us.txt"
    archive_path = make_zip(
        tmp_path / "persistent.zip",
        {member: HEADER + "ABC.US,D,20200102,000000,10,11,9,10.5,1000,0\n"},
    )
    archive = StooqArchive(archive_path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"not a zip")
    replacement.replace(archive_path)
    parsed = archive.parse_member(member)
    assert parsed.summary["valid_row_count"] == 1
    archive.close()
    with pytest.raises(RuntimeError, match="closed"):
        archive.members()
