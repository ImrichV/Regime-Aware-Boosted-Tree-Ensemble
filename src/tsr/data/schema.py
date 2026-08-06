from __future__ import annotations

from dataclasses import dataclass

EXPECTED_HEADER = (
    "<TICKER>",
    "<PER>",
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
    "<VOL>",
    "<OPENINT>",
)

CANONICAL_BAR_COLUMNS = (
    "instrument_id",
    "ticker",
    "exchange",
    "instrument_class",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source_member",
    "source_row_number",
)


@dataclass(frozen=True)
class InstrumentIdentity:
    instrument_id: str
    ticker: str
    exchange: str
    instrument_class: str
    source_member: str
