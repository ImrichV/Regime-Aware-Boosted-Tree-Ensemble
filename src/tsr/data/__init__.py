"""Canonical market-data module."""

from .stooq import StooqArchive, StooqCatalog, classify_member_path

__all__ = ["StooqArchive", "StooqCatalog", "classify_member_path"]
