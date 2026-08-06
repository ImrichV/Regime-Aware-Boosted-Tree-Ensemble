from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .schema import GeneratorMetadata


@dataclass(frozen=True)
class CandidateContext:
    instrument_id: str
    ticker: str
    exchange: str
    instrument_class: str
    bars: pd.DataFrame
    features: pd.DataFrame
    feature_schema_version: str
    feature_dataset_sha256: str

    def validate(self) -> None:
        if len(self.bars) != len(self.features):
            raise ValueError("bar/feature row count mismatch")
        required_bars = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_row_number",
        }
        missing_bars = required_bars.difference(self.bars.columns)
        if missing_bars:
            raise ValueError(f"bars missing columns: {sorted(missing_bars)}")
        required_features = {
            "date",
            "source_row_number",
            "segment_id",
            "segment_reset",
            "history_bars",
        }
        missing_features = required_features.difference(self.features.columns)
        if missing_features:
            raise ValueError(f"features missing columns: {sorted(missing_features)}")
        if not self.bars["date"].reset_index(drop=True).equals(
            self.features["date"].reset_index(drop=True)
        ):
            raise ValueError("bar/feature date alignment mismatch")
        if not np.array_equal(
            self.bars["source_row_number"].to_numpy(np.int64),
            self.features["source_row_number"].to_numpy(np.int64),
        ):
            raise ValueError("bar/feature source-row lineage mismatch")
        dates = pd.to_datetime(self.bars["date"])
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("candidate context dates must be unique and increasing")

    def prefix(self, length: int) -> "CandidateContext":
        if length < 0 or length > len(self.bars):
            raise ValueError("invalid prefix length")
        return CandidateContext(
            instrument_id=self.instrument_id,
            ticker=self.ticker,
            exchange=self.exchange,
            instrument_class=self.instrument_class,
            bars=self.bars.iloc[:length].copy(),
            features=self.features.iloc[:length].copy(),
            feature_schema_version=self.feature_schema_version,
            feature_dataset_sha256=self.feature_dataset_sha256,
        )


@dataclass(frozen=True)
class CandidateSignals:
    mask: np.ndarray
    raw_setup_strength: np.ndarray | None = None
    payload_by_row: Mapping[int, Mapping[str, Any]] | None = None

    def validate(self, row_count: int) -> None:
        mask = np.asarray(self.mask)
        if mask.dtype != np.bool_:
            raise ValueError("candidate mask must have bool dtype")
        if mask.ndim != 1 or len(mask) != row_count:
            raise ValueError("candidate mask length mismatch")
        if self.raw_setup_strength is not None:
            strength = np.asarray(self.raw_setup_strength)
            if strength.ndim != 1 or len(strength) != row_count:
                raise ValueError("raw_setup_strength length mismatch")
            selected = strength[mask]
            if np.any(~np.isfinite(selected)):
                raise ValueError("selected raw_setup_strength values must be finite")
        if self.payload_by_row is not None:
            for index in self.payload_by_row:
                if not isinstance(index, int) or index < 0 or index >= row_count:
                    raise ValueError("payload row index out of range")
                if not mask[index]:
                    raise ValueError("payload supplied for a non-candidate row")


class CandidateGenerator(abc.ABC):
    metadata: GeneratorMetadata

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.metadata.validate()

    @abc.abstractmethod
    def generate(self, context: CandidateContext) -> CandidateSignals:
        """Return causal candidate decisions for every row in context."""

    def validate_context(self, context: CandidateContext) -> None:
        context.validate()
        missing = set(self.metadata.required_features).difference(context.features.columns)
        if missing:
            raise ValueError(
                f"generator {self.metadata.generator_key} requires missing features: {sorted(missing)}"
            )


class GeneratorRegistry:
    def __init__(self) -> None:
        self._generators: dict[str, CandidateGenerator] = {}

    def register(self, generator: CandidateGenerator) -> None:
        key = generator.metadata.generator_key
        if key in self._generators:
            raise ValueError(f"duplicate generator key: {key}")
        self._generators[key] = generator

    def list(self) -> tuple[CandidateGenerator, ...]:
        return tuple(self._generators[key] for key in sorted(self._generators))

    def __len__(self) -> int:
        return len(self._generators)
