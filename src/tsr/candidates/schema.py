from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

MODULE_NAME = "module_03_candidate_framework"
MODULE_VERSION = "v1.0.0"
CANDIDATE_SCHEMA_VERSION = "candidate_schema_v1.0.0"

_DIRECTION_VALUES = {"LONG", "SHORT"}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$")


@dataclass(frozen=True)
class GeneratorMetadata:
    setup_family: str
    setup_version: str
    direction: str
    required_features: tuple[str, ...]
    description: str
    permissive: bool = True
    overlap_policy: str = "allow"
    decision_time: str = "signal_close"
    earliest_entry_rule: str = "next_valid_session_open"

    def validate(self) -> None:
        if not _NAME_RE.fullmatch(self.setup_family):
            raise ValueError(
                "setup_family must use lower-case snake_case and contain 2-64 characters"
            )
        if not _VERSION_RE.fullmatch(self.setup_version):
            raise ValueError("setup_version must be semantic, for example v1.0.0")
        if self.direction not in _DIRECTION_VALUES:
            raise ValueError(f"direction must be one of {sorted(_DIRECTION_VALUES)}")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if self.overlap_policy not in {"allow", "suppress_consecutive", "cooldown"}:
            raise ValueError("unsupported overlap_policy")
        if self.decision_time != "signal_close":
            raise ValueError("Module 03 v1 only permits signal_close decisions")
        if self.earliest_entry_rule != "next_valid_session_open":
            raise ValueError("Module 03 v1 only permits next_valid_session_open")
        if len(set(self.required_features)) != len(self.required_features):
            raise ValueError("required_features contains duplicates")

    @property
    def generator_key(self) -> str:
        return f"{self.setup_family}|{self.setup_version}|{self.direction}"


CANDIDATE_COLUMNS = (
    "candidate_id",
    "candidate_schema_version",
    "instrument_id",
    "ticker",
    "exchange",
    "instrument_class",
    "signal_date",
    "signal_source_row_number",
    "signal_segment_id",
    "setup_family",
    "setup_version",
    "direction",
    "decision_time",
    "earliest_entry_rule",
    "earliest_entry_date",
    "historical_entry_status",
    "raw_setup_strength",
    "generator_payload_json",
    "generator_config_sha256",
    "feature_schema_version",
    "feature_dataset_sha256",
)

STRING_COLUMNS = (
    "candidate_id",
    "candidate_schema_version",
    "instrument_id",
    "ticker",
    "exchange",
    "instrument_class",
    "setup_family",
    "setup_version",
    "direction",
    "decision_time",
    "earliest_entry_rule",
    "historical_entry_status",
    "generator_payload_json",
    "generator_config_sha256",
    "feature_schema_version",
    "feature_dataset_sha256",
)

INT_COLUMNS = ("signal_date", "signal_source_row_number", "signal_segment_id")
NULLABLE_INT_COLUMNS = ("earliest_entry_date",)
FLOAT_COLUMNS = ("raw_setup_strength",)


def canonical_json(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        payload = {}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_id(
    *,
    setup_family: str,
    setup_version: str,
    instrument_id: str,
    signal_date: int,
    direction: str,
) -> str:
    canonical = "|".join(
        (
            CANDIDATE_SCHEMA_VERSION,
            setup_family,
            setup_version,
            instrument_id,
            str(int(signal_date)),
            direction,
        )
    )
    return "cand_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "row_key": ["candidate_id"],
        "deterministic_id_components": [
            "candidate_schema_version",
            "setup_family",
            "setup_version",
            "instrument_id",
            "signal_date",
            "direction",
        ],
        "columns": list(CANDIDATE_COLUMNS),
        "timing": {
            "decision_time": "after signal daily bar closes",
            "earliest_entry": "next valid session open",
            "same_close_entry": "forbidden",
            "candidate_condition_may_use_next_bar": False,
            "earliest_entry_date_is_execution_alignment_metadata": True,
        },
        "duplicates": {
            "duplicate_candidate_id": "fatal",
            "same_ticker_date_across_families": "allowed",
            "consecutive_same_family_signals": "allowed by framework; generator may version a suppression policy",
        },
        "overlap": {
            "framework_default": "retain every qualifying signal",
            "position overlap": "not decided by Module 03",
        },
        "forbidden_fields": [
            "future_return",
            "label",
            "target_hit",
            "stop_hit",
            "exit_date",
            "exit_price",
            "trade_R",
            "model_score",
            "daily_rank",
            "portfolio_decision",
        ],
        "historical_entry_status_values": [
            "available",
            "no_later_bar",
            "next_bar_new_segment",
        ],
        "generator_metadata": asdict(
            GeneratorMetadata(
                setup_family="example_family",
                setup_version="v1.0.0",
                direction="LONG",
                required_features=("return_20",),
                description="Schema example only; not a registered playbook.",
            )
        ),
    }
    payload["schema_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_candidate_mapping(row: Mapping[str, Any]) -> None:
    missing = set(CANDIDATE_COLUMNS).difference(row)
    extra = set(row).difference(CANDIDATE_COLUMNS)
    if missing:
        raise ValueError(f"candidate row missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"candidate row has unknown columns: {sorted(extra)}")

    if row["candidate_schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema version mismatch")
    if row["direction"] not in _DIRECTION_VALUES:
        raise ValueError("invalid direction")
    if row["decision_time"] != "signal_close":
        raise ValueError("invalid decision_time")
    if row["earliest_entry_rule"] != "next_valid_session_open":
        raise ValueError("invalid earliest_entry_rule")
    if row["historical_entry_status"] not in {
        "available",
        "no_later_bar",
        "next_bar_new_segment",
    }:
        raise ValueError("invalid historical_entry_status")
    if row["historical_entry_status"] == "available" and row["earliest_entry_date"] is None:
        raise ValueError("available entry status requires earliest_entry_date")
    if row["historical_entry_status"] != "available" and row["earliest_entry_date"] is not None:
        raise ValueError("unavailable entry status requires null earliest_entry_date")
    if row["earliest_entry_date"] is not None and int(row["earliest_entry_date"]) <= int(row["signal_date"]):
        raise ValueError("earliest_entry_date must be later than signal_date")
    strength = row["raw_setup_strength"]
    if strength is not None and not math.isfinite(float(strength)):
        raise ValueError("raw_setup_strength must be finite or null")
    payload = row["generator_payload_json"]
    decoded = json.loads(payload)
    if canonical_json(decoded) != payload:
        raise ValueError("generator_payload_json is not canonical JSON")
    expected = candidate_id(
        setup_family=str(row["setup_family"]),
        setup_version=str(row["setup_version"]),
        instrument_id=str(row["instrument_id"]),
        signal_date=int(row["signal_date"]),
        direction=str(row["direction"]),
    )
    if row["candidate_id"] != expected:
        raise ValueError("candidate_id does not match deterministic components")
