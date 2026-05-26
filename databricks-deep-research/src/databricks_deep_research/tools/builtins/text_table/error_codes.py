from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ErrorCode(StrEnum):
    DISCOVERY_UNAVAILABLE = "discovery_unavailable"
    SCHEMA_STALE = "schema_stale"
    INFERENCE_FAILED = "inference_failed"
    INVALID_COLUMN = "invalid_column"
    INVALID_FILTER = "invalid_filter"
    BUDGET_EXCEEDED = "budget_exceeded"
    ROW_LIMIT_EXCEEDED = "row_limit_exceeded"
    PAYLOAD_LIMIT_EXCEEDED = "payload_limit_exceeded"
    GROUP_CARDINALITY_EXCEEDED = "group_cardinality_exceeded"
    NEIGHBOR_CONFIG_MISSING = "neighbor_config_missing"
    DEPRECATED_TOOL_KIND = "deprecated_tool_kind"
    DUPLICATE_BINDING = "duplicate_binding"
    INFERENCE_LOW_CONFIDENCE = "inference_low_confidence"
    INVALID_BINDING = "invalid_binding"
    SCHEMA_FETCH_FAILED = "schema_fetch_failed"


@dataclass(frozen=True)
class ToolError:
    error_code: ErrorCode
    message: str
    binding: str | None = None
    hint: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "binding": self.binding,
            "hint": self.hint,
            "details": dict(self.details),
        }


class ToolErrorException(Exception):
    def __init__(self, error: ToolError) -> None:
        super().__init__(f"{error.error_code}: {error.message}")
        self.error = error

    def __str__(self) -> str:
        return f"{self.error.error_code}: {self.error.message}"
