"""Unit tests for `deep_research.storage.param_codec`."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID

import pytest

from deep_research.storage.param_codec import encoded


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class TestEncode:
    def test_none(self) -> None:
        assert encoded(None) == (None, None)

    def test_bool_true(self) -> None:
        assert encoded(True) == ("true", "BOOLEAN")

    def test_bool_false(self) -> None:
        assert encoded(False) == ("false", "BOOLEAN")

    def test_int(self) -> None:
        assert encoded(42) == ("42", "BIGINT")

    def test_bool_not_int(self) -> None:
        # Critical: bool is a subclass of int; must match BOOLEAN first.
        assert encoded(True) != ("1", "BIGINT")

    def test_float(self) -> None:
        assert encoded(3.14) == ("3.14", "DOUBLE")

    def test_decimal(self) -> None:
        assert encoded(Decimal("1.23")) == ("1.23", "DECIMAL")

    def test_date(self) -> None:
        v, t = encoded(date(2025, 1, 2))
        assert v == "2025-01-02"
        assert t == "DATE"

    def test_naive_datetime_gets_utc(self) -> None:
        v, t = encoded(datetime(2025, 1, 2, 3, 4, 5))
        assert v.endswith("+00:00")
        assert t == "TIMESTAMP"

    def test_aware_datetime_preserves_tz(self) -> None:
        v, t = encoded(datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        assert "+00:00" in v
        assert t == "TIMESTAMP"

    def test_uuid(self) -> None:
        u = UUID("12345678-1234-5678-1234-567812345678")
        assert encoded(u) == (str(u), "STRING")

    def test_enum(self) -> None:
        assert encoded(_Color.RED) == ("red", "STRING")

    def test_bytes(self) -> None:
        v, t = encoded(b"\xde\xad\xbe\xef")
        assert v == "deadbeef"
        assert t == "BINARY"

    def test_dict_is_json(self) -> None:
        v, t = encoded({"a": 1, "b": 2})
        assert t == "STRING"
        assert json.loads(v) == {"a": 1, "b": 2}

    def test_list_is_json(self) -> None:
        v, t = encoded([1, 2, 3])
        assert t == "STRING"
        assert json.loads(v) == [1, 2, 3]

    def test_nested_uuid_serializes(self) -> None:
        u = UUID("12345678-1234-5678-1234-567812345678")
        v, _ = encoded({"id": u})
        assert json.loads(v) == {"id": str(u)}

    def test_nested_datetime_serializes(self) -> None:
        dt = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        v, _ = encoded({"ts": dt})
        assert json.loads(v)["ts"] == dt.isoformat()

    def test_nested_decimal_serializes_as_string(self) -> None:
        v, _ = encoded({"amount": Decimal("1.23")})
        assert json.loads(v) == {"amount": "1.23"}

    def test_fallback_str_coercion(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "weird"

        assert encoded(Weird()) == ("weird", "STRING")


@pytest.mark.skipif(
    True, reason="Requires databricks-sdk; covered by integration tests."
)
def test_to_param_requires_sdk_placeholder() -> None:  # pragma: no cover
    """Keeps the module testable without the SDK.

    The live `to_param` helper is exercised by the SQL-warehouse integration
    tests when `databricks-sdk` is installed.
    """
