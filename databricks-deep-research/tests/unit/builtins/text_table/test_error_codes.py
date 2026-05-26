from __future__ import annotations

from databricks_deep_research.tools.builtins.text_table.error_codes import (
    ErrorCode,
    ToolError,
    ToolErrorException,
)


def test_error_code_values_stable() -> None:
    assert ErrorCode.DISCOVERY_UNAVAILABLE == "discovery_unavailable"
    assert ErrorCode.BUDGET_EXCEEDED == "budget_exceeded"


def test_tool_error_serializes_to_dict() -> None:
    err = ToolError(
        error_code=ErrorCode.INVALID_COLUMN,
        message="column 'foo' not in schema",
        binding="treasury_chunks",
        hint="check schema",
        details={"column": "foo"},
    )
    out = err.to_dict()
    assert out["error_code"] == "invalid_column"
    assert out["binding"] == "treasury_chunks"
    assert out["details"] == {"column": "foo"}


def test_tool_error_exception_carries_payload() -> None:
    err = ToolError(error_code=ErrorCode.BUDGET_EXCEEDED, message="x")
    exc = ToolErrorException(err)
    assert exc.error.error_code == ErrorCode.BUDGET_EXCEEDED
    assert "budget_exceeded" in str(exc)
