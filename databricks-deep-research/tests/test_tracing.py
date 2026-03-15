"""Unit tests for the tracing module.

Verifies that trace_span works correctly both when mlflow is available
and when it is not, and that errors never propagate out.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from databricks_deep_research.tracing import trace_span

# ---------------------------------------------------------------------------
# 1. No-op when mlflow is not available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_span_noop_without_mlflow() -> None:
    """trace_span yields None when _HAS_MLFLOW is False."""
    with patch("databricks_deep_research.tracing._HAS_MLFLOW", False):
        async with trace_span("test_span") as span:
            assert span is None


@pytest.mark.asyncio
async def test_trace_span_noop_does_not_break_workflow() -> None:
    """Workflow code runs fine inside a no-op trace_span."""
    with patch("databricks_deep_research.tracing._HAS_MLFLOW", False):
        result = 0
        async with trace_span("test_span"):
            result = 42
        assert result == 42


# ---------------------------------------------------------------------------
# 2. Span creation when mlflow IS available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_span_creates_span_with_mlflow() -> None:
    """trace_span creates and closes a span when mlflow is available."""
    mock_span = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_span)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_mlflow = MagicMock()
    mock_mlflow.start_span.return_value = mock_cm

    mock_span_type = MagicMock()
    mock_span_type.CHAIN = "CHAIN"
    mock_span_type.AGENT = "AGENT"
    mock_span_type.TOOL = "TOOL"

    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", True),
        patch("databricks_deep_research.tracing.mlflow", mock_mlflow),
        patch("databricks_deep_research.tracing.SpanType", mock_span_type),
    ):
        async with trace_span("my_span", span_type="AGENT", attributes={"key": "val"}) as span:
            assert span is mock_span

        # Verify span was opened and closed
        mock_mlflow.start_span.assert_called_once()
        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once_with(None, None, None)
        mock_span.set_attributes.assert_called_once_with({"key": "val"})


# ---------------------------------------------------------------------------
# 3. Errors inside trace_span don't propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_span_creation_error_is_suppressed() -> None:
    """If mlflow.start_span raises, trace_span still yields None."""
    mock_mlflow = MagicMock()
    mock_mlflow.start_span.side_effect = RuntimeError("MLflow broken")

    mock_span_type = MagicMock()
    mock_span_type.CHAIN = "CHAIN"

    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", True),
        patch("databricks_deep_research.tracing.mlflow", mock_mlflow),
        patch("databricks_deep_research.tracing.SpanType", mock_span_type),
    ):
        async with trace_span("broken_span") as span:
            assert span is None
            # Workflow code still runs
            result = 1 + 1
        assert result == 2


@pytest.mark.asyncio
async def test_trace_span_workflow_errors_propagate() -> None:
    """Errors from user code inside trace_span DO propagate normally."""
    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", False),
        pytest.raises(ValueError, match="user error"),
    ):
        async with trace_span("test"):
            raise ValueError("user error")


# ---------------------------------------------------------------------------
# 4. ValueError during __exit__ is silently caught (asyncio.gather scenario)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_span_value_error_on_exit_suppressed() -> None:
    """ValueError during span __exit__ is caught (asyncio.gather context issue)."""
    mock_span = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_span)
    mock_cm.__exit__ = MagicMock(
        side_effect=ValueError("Token was created in a different Context")
    )

    mock_mlflow = MagicMock()
    mock_mlflow.start_span.return_value = mock_cm

    mock_span_type = MagicMock()
    mock_span_type.CHAIN = "CHAIN"

    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", True),
        patch("databricks_deep_research.tracing.mlflow", mock_mlflow),
        patch("databricks_deep_research.tracing.SpanType", mock_span_type),
    ):
        # Should NOT raise
        async with trace_span("gather_span") as span:
            assert span is mock_span


@pytest.mark.asyncio
async def test_trace_span_generic_exit_error_suppressed() -> None:
    """Non-ValueError during span __exit__ is also caught."""
    mock_span = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_span)
    mock_cm.__exit__ = MagicMock(side_effect=RuntimeError("cleanup failed"))

    mock_mlflow = MagicMock()
    mock_mlflow.start_span.return_value = mock_cm

    mock_span_type = MagicMock()
    mock_span_type.CHAIN = "CHAIN"

    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", True),
        patch("databricks_deep_research.tracing.mlflow", mock_mlflow),
        patch("databricks_deep_research.tracing.SpanType", mock_span_type),
    ):
        # Should NOT raise
        async with trace_span("cleanup_error_span") as span:
            assert span is mock_span


# ---------------------------------------------------------------------------
# 5. Nested spans maintain correct hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_trace_spans() -> None:
    """Nested trace_span calls work correctly."""
    call_order: list[str] = []

    def make_mock_cm(name: str) -> MagicMock:
        mock_span = MagicMock()
        mock_span.name = name
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_span)
        cm.__exit__ = MagicMock(return_value=False)

        def track_enter() -> MagicMock:
            call_order.append(f"enter_{name}")
            return mock_span

        def track_exit(*_args: Any) -> bool:
            call_order.append(f"exit_{name}")
            return False

        cm.__enter__.side_effect = track_enter
        cm.__exit__.side_effect = track_exit
        return cm

    cms = [make_mock_cm("outer"), make_mock_cm("inner")]
    call_idx = 0

    mock_mlflow = MagicMock()

    def start_span_side_effect(**_kwargs: Any) -> MagicMock:
        nonlocal call_idx
        cm = cms[call_idx]
        call_idx += 1
        return cm

    mock_mlflow.start_span.side_effect = start_span_side_effect

    mock_span_type = MagicMock()
    mock_span_type.CHAIN = "CHAIN"

    with (
        patch("databricks_deep_research.tracing._HAS_MLFLOW", True),
        patch("databricks_deep_research.tracing.mlflow", mock_mlflow),
        patch("databricks_deep_research.tracing.SpanType", mock_span_type),
    ):
        async with trace_span("outer") as outer_span:
            assert outer_span is not None
            async with trace_span("inner") as inner_span:
                assert inner_span is not None

    assert call_order == ["enter_outer", "enter_inner", "exit_inner", "exit_outer"]


# ---------------------------------------------------------------------------
# 6. Concurrent usage (asyncio.gather)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_span_concurrent_usage() -> None:
    """trace_span works correctly when used concurrently via asyncio.gather."""
    results: list[int] = []

    async def task(idx: int) -> int:
        with patch("databricks_deep_research.tracing._HAS_MLFLOW", False):
            async with trace_span(f"task_{idx}"):
                await asyncio.sleep(0.01)
                results.append(idx)
                return idx

    outcomes = await asyncio.gather(task(0), task(1), task(2))
    assert sorted(outcomes) == [0, 1, 2]
    assert sorted(results) == [0, 1, 2]
