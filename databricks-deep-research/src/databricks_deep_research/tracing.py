"""Optional MLflow tracing — zero overhead when mlflow is not installed."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.entities import SpanType

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def get_current_span() -> Any:
    """Return the current active MLflow span, or ``None`` if unavailable."""
    if not _HAS_MLFLOW:
        return None
    try:
        return mlflow.get_current_active_span()
    except Exception:
        return None


@contextlib.asynccontextmanager
async def trace_span(
    name: str,
    span_type: str = "CHAIN",
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """Create an MLflow span if available, otherwise no-op.

    Async-safe: handles asyncio.gather() context issues by manually
    managing __enter__/__exit__ and catching ValueError during cleanup
    (same pattern as app's safe_tool_span).
    """
    if not _HAS_MLFLOW:
        yield None
        return

    span = None
    span_cm = None

    try:
        mlflow_type = getattr(SpanType, span_type, SpanType.CHAIN)
        span_cm = mlflow.start_span(name=name, span_type=mlflow_type)
        span = span_cm.__enter__()
        if attributes and span:
            span.set_attributes(attributes)
    except Exception as e:
        logger.debug("TRACING_SPAN_CREATE_FAILED name=%s error=%s", name, e)
        span = None
        span_cm = None

    try:
        yield span
    finally:
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except ValueError:
                # "Token was created in a different Context" — benign in asyncio.gather
                pass
            except Exception as e:
                logger.debug("TRACING_SPAN_CLOSE_FAILED name=%s error=%s", name, e)
