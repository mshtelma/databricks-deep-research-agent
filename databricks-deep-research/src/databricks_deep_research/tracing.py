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


_mlflow_async_enabled: bool = False


def setup_mlflow_tracing(
    *,
    tracking_uri: str = "databricks",
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    async_logging: bool = True,
    openai_autolog: bool = True,
) -> bool:
    """Configure MLflow tracing end-to-end.

    Each step is isolated — a failure in one does not prevent the rest.
    Returns True when tracing is expected to produce recorded spans.
    Idempotent: safe to call more than once.
    """
    global _mlflow_async_enabled

    if not _HAS_MLFLOW:
        return False

    ok = True

    # Step 1: async logging (non-critical — only affects delivery mode)
    if async_logging:
        try:
            mlflow.config.enable_async_logging(True)
            _mlflow_async_enabled = True
        except Exception as e:
            logger.warning("MLflow async logging setup failed: %s", e)

    # Step 2: tracking URI
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as e:
        logger.warning("MLflow tracking URI setup failed: %s", e)
        ok = False

    # Step 3: experiment (by ID or name)
    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning("MLflow set_experiment failed: %s", e)
        ok = False

    # Step 4: enable tracing — CRITICAL for start_span() to record
    try:
        mlflow.tracing.enable()
    except Exception as e:
        logger.warning("mlflow.tracing.enable() failed: %s", e)
        ok = False

    # Step 5: auto-instrument OpenAI client calls (non-critical)
    if openai_autolog:
        try:
            mlflow.openai.autolog()
        except Exception as e:
            logger.debug("mlflow.openai.autolog() skipped: %s", e)

    return ok


def shutdown_mlflow_tracing() -> None:
    """Flush buffered async traces. Safe to call even if setup was not called."""
    global _mlflow_async_enabled

    if not _HAS_MLFLOW or not _mlflow_async_enabled:
        return
    try:
        mlflow.flush_trace_async_logging(terminate=True)
    except Exception as e:
        logger.debug("MLflow trace flush failed: %s", e)
    finally:
        _mlflow_async_enabled = False


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
