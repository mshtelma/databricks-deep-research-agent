"""MLflow tracing configuration."""

import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Generator, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import mlflow
import mlflow.openai
from mlflow.entities import SpanEvent, SpanType

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)

class _ConcreteSpanEvent(SpanEvent):
    """Concrete subclass to satisfy mypy abstract-class check on SpanEvent."""

    @classmethod
    def from_proto(cls, proto: Any) -> "_ConcreteSpanEvent":
        raise NotImplementedError

def _create_span_event(
    name: str,
    attributes: dict[
        str,
        str | bool | int | float
        | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float],
    ],
) -> SpanEvent:
    """Create a SpanEvent without triggering mypy abstract-class errors."""
    return _ConcreteSpanEvent(name=name, attributes=attributes)


P = ParamSpec("P")
R = TypeVar("R")

# Module-level flag: set to True only when setup_tracing() completes successfully
_tracing_enabled: bool = False


def is_tracing_enabled() -> bool:
    """Check if MLflow tracing was successfully initialized."""
    return _tracing_enabled


# ---------------------------------------------------------------------------
# Span Event Helpers
# ---------------------------------------------------------------------------


def log_trace_event(
    event_name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Log an event to the current MLflow span.

    Safe to call even if no span is active (silently does nothing).

    Args:
        event_name: Name of the event (e.g., "agent_transition", "tool_call").
        attributes: Optional key-value attributes for the event.
    """
    span = mlflow.get_current_active_span()
    if span:
        # Convert attributes to MLflow-compatible types (str, int, float, bool only)
        # Lists/tuples are serialized as JSON strings since protobuf can't handle Python lists
        safe_attrs: dict[
            str,
            str | bool | int | float
            | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float],
        ] = {}
        for k, v in (attributes or {}).items():
            if isinstance(v, str | bool | int | float):
                safe_attrs[k] = v
            elif isinstance(v, list | tuple):
                # Convert list/tuple to JSON string - protobuf can't handle Python lists
                safe_attrs[k] = json.dumps([str(item) for item in v])
            else:
                safe_attrs[k] = str(v)
        event = _create_span_event(name=event_name, attributes=safe_attrs)
        span.add_event(event)


def set_trace_attributes(attributes: dict[str, Any]) -> None:
    """Set attributes on the current MLflow span.

    Safe to call even if no span is active (silently does nothing).

    Args:
        attributes: Key-value attributes to set on the span.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes(attributes)


def setup_tracing() -> None:
    """Configure MLflow tracing for the application.

    Each initialization step is isolated so that a failure in one
    (e.g. set_experiment on a misconfigured workspace) does not
    prevent the remaining steps from executing.

    When MLFLOW_ENABLED=false, skips all MLflow setup and leaves
    _tracing_enabled as False so all safe wrappers become no-ops.
    """
    global _tracing_enabled
    settings = get_settings()

    if not settings.mlflow_enabled:
        logger.info("MLflow tracing disabled via MLFLOW_ENABLED=false")
        _tracing_enabled = False
        return

    from databricks_deep_research.tracing import setup_mlflow_tracing

    ok = setup_mlflow_tracing(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
        experiment_id=settings.mlflow_experiment_id or None,
    )

    _tracing_enabled = True
    logger.info(
        "MLflow tracing setup complete: tracking_uri=%s, experiment_ok=%s",
        settings.mlflow_tracking_uri,
        ok,
    )


def shutdown_tracing() -> None:
    """Flush buffered async traces on shutdown.

    Should be called during application shutdown to ensure all
    pending traces are written before the process exits.
    """
    if not _tracing_enabled:
        return
    from databricks_deep_research.tracing import shutdown_mlflow_tracing

    shutdown_mlflow_tracing()
    logger.info("MLflow traces flushed")


@contextlib.contextmanager
def safe_mlflow_run(run_name: str) -> Generator[None, None, None]:
    """Context manager wrapping mlflow.start_run() with graceful fallback.

    When tracing is disabled or MLflow fails to start the run,
    yields control without an active run. Caller code always executes.
    """
    if not _tracing_enabled:
        yield
        return

    run_started = False
    try:
        mlflow.start_run(run_name=run_name, nested=True)
        run_started = True
    except Exception as e:
        logger.warning("MLflow start_run failed (non-fatal): %s", e)

    try:
        yield
    finally:
        if run_started:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning("MLflow end_run failed (non-fatal): %s", e)


def safe_update_trace(metadata: dict[str, str]) -> None:
    """Update current trace metadata. No-op when tracing is disabled."""
    if not _tracing_enabled:
        return
    with contextlib.suppress(Exception):
        mlflow.update_current_trace(metadata=metadata)


def trace_agent(
    name: str,
    tier: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator to trace agent execution.

    Args:
        name: Agent name (e.g., "coordinator", "planner").
        tier: Model tier used by agent.

    Returns:
        Decorated function with tracing.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        @mlflow.trace(name=name, span_type=SpanType.AGENT)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Add agent metadata to span
            span = mlflow.get_current_active_span()
            if span:
                span.set_attributes(
                    {
                        "agent.name": name,
                        "agent.tier": tier or "unknown",
                    }
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_tool(name: str) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator to trace tool execution.

    Args:
        name: Tool name (e.g., "web_search", "web_crawl").

    Returns:
        Decorated function with tracing.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        @mlflow.trace(name=name, span_type=SpanType.TOOL)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_llm(
    name: str,
    tier: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator to trace LLM calls.

    Args:
        name: Operation name.
        tier: Model tier used.

    Returns:
        Decorated function with tracing.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        @mlflow.trace(name=name, span_type=SpanType.CHAT_MODEL)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            span = mlflow.get_current_active_span()
            if span:
                span.set_attributes(
                    {
                        "llm.tier": tier,
                    }
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def log_research_session(
    session_id: str,
    query: str,
    depth: str,
    status: str,
    duration_ms: float,
    sources_count: int,
    plan_iterations: int,
) -> None:
    """Log research session metrics to MLflow.

    Args:
        session_id: Research session ID.
        query: Original query.
        depth: Research depth.
        status: Final status.
        duration_ms: Total duration in milliseconds.
        sources_count: Number of sources used.
        plan_iterations: Number of plan iterations.
    """
    if not _tracing_enabled:
        return
    try:
        mlflow.log_metrics(
            {
                "research.duration_ms": duration_ms,
                "research.sources_count": sources_count,
                "research.plan_iterations": plan_iterations,
            }
        )
        mlflow.log_params(
            {
                "research.session_id": session_id,
                "research.depth": depth,
                "research.status": status,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log research session: {e}")


def log_feedback(
    session_id: str,
    message_id: str,
    rating: int,
    has_error_report: bool,
) -> None:
    """Log user feedback to MLflow.

    Args:
        session_id: Research session ID.
        message_id: Message ID.
        rating: Feedback rating (-1 or 1).
        has_error_report: Whether error report was provided.
    """
    if not _tracing_enabled:
        return
    try:
        mlflow.log_metrics(
            {
                "feedback.rating": rating,
                "feedback.has_error_report": 1 if has_error_report else 0,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")


@contextlib.asynccontextmanager
async def safe_tool_span(
    name: str,
    span_type: str = SpanType.TOOL,
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """Async context manager for MLflow spans that handles context issues.

    Unlike `with mlflow.start_span()`, this properly handles:
    - asyncio.gather() inside the span
    - Task cancellation
    - Context mismatch during GC cleanup

    The error "Token was created in a different Context" occurs when:
    1. A sync generator-based context manager (mlflow.start_span) creates a ContextVar token
    2. asyncio.gather() spawns concurrent tasks that run in different contexts
    3. If cleanup happens during GC (e.g., due to exception/cancellation), the
       generator finalizes in a different context than where the token was created

    This wrapper catches and suppresses the benign ValueError during cleanup.
    The span data is already recorded - only the ContextVar token cleanup fails.

    Args:
        name: Span name for tracing.
        span_type: MLflow span type (default: TOOL).
        attributes: Optional initial attributes to set on span.

    Yields:
        The MLflow span object (or None if tracing setup fails).
    """
    span = None
    span_cm = None

    try:
        span_cm = mlflow.start_span(name=name, span_type=span_type)
        span = span_cm.__enter__()
        if attributes and span:
            span.set_attributes(attributes)
    except Exception as e:
        logger.warning(f"Span creation failed (non-fatal): {e}")
        span = None
        span_cm = None

    try:
        yield span  # Caller can set more attributes via span
    finally:
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except ValueError as e:
                # Catch ALL ValueErrors during __exit__ - span data is already recorded.
                # The most common is "Token was created in a different Context" but any
                # ValueError here is benign since the span is already persisted.
                logger.debug(f"Span cleanup suppressed (benign): {type(e).__name__}: {e}")
            except Exception as e:
                # Other exceptions during cleanup are also benign - span is recorded
                logger.debug(f"Span cleanup warning: {type(e).__name__}: {e}")


def log_research_config(depth: str) -> None:
    """Log research configuration to MLflow run.

    Uses mlflow.log_params() for searchable run-level attributes.
    Uses mlflow.log_dict() for full config artifact.

    Logs:
    - Research type config (depth, synthesis_mode, generation_mode)
    - Citation verification settings
    - Researcher config (mode, limits)
    - Full config as JSON artifact

    Args:
        depth: Research depth (light, medium, extended).
    """
    if not _tracing_enabled:
        return
    from deep_research.agent.config import get_citation_config_for_depth, get_research_type_config
    from deep_research.core.app_config import get_app_config

    try:
        app_config = get_app_config()

        # Get depth-specific config
        research_type = get_research_type_config(depth)
        citation_config = get_citation_config_for_depth(depth)

        # Log as RUN PARAMS (searchable/filterable in MLflow UI)
        params: dict[str, str | int | float] = {
            # Research type
            "config.research_depth": depth,
            "config.researcher_mode": research_type.researcher.mode.value,
            "config.max_tool_calls": research_type.researcher.max_tool_calls,
            # Steps and report limits
            "config.steps_min": research_type.steps.min,
            "config.steps_max": research_type.steps.max,
            "config.report_max_words": research_type.report_limits.max_words,
            "config.report_max_tokens": research_type.report_limits.max_tokens,
            # Citation verification core
            "config.citation_enabled": str(citation_config.enabled),
            "config.synthesis_mode": citation_config.synthesis_mode.value,
            "config.generation_mode": citation_config.generation_mode.value,
            # Stage toggles
            "config.cv_evidence_preselection": str(
                citation_config.enable_evidence_preselection
            ),
            "config.cv_interleaved_generation": str(
                citation_config.enable_interleaved_generation
            ),
            "config.cv_confidence_classification": str(
                citation_config.enable_confidence_classification
            ),
            "config.cv_citation_correction": str(
                citation_config.enable_citation_correction
            ),
            "config.cv_numeric_qa_verification": str(
                citation_config.enable_numeric_qa_verification
            ),
            "config.cv_verification_retrieval": str(
                citation_config.enable_verification_retrieval
            ),
            # Model config
            "config.default_role": app_config.default_role,
        }

        mlflow.log_params(params)

        # Log full config as JSON artifact
        config_dict = {
            "research_type": research_type.model_dump(mode="json"),
            "citation_verification": citation_config.model_dump(mode="json"),
            "default_role": app_config.default_role,
            "endpoints": {
                k: v.model_dump(mode="json") for k, v in app_config.endpoints.items()
            },
            "models": {
                k: v.model_dump(mode="json") for k, v in app_config.models.items()
            },
        }
        mlflow.log_dict(config_dict, "config/research_config.json")

    except Exception as e:
        # Don't fail research if config logging fails
        logger.warning(f"Failed to log research config: {e}")
