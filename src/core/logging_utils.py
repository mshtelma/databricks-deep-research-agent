"""Structured logging utilities for comprehensive request/response tracing."""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.core.config import get_settings
from src.core.tracing import log_trace_event


@dataclass
class LogContext:
    """Context for correlating logs across a request lifecycle."""

    session_id: str | None = None
    request_id: str | None = None
    agent_name: str | None = None
    step: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result: dict[str, Any] = {}
        if self.session_id:
            result["session_id"] = self.session_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.agent_name:
            result["agent"] = self.agent_name
        if self.step is not None:
            result["step"] = self.step
        result.update(self.extra)
        return result

    def format_prefix(self) -> str:
        """Format as log prefix string."""
        parts = []
        if self.session_id:
            parts.append(f"session={self.session_id[:8]}")
        if self.request_id:
            parts.append(f"req={self.request_id[:8]}")
        if self.agent_name:
            parts.append(f"agent={self.agent_name}")
        if self.step is not None:
            parts.append(f"step={self.step}")
        return " | ".join(parts)


def truncate(text: str | None, max_length: int = 100) -> str:
    """Truncate text for logging, adding ellipsis if truncated."""
    if text is None:
        return "<none>"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_log_dict(data: dict[str, Any]) -> str:
    """Format dictionary as key=value pairs for logging."""
    parts = []
    for key, value in data.items():
        if isinstance(value, str):
            # Quote strings, truncate if long
            truncated = truncate(value, 80)
            parts.append(f'{key}="{truncated}"')
        elif isinstance(value, list | tuple):
            # Show first few items of lists
            if len(value) <= 3:
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}=[{value[0]}, {value[1]}, ... +{len(value)-2} more]")
        elif isinstance(value, dict):
            parts.append(f"{key}={{...{len(value)} keys}}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


class StructuredLogger:
    """Logger wrapper that adds structured context to all log messages."""

    def __init__(self, name: str, context: LogContext | None = None):
        self._logger = logging.getLogger(name)
        self._context = context or LogContext()

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with context and extra data."""
        prefix = self._context.format_prefix()
        if kwargs:
            extra = format_log_dict(kwargs)
            if prefix:
                return f"{prefix} | {message} | {extra}"
            return f"{message} | {extra}"
        if prefix:
            return f"{prefix} | {message}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._logger.error(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with context and traceback."""
        self._logger.exception(self._format_message(message, **kwargs))

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        """Create new logger with additional context."""
        new_context = LogContext(
            session_id=kwargs.get("session_id", self._context.session_id),
            request_id=kwargs.get("request_id", self._context.request_id),
            agent_name=kwargs.get("agent_name", self._context.agent_name),
            step=kwargs.get("step", self._context.step),
            extra={**self._context.extra, **{k: v for k, v in kwargs.items()
                   if k not in ("session_id", "request_id", "agent_name", "step")}},
        )
        return StructuredLogger(self._logger.name, new_context)


def get_logger(name: str, context: LogContext | None = None) -> StructuredLogger:
    """Get a structured logger with optional context."""
    return StructuredLogger(name, context)


# Specialized logging functions for common operations

def log_llm_request(
    logger: StructuredLogger,
    endpoint: str,
    tier: str,
    messages: list[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    estimated_tokens: int,
) -> None:
    """Log an LLM request."""
    settings = get_settings()

    # Get message summary
    message_count = len(messages)
    last_message = messages[-1] if messages else {}
    last_role = last_message.get("role", "unknown")
    last_content = last_message.get("content", "")

    logger.info(
        "LLM_REQUEST",
        endpoint=endpoint,
        tier=tier,
        messages=message_count,
        last_role=last_role,
        last_content_preview=truncate(last_content, 150) if settings.debug else truncate(last_content, 50),
        temperature=temperature,
        max_tokens=max_tokens,
        est_tokens=estimated_tokens,
    )


def log_llm_response(
    logger: StructuredLogger,
    endpoint: str,
    duration_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    content: str,
    structured_type: str | None = None,
) -> None:
    """Log an LLM response."""
    settings = get_settings()

    logger.info(
        "LLM_RESPONSE",
        endpoint=endpoint,
        duration_ms=round(duration_ms, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        content_len=len(content),
        content_preview=truncate(content, 200) if settings.debug else truncate(content, 80),
        structured=structured_type,
    )


def log_llm_error(
    logger: StructuredLogger,
    endpoint: str,
    error: Exception,
    is_rate_limit: bool = False,
    will_fallback: bool = False,
) -> None:
    """Log an LLM error."""
    logger.error(
        "LLM_ERROR",
        endpoint=endpoint,
        error_type=type(error).__name__,
        error=str(error)[:200],
        is_rate_limit=is_rate_limit,
        will_fallback=will_fallback,
    )


def log_llm_fallback(
    logger: StructuredLogger,
    from_endpoint: str,
    to_endpoint: str,
    reason: str,
) -> None:
    """Log an LLM fallback attempt."""
    logger.warning(
        "LLM_FALLBACK",
        from_endpoint=from_endpoint,
        to_endpoint=to_endpoint,
        reason=reason,
    )


def log_search_request(
    logger: StructuredLogger,
    query: str,
    count: int,
) -> None:
    """Log a search request."""
    logger.info(
        "SEARCH_REQUEST",
        query=query,
        count=count,
    )


def log_search_response(
    logger: StructuredLogger,
    query: str,
    result_count: int,
    urls: list[str],
    duration_ms: float,
) -> None:
    """Log a search response."""
    logger.info(
        "SEARCH_RESPONSE",
        query=truncate(query, 60),
        results=result_count,
        urls=urls[:5],  # First 5 URLs
        duration_ms=round(duration_ms, 1),
    )


def log_search_error(
    logger: StructuredLogger,
    query: str,
    error: Exception,
    status_code: int | None = None,
) -> None:
    """Log a search error."""
    logger.error(
        "SEARCH_ERROR",
        query=truncate(query, 60),
        error_type=type(error).__name__,
        error=str(error)[:200],
        status_code=status_code,
    )


def log_crawl_request(
    logger: StructuredLogger,
    url: str,
) -> None:
    """Log a crawl request."""
    logger.debug(
        "CRAWL_REQUEST",
        url=url,
    )


def log_crawl_response(
    logger: StructuredLogger,
    url: str,
    status_code: int,
    content_length: int,
    duration_ms: float,
) -> None:
    """Log a crawl response."""
    logger.info(
        "CRAWL_RESPONSE",
        url=truncate(url, 80),
        status=status_code,
        content_len=content_length,
        duration_ms=round(duration_ms, 1),
    )


def log_crawl_error(
    logger: StructuredLogger,
    url: str,
    error: Exception,
) -> None:
    """Log a crawl error."""
    logger.warning(
        "CRAWL_ERROR",
        url=truncate(url, 80),
        error_type=type(error).__name__,
        error=str(error)[:150],
    )


def log_agent_phase(
    logger: StructuredLogger,
    phase: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Log an agent phase transition."""
    if details:
        logger.info(f"AGENT_PHASE:{phase}", **details)
    else:
        logger.info(f"AGENT_PHASE:{phase}")


def log_agent_decision(
    logger: StructuredLogger,
    decision: str,
    reasoning: str,
    **kwargs: Any,
) -> None:
    """Log an agent decision."""
    logger.info(
        "AGENT_DECISION",
        decision=decision,
        reasoning=truncate(reasoning, 150),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Enhanced Logging Functions (Console + MLflow Trace)
# ---------------------------------------------------------------------------


def log_llm_prompt(
    logger: StructuredLogger,
    system_prompt: str | None,
    user_prompt: str,
    max_prompt_length: int = 2000,
) -> None:
    """Log LLM prompt details at DEBUG level and to MLflow trace.

    Args:
        logger: Structured logger instance.
        system_prompt: System prompt content (may be None).
        user_prompt: User prompt content.
        max_prompt_length: Maximum length before truncation (default 2000).
    """
    sys_truncated = truncate(system_prompt, max_prompt_length) if system_prompt else "(none)"
    user_truncated = truncate(user_prompt, max_prompt_length)

    # Console log (DEBUG)
    logger.debug(
        "LLM_PROMPT",
        system_len=len(system_prompt) if system_prompt else 0,
        system_prompt=sys_truncated,
        user_len=len(user_prompt),
        user_prompt=user_truncated,
    )

    # MLflow trace event
    log_trace_event("llm_prompt", {
        "system_prompt": sys_truncated,
        "user_prompt": user_truncated,
        "system_len": len(system_prompt) if system_prompt else 0,
        "user_len": len(user_prompt),
    })


def log_agent_transition(
    logger: StructuredLogger,
    from_agent: str | None,
    to_agent: str,
    reason: str | None = None,
) -> None:
    """Log transition between agents at INFO level and to MLflow trace.

    Args:
        logger: Structured logger instance.
        from_agent: Previous agent name (None if starting).
        to_agent: Next agent name.
        reason: Optional reason for transition.
    """
    if from_agent:
        msg = f"AGENT_TRANSITION: {from_agent} -> {to_agent}"
    else:
        msg = f"AGENT_START: {to_agent}"

    # Console log (INFO)
    if reason:
        logger.info(msg, reason=reason)
    else:
        logger.info(msg)

    # MLflow trace event
    log_trace_event("agent_transition", {
        "from_agent": from_agent or "(start)",
        "to_agent": to_agent,
        "reason": reason or "",
    })


def log_search_queries_generated(
    logger: StructuredLogger,
    step_title: str,
    queries: list[str],
) -> None:
    """Log generated search queries at INFO level and to MLflow trace.

    Args:
        logger: Structured logger instance.
        step_title: Title of the research step.
        queries: List of generated search queries.
    """
    # Console log (INFO)
    logger.info(
        "SEARCH_QUERIES_GENERATED",
        step=truncate(step_title, 40),
        count=len(queries),
        queries=queries[:3],
    )

    # MLflow trace event
    log_trace_event("search_queries_generated", {
        "step": truncate(step_title, 40),
        "count": len(queries),
        "queries": queries[:5],  # Show more in trace
    })


def log_urls_selected(
    logger: StructuredLogger,
    purpose: str,
    urls: list[str],
    from_total: int,
) -> None:
    """Log URL selection at DEBUG level and to MLflow trace.

    Args:
        logger: Structured logger instance.
        purpose: Why URLs were selected (e.g., "crawl", "analyze").
        urls: List of selected URLs.
        from_total: Total URLs available before selection.
    """
    # Console log (DEBUG)
    logger.debug(
        "URLS_SELECTED",
        purpose=purpose,
        selected=len(urls),
        from_total=from_total,
        urls=urls,
    )

    # MLflow trace event
    log_trace_event("urls_selected", {
        "purpose": purpose,
        "selected": len(urls),
        "from_total": from_total,
        "urls": urls,
    })


def log_tool_call(
    logger: StructuredLogger,
    tool_name: str,
    params: dict[str, Any],
) -> None:
    """Log tool invocation at INFO level and to MLflow trace.

    Args:
        logger: Structured logger instance.
        tool_name: Name of the tool being called.
        params: Parameters passed to the tool.
    """
    truncated_params = {
        k: truncate(str(v), 100) if isinstance(v, str) else v
        for k, v in params.items()
    }

    # Console log (INFO)
    logger.info(f"TOOL_CALL:{tool_name}", **truncated_params)

    # MLflow trace event
    log_trace_event(f"tool_call_{tool_name}", {
        "tool": tool_name,
        **truncated_params,
    })


@contextmanager
def timed_operation(logger: StructuredLogger, operation: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Context manager for timing an operation."""
    start = time.perf_counter()
    result: dict[str, Any] = {"success": False}
    try:
        yield result
        result["success"] = True
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        level = "info" if result["success"] else "error"
        getattr(logger, level)(
            f"TIMED:{operation}",
            duration_ms=round(duration_ms, 1),
            success=result["success"],
            **kwargs,
        )


@asynccontextmanager
async def async_timed_operation(
    logger: StructuredLogger, operation: str, **kwargs: Any
) -> AsyncIterator[dict[str, Any]]:
    """Async context manager for timing an operation."""
    start = time.perf_counter()
    result: dict[str, Any] = {"success": False}
    try:
        yield result
        result["success"] = True
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        level = "info" if result["success"] else "error"
        getattr(logger, level)(
            f"TIMED:{operation}",
            duration_ms=round(duration_ms, 1),
            success=result["success"],
            **kwargs,
        )
