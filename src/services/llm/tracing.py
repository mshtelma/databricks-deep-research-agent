"""MLflow tracing utilities for streaming LLM responses.

This module provides wrappers to add MLflow tracing to async streaming
LLM responses. The key challenge is that MLflow spans don't work well
with async generators - the span context may not survive suspend/resume cycles.

The solution is TracedAsyncIterator which:
- Creates the span when iteration begins (not when generator is created)
- Ends the span when iterator exhausts or errors
- Captures metrics incrementally during streaming
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import mlflow
from mlflow.entities import SpanType

from src.services.llm.types import ModelTier, StreamWithToolsChunk


def _normalize_messages_for_span(messages: list[dict[str, Any]] | None) -> list[dict[str, str]] | None:
    """Normalize messages for MLflow span logging.

    This is a local copy to avoid circular import with client.py.
    Converts message content from list format (used by cache_control or
    Gemini responses) to plain strings to avoid Pydantic serialization
    warnings when MLflow logs the messages.

    Args:
        messages: List of message dicts, potentially with list content.

    Returns:
        List of message dicts with all content normalized to strings, or None.
    """
    if messages is None:
        return None

    normalized: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        # Normalize list content to string
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        elif not isinstance(content, str):
            content = str(content) if content else ""

        normalized_msg: dict[str, str] = {
            "role": str(msg.get("role", "user")),
            "content": content,
        }
        # Preserve tool_call_id if present (for tool response messages)
        if "tool_call_id" in msg:
            normalized_msg["tool_call_id"] = str(msg["tool_call_id"])
        if "name" in msg:
            normalized_msg["name"] = str(msg["name"])
        normalized.append(normalized_msg)
    return normalized

T = TypeVar("T")


@dataclass
class StreamMetrics:
    """Accumulated metrics during streaming."""

    start_time: float = field(default_factory=time.perf_counter)
    chunk_count: int = 0
    total_content_len: int = 0
    tool_calls_count: int = 0
    error: Exception | None = None


class TracedAsyncIterator(AsyncIterator[T]):
    """Async iterator wrapper that creates MLflow span on consumption.

    The span is:
    - Created when iteration begins (first __anext__ call)
    - Ended when iterator is exhausted or an error occurs
    - Set with error attributes if an exception is raised

    This ensures the span accurately reflects the streaming duration,
    not the generator creation time.

    Example:
        async def my_stream() -> AsyncIterator[str]:
            yield "Hello"
            yield " world"

        # Wrap with tracing
        traced = TracedAsyncIterator(
            inner=my_stream(),
            span_name="my_stream",
            tier=ModelTier.SIMPLE,
            endpoint="my-endpoint",
            message_count=1,
        )

        # Span is created on first iteration
        async for chunk in traced:
            print(chunk)
        # Span ends when iterator exhausts
    """

    def __init__(
        self,
        inner: AsyncIterator[T],
        span_name: str,
        tier: ModelTier,
        endpoint: str,
        message_count: int,
        max_tokens: int | None = None,
        temperature: float | None = None,
        has_tools: bool = False,
        messages: list[dict[str, Any]] | None = None,
    ):
        """Initialize the traced iterator wrapper.

        Args:
            inner: The underlying async iterator to wrap.
            span_name: Name for the MLflow span.
            tier: Model tier (simple, analytical, complex).
            endpoint: Endpoint identifier.
            message_count: Number of messages in the request.
            max_tokens: Max tokens parameter.
            temperature: Temperature parameter.
            has_tools: Whether this stream includes tool calls.
            messages: Input messages for tracing visibility.
        """
        self._inner = inner
        self._span_name = span_name
        self._tier = tier
        self._endpoint = endpoint
        self._message_count = message_count
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._has_tools = has_tools
        self._messages = messages
        self._span_cm: Any = None  # The context manager from mlflow.start_span()
        self._span: Any = None  # The actual span (from __enter__)
        self._metrics = StreamMetrics()
        self._accumulated_content: list[str] = []  # Accumulate streamed content
        self._started = False
        self._finished = False

    def _start_span(self) -> None:
        """Start the MLflow span on first iteration."""
        if self._started:
            return
        self._started = True
        self._metrics.start_time = time.perf_counter()

        # Use context manager protocol manually - this properly sets up trace context
        # mlflow.start_span() returns a context manager; we enter it to get the span
        self._span_cm = mlflow.start_span(
            name=self._span_name,
            span_type=SpanType.CHAT_MODEL,
        )
        self._span = self._span_cm.__enter__()

        # Set inputs and attributes
        # Normalize messages to avoid Pydantic serialization warnings with list content
        if self._messages:
            normalized = _normalize_messages_for_span(self._messages)
            if normalized:
                self._span.set_inputs({"messages": normalized})

        self._span.set_attributes({
            "llm.tier": self._tier.value,
            "llm.endpoint": self._endpoint,
            "llm.message_count": self._message_count,
            "llm.max_tokens": self._max_tokens,
            "llm.temperature": self._temperature,
            "llm.streaming": True,
            "llm.has_tools": self._has_tools,
        })

    def _end_span(self) -> None:
        """End the MLflow span with final metrics."""
        if self._finished or self._span is None:
            return
        self._finished = True

        duration_ms = (time.perf_counter() - self._metrics.start_time) * 1000

        # Set final attributes
        attrs: dict[str, Any] = {
            "llm.latency_ms": round(duration_ms, 1),
            "llm.chunk_count": self._metrics.chunk_count,
            "llm.content_len": self._metrics.total_content_len,
        }

        if self._has_tools:
            attrs["llm.tool_calls_count"] = self._metrics.tool_calls_count

        if self._metrics.error:
            attrs["llm.error"] = True
            attrs["llm.error_type"] = type(self._metrics.error).__name__
            attrs["llm.error_message"] = str(self._metrics.error)[:200]

        self._span.set_attributes(attrs)

        # Set accumulated output content
        accumulated = "".join(self._accumulated_content)
        self._span.set_outputs({"content": accumulated})

        # Exit the context manager properly - this ends the span and cleans up trace context
        self._span_cm.__exit__(None, None, None)

    def __aiter__(self) -> "TracedAsyncIterator[T]":
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> T:
        """Get the next item from the wrapped iterator.

        Creates the span on first call, tracks metrics for each chunk,
        and ends the span on exhaustion or error.
        """
        self._start_span()

        try:
            item = await self._inner.__anext__()
            self._metrics.chunk_count += 1

            # Track content length for string chunks
            if isinstance(item, str):
                self._metrics.total_content_len += len(item)
                self._accumulated_content.append(item)
            # Track tool calls for StreamWithToolsChunk
            elif isinstance(item, StreamWithToolsChunk):
                if item.content:
                    self._metrics.total_content_len += len(item.content)
                    self._accumulated_content.append(item.content)
                if item.tool_calls:
                    self._metrics.tool_calls_count += len(item.tool_calls)

            return item

        except StopAsyncIteration:
            self._end_span()
            raise
        except Exception as e:
            self._metrics.error = e
            self._end_span()
            raise


def traced_stream(
    inner: AsyncIterator[str],
    tier: ModelTier,
    endpoint: str,
    message_count: int,
    max_tokens: int | None = None,
    temperature: float | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Wrap a streaming iterator with MLflow tracing.

    Args:
        inner: The underlying async iterator from OpenAI streaming.
        tier: Model tier (simple, analytical, complex).
        endpoint: Endpoint identifier.
        message_count: Number of messages in the request.
        max_tokens: Max tokens parameter.
        temperature: Temperature parameter.
        messages: Input messages for tracing visibility.

    Returns:
        Traced async iterator that captures span on consumption.
    """
    return TracedAsyncIterator(
        inner=inner,
        span_name=f"llm_stream_{tier.value}",
        tier=tier,
        endpoint=endpoint,
        message_count=message_count,
        max_tokens=max_tokens,
        temperature=temperature,
        has_tools=False,
        messages=messages,
    )


def traced_stream_with_tools(
    inner: AsyncIterator[StreamWithToolsChunk],
    tier: ModelTier,
    endpoint: str,
    message_count: int,
    max_tokens: int | None = None,
    temperature: float | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> AsyncIterator[StreamWithToolsChunk]:
    """Wrap a streaming-with-tools iterator with MLflow tracing.

    Args:
        inner: The underlying async iterator.
        tier: Model tier.
        endpoint: Endpoint identifier.
        message_count: Number of messages.
        max_tokens: Max tokens parameter.
        temperature: Temperature parameter.
        messages: Input messages for tracing visibility.

    Returns:
        Traced async iterator that captures span and tool call metrics.
    """
    return TracedAsyncIterator(
        inner=inner,
        span_name=f"llm_stream_tools_{tier.value}",
        tier=tier,
        endpoint=endpoint,
        message_count=message_count,
        max_tokens=max_tokens,
        temperature=temperature,
        has_tools=True,
        messages=messages,
    )
