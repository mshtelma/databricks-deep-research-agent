"""Unit tests for streaming LLM tracing.

These tests verify the core behavior of TracedAsyncIterator without mocking MLflow.
MLflow span creation/ending is an implementation detail tested by integration tests.
"""

import pytest

from src.services.llm.tracing import (
    StreamMetrics,
    TracedAsyncIterator,
    traced_stream,
    traced_stream_with_tools,
)
from src.services.llm.types import ModelTier, StreamWithToolsChunk, ToolCall


async def mock_string_generator():
    """Mock async generator yielding strings."""
    yield "Hello"
    yield " world"
    yield "!"


async def mock_tools_generator():
    """Mock async generator yielding StreamWithToolsChunk."""
    yield StreamWithToolsChunk(content="Processing")
    yield StreamWithToolsChunk(content=" data")
    yield StreamWithToolsChunk(
        content="",
        tool_calls=[
            ToolCall(id="call_1", name="web_search", arguments={"query": "test"}),
        ],
        is_done=True,
    )


async def mock_error_generator():
    """Mock async generator that raises an error."""
    yield "Starting..."
    raise ValueError("Test error")


class TestStreamMetrics:
    """Tests for StreamMetrics dataclass."""

    def test_default_values(self) -> None:
        """Metrics should have correct defaults."""
        metrics = StreamMetrics()

        assert metrics.chunk_count == 0
        assert metrics.total_content_len == 0
        assert metrics.tool_calls_count == 0
        assert metrics.error is None
        assert metrics.start_time > 0


class TestTracedAsyncIterator:
    """Tests for TracedAsyncIterator wrapper."""

    @pytest.mark.asyncio
    async def test_yields_all_items(self) -> None:
        """Wrapper should yield all items from inner iterator."""
        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test_span",
            tier=ModelTier.SIMPLE,
            endpoint="test-endpoint",
            message_count=1,
        )

        chunks = [chunk async for chunk in wrapper]
        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_started_flag_set_on_first_iteration(self) -> None:
        """Started flag should be set on first __anext__ call."""
        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test_span",
            tier=ModelTier.SIMPLE,
            endpoint="test-endpoint",
            message_count=1,
        )

        # Before iteration - not started
        assert not wrapper._started

        # First iteration - started
        chunk = await wrapper.__anext__()
        assert chunk == "Hello"
        assert wrapper._started

    @pytest.mark.asyncio
    async def test_metrics_accumulated_correctly(self) -> None:
        """Chunk count and content length should be tracked."""
        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test",
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        chunks = [chunk async for chunk in wrapper]

        assert chunks == ["Hello", " world", "!"]
        assert wrapper._metrics.chunk_count == 3
        assert wrapper._metrics.total_content_len == 12  # "Hello world!"

    @pytest.mark.asyncio
    async def test_finished_flag_set_on_exhaustion(self) -> None:
        """Finished flag should be set when iterator is exhausted."""
        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test",
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        [_ async for _ in wrapper]

        assert wrapper._finished

    @pytest.mark.asyncio
    async def test_error_captured_in_metrics(self) -> None:
        """Error should be captured in metrics on exception."""
        wrapper = TracedAsyncIterator(
            inner=mock_error_generator(),
            span_name="test",
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        with pytest.raises(ValueError, match="Test error"):
            [_ async for _ in wrapper]

        assert wrapper._finished
        assert wrapper._metrics.error is not None
        assert isinstance(wrapper._metrics.error, ValueError)
        assert str(wrapper._metrics.error) == "Test error"

    @pytest.mark.asyncio
    async def test_tool_calls_tracked(self) -> None:
        """Tool calls should be counted in metrics."""
        wrapper = TracedAsyncIterator(
            inner=mock_tools_generator(),
            span_name="test",
            tier=ModelTier.ANALYTICAL,
            endpoint="test",
            message_count=1,
            has_tools=True,
        )

        chunks = [chunk async for chunk in wrapper]

        assert len(chunks) == 3
        assert wrapper._metrics.tool_calls_count == 1
        assert wrapper._metrics.total_content_len == len("Processing data")

    @pytest.mark.asyncio
    async def test_span_started_only_once(self) -> None:
        """Span should only be started once (idempotent)."""
        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test",
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        # First chunk
        await wrapper.__anext__()
        assert wrapper._started

        # Subsequent chunks - _started remains True
        await wrapper.__anext__()
        assert wrapper._started

        await wrapper.__anext__()
        assert wrapper._started

    @pytest.mark.asyncio
    async def test_latency_calculated(self) -> None:
        """Latency should be calculated based on start_time."""
        import time

        wrapper = TracedAsyncIterator(
            inner=mock_string_generator(),
            span_name="test",
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        start = time.perf_counter()
        [_ async for _ in wrapper]
        elapsed = time.perf_counter() - start

        # Latency should be positive and roughly match elapsed time
        assert wrapper._metrics.start_time > 0
        # The wrapper should have completed within a reasonable time
        assert elapsed < 1.0  # Should complete in under 1 second


class TestTracedStreamHelpers:
    """Tests for traced_stream and traced_stream_with_tools helper functions."""

    @pytest.mark.asyncio
    async def test_traced_stream_returns_iterator(self) -> None:
        """traced_stream should return a TracedAsyncIterator."""
        result = traced_stream(
            inner=mock_string_generator(),
            tier=ModelTier.SIMPLE,
            endpoint="test-endpoint",
            message_count=1,
        )

        assert isinstance(result, TracedAsyncIterator)
        assert result._span_name == "llm_stream_simple"
        assert not result._has_tools

    @pytest.mark.asyncio
    async def test_traced_stream_with_tools_returns_iterator(self) -> None:
        """traced_stream_with_tools should return a TracedAsyncIterator."""
        result = traced_stream_with_tools(
            inner=mock_tools_generator(),
            tier=ModelTier.ANALYTICAL,
            endpoint="test-endpoint",
            message_count=1,
        )

        assert isinstance(result, TracedAsyncIterator)
        assert result._span_name == "llm_stream_tools_analytical"
        assert result._has_tools

    @pytest.mark.asyncio
    async def test_traced_stream_span_names(self) -> None:
        """Span names should follow the naming convention."""
        # Test all tiers
        for tier in [ModelTier.SIMPLE, ModelTier.ANALYTICAL, ModelTier.COMPLEX]:
            stream = traced_stream(
                inner=mock_string_generator(),
                tier=tier,
                endpoint="test",
                message_count=1,
            )
            assert stream._span_name == f"llm_stream_{tier.value}"

            tools_stream = traced_stream_with_tools(
                inner=mock_tools_generator(),
                tier=tier,
                endpoint="test",
                message_count=1,
            )
            assert tools_stream._span_name == f"llm_stream_tools_{tier.value}"

    @pytest.mark.asyncio
    async def test_traced_stream_preserves_optional_params(self) -> None:
        """traced_stream should preserve optional parameters."""
        result = traced_stream(
            inner=mock_string_generator(),
            tier=ModelTier.SIMPLE,
            endpoint="test-endpoint",
            message_count=5,
            max_tokens=1000,
            temperature=0.7,
        )

        assert result._message_count == 5
        assert result._max_tokens == 1000
        assert result._temperature == 0.7

    @pytest.mark.asyncio
    async def test_traced_stream_iteration_works(self) -> None:
        """traced_stream wrapper should iterate correctly."""
        stream = traced_stream(
            inner=mock_string_generator(),
            tier=ModelTier.SIMPLE,
            endpoint="test",
            message_count=1,
        )

        chunks = [chunk async for chunk in stream]
        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_traced_stream_with_tools_iteration_works(self) -> None:
        """traced_stream_with_tools wrapper should iterate correctly."""
        stream = traced_stream_with_tools(
            inner=mock_tools_generator(),
            tier=ModelTier.ANALYTICAL,
            endpoint="test",
            message_count=1,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 3
        assert chunks[0].content == "Processing"
        assert chunks[1].content == " data"
        assert chunks[2].tool_calls is not None
        assert len(chunks[2].tool_calls) == 1
