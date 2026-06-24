"""Unit tests for the bounded follow-up live-search escape hatch (spec §4.7).

Domain-agnostic: no example/benchmark-specific topic strings. The search client
and LLM are stubbed — no network. These assert the *bounds* (hard result cap +
timeout), the graceful fallback contract, and that provenance is surfaced in the
streamed events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest

from deep_research.agent.followup.live_search import (
    LiveSearchUnavailable,
    stream_live_search_answer,
)
from deep_research.schemas.streaming import (
    StepCompletedEvent,
    SynthesisProgressEvent,
    ToolCallEvent,
    ToolResultEvent,
)


@dataclass
class _FakeResult:
    """Duck-typed framework SearchResult."""

    url: str
    title: str = "Untitled"
    snippet: str = "snippet"
    content: str | None = None
    relevance_score: float | None = 0.5


class _FakeSearchClient:
    """Records the count it was asked for and returns a fixed result list."""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.last_count: int | None = None
        self.call_count = 0

    async def search(
        self, query: str, *, count: int = 10, freshness: str | None = None
    ) -> list[_FakeResult]:
        self.call_count += 1
        self.last_count = count
        # Honor the requested count the way a real backend would.
        return self._results[:count]


class _HangingSearchClient:
    """A search client whose ``search`` never returns (to exercise the timeout)."""

    def __init__(self) -> None:
        self.call_count = 0

    async def search(
        self, query: str, *, count: int = 10, freshness: str | None = None
    ) -> list[_FakeResult]:
        self.call_count += 1
        await asyncio.Event().wait()  # blocks forever
        return []  # pragma: no cover


class _FakeLLM:
    """Stub LLMClient: ``stream`` yields fixed chunks; nothing else is used."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def stream(
        self, messages: list[dict[str, str]], tier: Any, **_: Any
    ) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


async def _collect(agen: Any) -> list[Any]:
    return [evt async for evt in agen]


# ---------------------------------------------------------------------------
# Happy path: bounded search yields provenance + a grounded answer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_search_streams_provenance_and_answer() -> None:
    client = _FakeSearchClient(
        [_FakeResult(url=f"https://example.test/{i}") for i in range(3)]
    )
    llm = _FakeLLM(["Here ", "is ", "the answer."])

    events = await _collect(
        stream_live_search_answer(
            query="a focused factual lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=client,
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    # No fallback sentinel on the happy path.
    assert not any(isinstance(e, LiveSearchUnavailable) for e in events)

    # Provenance is surfaced: a tool call AND a tool result attributing the
    # fresh sources (NOT silently merged into the prior pool).
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    step_done = [e for e in events if isinstance(e, StepCompletedEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "web_search"
    assert tool_calls[0].source_type == "web_search"
    assert len(tool_results) == 1
    assert tool_results[0].sources_added == 3
    assert tool_results[0].sources_crawled == 3
    assert len(step_done) == 1
    assert step_done[0].sources_found == 3

    # The grounded answer streamed as synthesis progress chunks.
    answer = "".join(
        e.content_chunk for e in events if isinstance(e, SynthesisProgressEvent)
    )
    assert answer == "Here is the answer."


# ---------------------------------------------------------------------------
# Hard cap: never admit more than max_results, even if the backend over-returns.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_cap_is_enforced() -> None:
    # Backend has 20 candidates; cap is 5.
    client = _FakeSearchClient(
        [_FakeResult(url=f"https://example.test/{i}") for i in range(20)]
    )
    llm = _FakeLLM(["answer"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=client,
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    # The backend was asked for exactly the cap.
    assert client.last_count == 5
    # And at most the cap was admitted/attributed.
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].sources_added == 5
    step_done = [e for e in events if isinstance(e, StepCompletedEvent)]
    assert step_done[0].sources_found == 5


@pytest.mark.asyncio
async def test_result_cap_overreturn_is_sliced() -> None:
    # Even if a misbehaving backend ignores ``count`` and returns more, the
    # module slices to the cap as a belt-and-braces guard.
    class _OverReturningClient(_FakeSearchClient):
        async def search(
            self, query: str, *, count: int = 10, freshness: str | None = None
        ) -> list[_FakeResult]:
            self.call_count += 1
            self.last_count = count
            return self._results  # ignores count entirely

    client = _OverReturningClient(
        [_FakeResult(url=f"https://example.test/{i}") for i in range(20)]
    )
    llm = _FakeLLM(["answer"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=client,
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].sources_added == 5


# ---------------------------------------------------------------------------
# Timeout: graceful fallback (sentinel + no content), bound respected.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_falls_back_gracefully() -> None:
    client = _HangingSearchClient()
    llm = _FakeLLM(["should not be used"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=client,
            max_results=5,
            timeout_seconds=0.05,  # tiny → wait_for raises TimeoutError
        )
    )

    # Exactly one fallback sentinel, and it is the final yielded item.
    assert isinstance(events[-1], LiveSearchUnavailable)
    assert events[-1].reason == "no_live_results"
    # No answer content was streamed.
    assert not any(isinstance(e, SynthesisProgressEvent) for e in events)
    # The search was attempted (bounded), not skipped.
    assert client.call_count == 1


# ---------------------------------------------------------------------------
# Empty results / no client: graceful fallback.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_results_falls_back() -> None:
    client = _FakeSearchClient([])
    llm = _FakeLLM(["unused"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=client,
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    assert isinstance(events[-1], LiveSearchUnavailable)
    assert not any(isinstance(e, SynthesisProgressEvent) for e in events)


@pytest.mark.asyncio
async def test_no_search_client_falls_back_without_search() -> None:
    llm = _FakeLLM(["unused"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=None,
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], LiveSearchUnavailable)
    assert events[0].reason == "no_web_search_client"


@pytest.mark.asyncio
async def test_search_error_falls_back() -> None:
    class _BoomClient:
        async def search(
            self, query: str, *, count: int = 10, freshness: str | None = None
        ) -> list[_FakeResult]:
            raise RuntimeError("backend exploded")

    llm = _FakeLLM(["unused"])

    events = await _collect(
        stream_live_search_answer(
            query="lookup",
            conversation_history=[],
            chat_id=uuid4(),
            llm=llm,  # type: ignore[arg-type]
            web_search_client=_BoomClient(),
            max_results=5,
            timeout_seconds=5.0,
        )
    )

    assert isinstance(events[-1], LiveSearchUnavailable)
    assert not any(isinstance(e, SynthesisProgressEvent) for e in events)
