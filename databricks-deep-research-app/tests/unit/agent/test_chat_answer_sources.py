"""Regression tests for the follow-up "chat about results" source loader.

The follow-up pool used to read prior sources via ``select(Source)`` on the
legacy ``sources`` table. That table is DROPPED on event-sourced deployments
(``cleanup_legacy_tables.sql``), so the query raised ``UndefinedTableError`` →
``FOLLOWUP_POOL_INIT_FAILED`` → follow-ups silently lost source grounding.

The fix threads the event-sourced ``storage_stack`` into the pool so sources are
read from ``ChatState.sources[]`` (the store the synthesizer writes to). These
tests assert the cached read works, never touches SQL when the cache serves the
sources, and still falls back to SQL only on a cache error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatState,
)
from deep_research.storage.documents import Source as DocSource

pytestmark = pytest.mark.unit


class _ExplodingSession:
    """AsyncSession stand-in that fails if the dropped-table SQL path is taken."""

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "SQL path must not run when storage_stack serves the sources"
        )


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return list(self._rows)


class _SqlSession:
    """AsyncSession stand-in that records the SQL fallback being used."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows
        self.executed = False

    async def execute(self, *args: Any, **kwargs: Any) -> _FakeResult:
        self.executed = True
        return _FakeResult(self._rows)


def _fake_stack(doc: ChatDocument) -> Any:
    """Minimal storage_stack exposing only ``cache.get`` (what the pool uses)."""
    return SimpleNamespace(cache=SimpleNamespace(get=AsyncMock(return_value=doc)))


def _raising_stack() -> Any:
    async def _raise(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("cache boom")

    return SimpleNamespace(cache=SimpleNamespace(get=_raise))


def _doc(chat_id: Any, specs: list[tuple[str, str, str, str, float | None]]) -> ChatDocument:
    sources = [
        DocSource(
            url=url,
            title=title,
            source_type="web",
            metadata={"snippet": snippet, "content": content, "relevance_score": score},
        )
        for (url, title, snippet, content, score) in specs
    ]
    return ChatDocument(
        meta=ChatMeta(chat_id=chat_id, user_id="u1"),
        state=ChatState(sources=sources),
    )


async def test_get_all_sources_reads_event_store_sorted_and_skips_sql() -> None:
    from deep_research.services.chat_source_pool_service import ChatSourcePoolService

    chat_id = uuid4()
    doc = _doc(
        chat_id,
        [
            ("https://b.example", "Beta", "snip b", "content b", 0.2),
            ("https://a.example", "Alpha", "snip a", "content a", 0.9),
        ],
    )
    pool = ChatSourcePoolService(_ExplodingSession(), storage_stack=_fake_stack(doc))  # type: ignore[arg-type]

    out = await pool.get_all_sources(chat_id)

    # Relevance desc; mapped from DocSource.metadata onto the Source surface.
    assert [s.url for s in out] == ["https://a.example", "https://b.example"]
    assert out[0].title == "Alpha"
    assert out[0].snippet == "snip a"
    assert out[0].content == "content a"
    assert out[0].relevance_score == 0.9


async def test_get_all_sources_empty_doc_returns_empty_without_sql() -> None:
    from deep_research.services.chat_source_pool_service import ChatSourcePoolService

    chat_id = uuid4()
    doc = ChatDocument(meta=ChatMeta(chat_id=chat_id, user_id="u1"), state=ChatState())
    pool = ChatSourcePoolService(_ExplodingSession(), storage_stack=_fake_stack(doc))  # type: ignore[arg-type]

    # No sources in the doc → [] WITHOUT falling through to the dropped table.
    assert await pool.get_all_sources(chat_id) == []


async def test_get_all_sources_falls_back_to_sql_only_on_cache_error() -> None:
    from deep_research.services.chat_source_pool_service import ChatSourcePoolService

    chat_id = uuid4()
    sentinel = [SimpleNamespace(url="https://sql.example", title="from sql")]
    session = _SqlSession(sentinel)
    pool = ChatSourcePoolService(session, storage_stack=_raising_stack())  # type: ignore[arg-type]

    out = await pool.get_all_sources(chat_id)

    assert session.executed is True  # cache raised → SQL fallback engaged
    assert out == sentinel


async def test_load_chat_sources_builds_pool_and_sourceinfo_from_event_store() -> None:
    from deep_research.agent.followup.chat_answer import _load_chat_sources

    chat_id = uuid4()
    doc = _doc(chat_id, [("https://a.example", "Alpha", "snip a", "content a", 0.9)])

    pool, sources = await _load_chat_sources(
        _ExplodingSession(),  # type: ignore[arg-type]
        chat_id,
        None,
        _fake_stack(doc),
    )

    assert pool is not None
    assert pool.has_index  # index built without touching SQL
    assert [s.url for s in sources] == ["https://a.example"]
    assert sources[0].snippet == "snip a"
    assert sources[0].content == "content a"
