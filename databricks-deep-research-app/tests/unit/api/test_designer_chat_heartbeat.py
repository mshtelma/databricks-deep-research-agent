"""Tests for the designer-chat SSE rendering layer (route side of the
async-dispatch + reconnect fix for the "Network error").

The turn now runs decoupled in ``designer_turn_registry`` (buffer / producer /
resume / idle-grace semantics are covered in test_turn_registry.py). Here we test
the thin route layer:

* ``_format_sse`` — sequence tagging via the SSE ``id:`` line.
* ``_sse_render`` — maps buffered chunks → SSE frames (``:keepalive`` comment for
  an idle gap, ``id:``-tagged event frames otherwise), and the CRITICAL property
  that closing a reader does NOT cancel the producer (the turn must survive for a
  reconnect — the inverse of the old inline ``finally`` cancel).

Lives under tests/unit/api because importing ``deep_research.api.v1.agent_designer``
eagerly builds Settings — the package conftest here supplies the test env.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from deep_research.agent_designer.sse_events import (
    DesignerSSEEvent,
    DoneEvent,
    MessageEvent,
)
from deep_research.agent_designer.turn_registry import DesignerTurnRegistry
from deep_research.api.v1 import agent_designer as api


def _noop(_exc: BaseException) -> None:
    pass


def test_format_sse_without_seq_has_no_id_line() -> None:
    frame = api._format_sse("turn_started", {"turn_id": "abc"})
    assert frame == 'event: turn_started\ndata: {"turn_id": "abc"}\n\n'
    assert "id:" not in frame


def test_format_sse_with_seq_prepends_id_line() -> None:
    frame = api._format_sse("message", {"content": "hi"}, seq=7)
    assert frame.startswith("id: 7\n")
    assert "event: message\n" in frame
    assert '"content": "hi"' in frame
    assert frame.endswith("\n\n")


async def test_sse_render_emits_id_tagged_events_and_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)
    monkeypatch.setattr(api, "designer_turn_registry", reg)
    monkeypatch.setattr(api, "DESIGNER_CHAT_HEARTBEAT_SECONDS", 0.05)
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="hi")
        await release.wait()  # silent gap >> heartbeat → keepalives
        yield DoneEvent()

    turn = reg.start(owner_user_id="u", session_id="s", producer=producer(), on_error=_noop)
    frames: list[str] = []

    async def read() -> None:
        async for frame in api._sse_render(turn, start_seq=0):
            frames.append(frame)
            if "event: done" in frame:
                break

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.2)
    assert any(f == ":keepalive\n\n" for f in frames), "expected keepalive during the stall"
    release.set()
    await asyncio.wait_for(reader, timeout=2)

    joined = "".join(frames)
    assert "id: 0\nevent: message" in joined  # first event is seq 0, id-tagged
    assert "event: done" in joined


async def test_sse_render_close_does_not_cancel_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the reader (client disconnect) must leave the producer running so a
    reconnect can resume — the opposite of the old inline-cancel behaviour."""
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)
    monkeypatch.setattr(api, "designer_turn_registry", reg)
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        await release.wait()
        yield DoneEvent()

    turn = reg.start(owner_user_id="u", session_id="s", producer=producer(), on_error=_noop)
    agen = api._sse_render(turn, start_seq=0)
    first = await agen.__anext__()
    assert "event: message" in first
    await agen.aclose()  # reader disconnect — must NOT cancel the producer

    release.set()
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)
    assert turn.status == "done"
    assert [e.type for e in turn.events] == ["message", "done"]
