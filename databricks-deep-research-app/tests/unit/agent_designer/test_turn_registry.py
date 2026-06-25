"""Unit tests for the in-memory designer turn registry (async dispatch + resume).

Pure asyncio — no app settings / HTTP. ``asyncio_mode = "auto"`` runs the
``async def test_*`` functions directly.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from deep_research.agent_designer.sse_events import (
    DesignerSSEEvent,
    DoneEvent,
    MessageEvent,
)
from deep_research.agent_designer.turn_registry import (
    DesignerTurnRegistry,
    StreamChunk,
)


def _noop_error(_exc: BaseException) -> None:  # default on_error
    pass


async def _drain_stream(
    reg: DesignerTurnRegistry, turn, *, start_seq: int = 0, heartbeat: float = 10.0
) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    async for chunk in reg.stream(turn, start_seq=start_seq, heartbeat_seconds=heartbeat):
        chunks.append(chunk)
    return chunks


def _event_types(chunks: list[StreamChunk]) -> list[str]:
    return [c.event.type for c in chunks if c.event is not None]


def _seqs(chunks: list[StreamChunk]) -> list[int]:
    return [c.seq for c in chunks if c.event is not None]


# --------------------------------------------------------------------------- #
# Happy path: drain → done, stream all in order with contiguous seq.
# --------------------------------------------------------------------------- #
async def test_drain_to_done_and_stream_all() -> None:
    reg = DesignerTurnRegistry()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        yield MessageEvent(content="b")
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )
    chunks = await _drain_stream(reg, turn)
    assert _event_types(chunks) == ["message", "message", "done"]
    assert _seqs(chunks) == [0, 1, 2]
    assert turn.status == "done"


# Producer finishes BEFORE any reader attaches — no events lost (buffer + seq).
async def test_producer_outruns_reader() -> None:
    reg = DesignerTurnRegistry()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        yield MessageEvent(content="b")
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)  # fully drain first
    assert turn.status == "done"

    chunks = await _drain_stream(reg, turn)
    assert _event_types(chunks) == ["message", "message", "done"]
    assert _seqs(chunks) == [0, 1, 2]


# Resume from a sequence: replay events[since:] exactly, no dupes/gaps.
async def test_resume_from_seq() -> None:
    reg = DesignerTurnRegistry()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        yield MessageEvent(content="b")
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)

    chunks = await _drain_stream(reg, turn, start_seq=1)
    assert _event_types(chunks) == ["message", "done"]
    assert _seqs(chunks) == [1, 2]  # starts at 1, contiguous


# Idle buffer → keepalive emitted; real events still flow afterward
# (proves the wait_for/Condition cancellation does not corrupt the lock).
async def test_keepalive_then_events_flow() -> None:
    reg = DesignerTurnRegistry()
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        await release.wait()
        yield MessageEvent(content="hi")
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )
    chunks: list[StreamChunk] = []

    async def read() -> None:
        async for chunk in reg.stream(turn, start_seq=0, heartbeat_seconds=0.05):
            chunks.append(chunk)
            if chunk.event is not None and chunk.event.type == "done":
                break

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.2)  # several heartbeat windows while stalled
    assert any(c.event is None for c in chunks), "expected at least one keepalive"
    release.set()
    await asyncio.wait_for(reader, timeout=2)
    assert _event_types(chunks) == ["message", "done"]  # events flow after keepalives


# THE key property: a reader disconnect does NOT cancel the producer.
async def test_producer_survives_reader_cancel() -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)  # don't reclaim during test
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        await release.wait()
        yield MessageEvent(content="b")
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )

    async def read() -> None:
        async for _chunk in reg.stream(turn, start_seq=0, heartbeat_seconds=10):
            pass

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.05)  # reader drains seq 0, then blocks
    reader.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await reader

    release.set()  # producer keeps running independently
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)
    assert turn.status == "done"
    assert [e.type for e in turn.events] == ["message", "message", "done"]


# Authorization: a foreign user (or unknown id) cannot resolve the turn.
async def test_owner_check_blocks_foreign_user() -> None:
    reg = DesignerTurnRegistry()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="owner", session_id="s1", producer=producer(), on_error=_noop_error
    )
    assert reg.get(turn.turn_id, owner_user_id="owner") is turn
    assert reg.get(turn.turn_id, owner_user_id="intruder") is None
    assert reg.get("does-not-exist", owner_user_id="owner") is None


# Per-user concurrency count (route raises 429 above the cap).
async def test_active_count_for_user() -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)
    release = asyncio.Event()

    async def long_producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="x")
        await release.wait()
        yield DoneEvent()

    async def fast_producer() -> AsyncIterator[DesignerSSEEvent]:
        yield DoneEvent()

    t1 = reg.start(owner_user_id="A", session_id="s", producer=long_producer(), on_error=_noop_error)
    reg.start(owner_user_id="A", session_id="s", producer=long_producer(), on_error=_noop_error)
    assert reg.active_count_for_user("A") == 2
    assert reg.active_count_for_user("B") == 0

    fast = reg.start(
        owner_user_id="A", session_id="s", producer=fast_producer(), on_error=_noop_error
    )
    assert fast._task is not None
    await asyncio.wait_for(fast._task, timeout=2)  # terminal → not counted
    assert reg.active_count_for_user("A") == 2

    release.set()
    assert t1._task is not None
    await asyncio.wait_for(t1._task, timeout=2)


# Idle-grace reclaim cancels a running, reader-less turn.
async def test_idle_grace_cancels_readerless_turn() -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=0.05)
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        await release.wait()  # never released in this test
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )

    async def read_once() -> None:
        async for chunk in reg.stream(turn, start_seq=0, heartbeat_seconds=10):
            if chunk.event is not None and chunk.event.type == "message":
                break  # leave → reader detaches → refcount 0

    reader = asyncio.create_task(read_once())
    await asyncio.wait_for(reader, timeout=2)
    await asyncio.sleep(0.15)  # > idle grace
    assert turn.status == "cancelled"
    assert turn._task is not None and turn._task.done()


# A reconnect within the grace window cancels the reclaim — turn survives.
async def test_reattach_within_grace_keeps_turn_alive() -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=0.3)
    release = asyncio.Event()

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        await release.wait()
        yield DoneEvent()

    turn = reg.start(
        owner_user_id="u1", session_id="s1", producer=producer(), on_error=_noop_error
    )

    async def read_seq0() -> None:
        async for chunk in reg.stream(turn, start_seq=0, heartbeat_seconds=10):
            if chunk.event is not None and chunk.event.type == "message":
                break

    await asyncio.wait_for(asyncio.create_task(read_seq0()), timeout=2)  # detach → grace scheduled

    # Reconnect quickly (before 0.3s) — attach cancels the pending reclaim.
    reconnect = asyncio.create_task(_drain_stream(reg, turn, start_seq=1, heartbeat=10))
    await asyncio.sleep(0.05)
    release.set()
    chunks = await asyncio.wait_for(reconnect, timeout=2)
    assert turn.status == "done"  # NOT cancelled
    assert _event_types(chunks) == ["done"]  # resumed from seq 1 (the DoneEvent)


# Producer exception → sanitized error+done terminal in the buffer; on_error fired.
async def test_error_path_sanitized_terminal() -> None:
    reg = DesignerTurnRegistry()
    captured: list[BaseException] = []

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")
        raise RuntimeError("boom-internal-detail")

    turn = reg.start(
        owner_user_id="u1",
        session_id="s1",
        producer=producer(),
        on_error=captured.append,
    )
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)
    assert turn.status == "error"
    assert [e.type for e in turn.events] == ["message", "error", "done"]
    assert len(captured) == 1 and isinstance(captured[0], RuntimeError)
    error_event = turn.events[1]
    assert error_event.type == "error"
    assert "boom-internal-detail" not in error_event.message  # sanitized


# Sweep evicts terminal, reader-less, expired turns; spares running turns.
async def test_sweep_evicts_terminal_keeps_running() -> None:
    reg = DesignerTurnRegistry(ttl_seconds=0.0)
    release = asyncio.Event()

    async def fast() -> AsyncIterator[DesignerSSEEvent]:
        yield DoneEvent()

    async def slow() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="x")
        await release.wait()
        yield DoneEvent()

    done_turn = reg.start(owner_user_id="u", session_id="s", producer=fast(), on_error=_noop_error)
    running_turn = reg.start(owner_user_id="u", session_id="s", producer=slow(), on_error=_noop_error)
    assert done_turn._task is not None
    await asyncio.wait_for(done_turn._task, timeout=2)

    evicted = reg.sweep()
    assert evicted == 1
    assert reg.get(done_turn.turn_id, owner_user_id="u") is None
    assert reg.get(running_turn.turn_id, owner_user_id="u") is running_turn  # spared

    release.set()
    assert running_turn._task is not None
    await asyncio.wait_for(running_turn._task, timeout=2)
