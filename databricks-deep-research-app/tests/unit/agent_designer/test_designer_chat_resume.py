"""Endpoint tests for the designer-chat async dispatch + reconnect (POST /chat
and the new GET /chat/{turn_id}/events resume route).

Calls the route handlers directly (mirroring test_api_chat_assets.py) with a fake
orchestrator + a per-test in-memory registry monkeypatched into the module, so no
HTTP server / DB is required.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from fastapi import HTTPException

from deep_research.agent_designer.sse_events import (
    DesignerSSEEvent,
    DoneEvent,
    MessageEvent,
)
from deep_research.agent_designer.turn_registry import DesignerTurnRegistry
from deep_research.api.v1 import agent_designer as api
from deep_research.core.auth import UserIdentity


class _FakeRequest:
    def __init__(self) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(llm_client=object()))
        self.state = SimpleNamespace(obo_token="obo-token")
        self.headers: dict[str, str] = {}


class _FakeOrchestrator:
    def __init__(self, _llm: Any, _discovery: Any) -> None:
        pass

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None = None,
        assets: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        return messages

    def check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[Any] | None = None,
    ) -> None:
        return None

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        session_id: str | None,
        user_token: str,
        current_user_id: str = "",
        assets: list[Any] | None = None,
        skill_names: list[str] | None = None,
    ) -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="hi")
        yield DoneEvent()


def _user(user_id: str = "designer-user") -> UserIdentity:
    return UserIdentity(user_id=user_id, email=f"{user_id}@x.com", display_name=user_id)


def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "DiscoveryService", lambda: object())
    monkeypatch.setattr(
        "deep_research.core.trace_provenance.set_trace_provenance",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "deep_research.agent_designer.orchestrator.DesignerChatOrchestrator",
        _FakeOrchestrator,
    )


async def _drain(response: Any) -> list[str]:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return parts


def _request(**overrides: Any) -> Any:
    body = {
        "messages": [{"role": "user", "content": "build an agent"}],
        "current_ast": None,
        "session_id": "sess-1",
        "assets": [],
    }
    body.update(overrides)
    return api.ChatRequest.model_validate(body)


async def test_chat_emits_turn_started_first_then_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "designer_turn_registry", DesignerTurnRegistry(idle_grace_seconds=100.0))
    _patch_orchestrator(monkeypatch)

    response = await api.chat(_request(), _user(), _FakeRequest())  # type: ignore[arg-type]
    frames = await _drain(response)
    joined = "".join(frames)

    # First frame is the turn_started handshake carrying the turn_id (no id:).
    assert frames[0].startswith("event: turn_started\n")
    turn_id = json.loads(frames[0].split("data: ", 1)[1].strip())["turn_id"]
    assert turn_id
    # Buffered events follow, id-tagged from seq 0, terminated by done.
    assert "id: 0\nevent: message" in joined
    assert "event: done" in joined
    assert joined.index("turn_started") < joined.index("event: message")


async def test_chat_per_user_cap_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = DesignerTurnRegistry(per_user_max=1, idle_grace_seconds=100.0)
    monkeypatch.setattr(api, "designer_turn_registry", reg)
    _patch_orchestrator(monkeypatch)
    release = asyncio.Event()

    async def long_producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="busy")
        await release.wait()
        yield DoneEvent()

    held = reg.start(
        owner_user_id="designer-user", session_id="s", producer=long_producer(), on_error=lambda _e: None
    )
    try:
        with pytest.raises(HTTPException) as exc_info:
            await api.chat(_request(), _user(), _FakeRequest())  # type: ignore[arg-type]
        assert exc_info.value.status_code == 429
    finally:
        release.set()
        assert held._task is not None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(held._task, timeout=2)


async def test_chat_events_resume_from_since(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)
    monkeypatch.setattr(api, "designer_turn_registry", reg)

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield MessageEvent(content="a")  # seq 0
        yield MessageEvent(content="b")  # seq 1
        yield DoneEvent()  # seq 2

    turn = reg.start(owner_user_id="owner", session_id="s", producer=producer(), on_error=lambda _e: None)
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)

    response = await api.chat_events(turn.turn_id, _user("owner"), since=1)  # type: ignore[arg-type]
    joined = "".join(await _drain(response))

    assert "id: 1\nevent: message" in joined  # resumes at seq 1
    assert "id: 0" not in joined  # seq 0 not replayed
    assert "event: done" in joined


async def test_chat_events_404_for_foreign_user_and_unknown_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = DesignerTurnRegistry(idle_grace_seconds=100.0)
    monkeypatch.setattr(api, "designer_turn_registry", reg)

    async def producer() -> AsyncIterator[DesignerSSEEvent]:
        yield DoneEvent()

    turn = reg.start(owner_user_id="owner", session_id="s", producer=producer(), on_error=lambda _e: None)
    assert turn._task is not None
    await asyncio.wait_for(turn._task, timeout=2)

    with pytest.raises(HTTPException) as foreign:
        await api.chat_events(turn.turn_id, _user("intruder"), since=0)  # type: ignore[arg-type]
    assert foreign.value.status_code == 404

    with pytest.raises(HTTPException) as unknown:
        await api.chat_events("does-not-exist", _user("owner"), since=0)  # type: ignore[arg-type]
    assert unknown.value.status_code == 404
