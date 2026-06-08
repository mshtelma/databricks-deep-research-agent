from __future__ import annotations

import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from deep_research.agent_designer.sse_events import DesignerSSEEvent, DoneEvent
from deep_research.api.v1 import agent_designer as api
from deep_research.core.auth import UserIdentity


class _FakeRequest:
    def __init__(self) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(llm_client=object()))
        self.state = SimpleNamespace(obo_token="obo-token")
        self.headers: dict[str, str] = {}


class _FakeOrchestrator:
    instances: list[_FakeOrchestrator] = []

    def __init__(self, _llm: Any, _discovery: Any) -> None:
        self.check_limits_calls: list[dict[str, Any]] = []
        self.run_turn_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    def check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[Any] | None = None,
    ) -> None:
        self.check_limits_calls.append(
            {"messages": messages, "current_ast": current_ast, "assets": assets}
        )

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        session_id: str | None,
        user_token: str,
        current_user_id: str = "",
        assets: list[Any] | None = None,
    ) -> AsyncIterator[DesignerSSEEvent]:
        self.run_turn_calls.append(
            {
                "messages": messages,
                "current_ast": current_ast,
                "session_id": session_id,
                "user_token": user_token,
                "current_user_id": current_user_id,
                "assets": assets,
            }
        )
        yield DoneEvent()


@pytest.mark.asyncio
async def test_chat_passes_assets_to_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOrchestrator.instances.clear()
    monkeypatch.setattr(api, "DiscoveryService", lambda: object())
    monkeypatch.setattr(
        "deep_research.core.trace_provenance.set_trace_provenance",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "deep_research.agent_designer.orchestrator.DesignerChatOrchestrator",
        _FakeOrchestrator,
    )

    assets = [
        {
            "kind": "delta_table",
            "full_name": "main.officeqa_benchmark.treasury_tables",
            "usage": "required",
            "field_roles": {"primary_key": "chunk_id", "content": "content"},
            "metadata": {"warehouse_id": "warehouse-123"},
        }
    ]
    request = api.ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "build an OfficeQA agent"}],
            "current_ast": None,
            "session_id": "asset-session",
            "assets": assets,
        }
    )
    user = UserIdentity(
        user_id="designer-user",
        email="designer@example.com",
        display_name="Designer User",
    )

    response = await api.chat(request, user, _FakeRequest())  # type: ignore[arg-type]
    body_parts: list[str] = []
    async for chunk in response.body_iterator:
        body_parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

    assert "event: done" in "".join(body_parts)
    orchestrator = _FakeOrchestrator.instances[0]
    checked_assets = orchestrator.check_limits_calls[0]["assets"]
    run_assets = orchestrator.run_turn_calls[0]["assets"]
    assert checked_assets == request.assets
    assert run_assets == request.assets
    assert run_assets[0].full_name == "main.officeqa_benchmark.treasury_tables"
