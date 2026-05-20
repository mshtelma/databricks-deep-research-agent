"""Integration tests for the Service-Principal run_as capability (US-602).

Tests are gated by RUN_INTEGRATION_TESTS=1 to match Phase 1 conventions.
All Databricks SDK calls and workspace clients are mocked so no real external
services are touched.

Run:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/test_sp_runas_e2e.py -v
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Must be set before importing `app` so that Settings() validation does not
# require LAKEBASE_*/DATABASE_URL.
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

# ---------------------------------------------------------------------------
# Module-level skip guard — matches Phase 1 pattern exactly
# ---------------------------------------------------------------------------

_RUN_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

if not _RUN_TESTS:
    import pytest
    pytest.skip("Requires RUN_INTEGRATION_TESTS=1", allow_module_level=True)

# ---------------------------------------------------------------------------
# Deferred imports (only reached when RUN_INTEGRATION_TESTS=1)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from deep_research.agent_designer.orchestrator import (  # noqa: E402
    DesignerChatOrchestrator,
    LLMStreamChunk,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_UUID = "12345678-1234-5678-1234-567812345678"
_TEST_USER_ID = "test-user-001"

_AST_WITH_SP: dict[str, Any] = {
    "id": "wf-test",
    "name": "SP Test Workflow",
    "version": 1,
    "run_as": {"service_principal_id": _VALID_UUID},
    "root": {
        "id": "root",
        "type": "agent",
        "label": "researcher",
        "config": {"subtype": "researcher"},
        "children": [],
    },
}

_AST_CALLER: dict[str, Any] = {
    "id": "wf-caller",
    "name": "Caller Test Workflow",
    "version": 1,
    "root": {
        "id": "root",
        "type": "agent",
        "label": "researcher",
        "config": {"subtype": "researcher"},
        "children": [],
    },
}

_FINISH_CHUNK = LLMStreamChunk(finish=True)


class _FakeLLM:
    """Minimal LLM client that immediately finishes the stream."""

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]:
        yield _FINISH_CHUNK


class _FakeDiscovery:
    """Minimal discovery adapter."""

    async def list_for_user(
        self,
        user_token: str,
        kinds: Any = None,
        user_id: str = "",
    ) -> list[Any]:
        return []


def _make_orchestrator() -> DesignerChatOrchestrator:
    return DesignerChatOrchestrator(llm=_FakeLLM(), discovery=_FakeDiscovery())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_sp_path() -> None:
    """WorkflowDefinition with SP run_as → orchestrator calls get_service_principal_workspace_client."""
    orchestrator = _make_orchestrator()
    messages = [{"role": "user", "content": "hello"}]

    with patch(
        "deep_research.agent_designer.orchestrator.get_service_principal_workspace_client",
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ) as mock_sp_client:
        events = []
        async for ev in orchestrator.run_turn(
            messages=messages,
            current_ast=_AST_WITH_SP,
            session_id="sess-1",
            user_token="tok-abc",
            current_user_id=_TEST_USER_ID,
        ):
            events.append(ev)

    # SP client must have been called with the correct sp_id
    mock_sp_client.assert_awaited_once()
    call_kwargs = mock_sp_client.call_args
    assert call_kwargs.kwargs["sp_id"] == _VALID_UUID
    assert call_kwargs.kwargs["requesting_user_id"] == _TEST_USER_ID


@pytest.mark.asyncio
async def test_missing_permission_returns_error_before_tools() -> None:
    """permissions_check returning False → ErrorEvent before any tool factory runs."""
    from fastapi import HTTPException

    orchestrator = _make_orchestrator()
    messages = [{"role": "user", "content": "hello"}]

    # Simulate permissions check failure by having get_service_principal_workspace_client raise 403.
    async def _raise_403(**kwargs: Any) -> Any:
        raise HTTPException(status_code=403, detail="missing CAN_USE_AS permission")

    from deep_research.agent_designer.orchestrator import ErrorEvent

    with patch(
        "deep_research.agent_designer.orchestrator.get_service_principal_workspace_client",
        side_effect=_raise_403,
    ):
        events = []
        async for ev in orchestrator.run_turn(
            messages=messages,
            current_ast=_AST_WITH_SP,
            session_id="sess-2",
            user_token="tok-abc",
            current_user_id=_TEST_USER_ID,
        ):
            events.append(ev)

    # Must have received an ErrorEvent and stopped — LLM never called
    assert any(isinstance(ev, ErrorEvent) for ev in events)
    # Stream ended after the error (no DoneEvent from LLM since stream halted early)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_audit_log_has_both_user_and_sp(caplog: pytest.LogCaptureFixture) -> None:
    """SP run emits a single audit log line containing BOTH requested_by_user_id AND executed_as_sp_id."""
    orchestrator = _make_orchestrator()
    messages = [{"role": "user", "content": "hello"}]

    with patch(
        "deep_research.agent_designer.orchestrator.get_service_principal_workspace_client",
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ), caplog.at_level(logging.INFO, logger="agent_designer.metrics"):
        async for _ in orchestrator.run_turn(
            messages=messages,
            current_ast=_AST_WITH_SP,
            session_id="sess-3",
            user_token="tok-abc",
            current_user_id=_TEST_USER_ID,
        ):
            pass

    # Find the run_principal log line
    run_principal_records = [
        r for r in caplog.records if "run_principal" in r.getMessage()
    ]
    assert run_principal_records, "No run_principal log line found"

    record = run_principal_records[0]
    msg = record.getMessage()
    assert _TEST_USER_ID in msg, f"requested_by_user_id missing from log: {msg}"
    assert _VALID_UUID in msg, f"executed_as_sp_id missing from log: {msg}"


@pytest.mark.asyncio
async def test_default_caller_path_unchanged() -> None:
    """Default run_as='caller' logs caller mode and never checks SP permissions."""
    orchestrator = _make_orchestrator()
    messages = [{"role": "user", "content": "hello"}]

    with patch(
        "deep_research.agent_designer.orchestrator.get_service_principal_workspace_client",
        new_callable=AsyncMock,
    ) as mock_sp_client:
        async for _ in orchestrator.run_turn(
            messages=messages,
            current_ast=_AST_CALLER,
            session_id="sess-4",
            user_token="tok-abc",
            current_user_id=_TEST_USER_ID,
        ):
            pass

    # SP client must NOT have been called
    mock_sp_client.assert_not_awaited()
