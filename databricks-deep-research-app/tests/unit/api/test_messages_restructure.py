"""Unit tests for the structured-output restructure endpoint + background fn.

Covers: 202 + task scheduling, subset marking, 404 (non-owner / no envelope),
400 (unknown slot), 409 (pending + recent), and the background function's
never-stuck-pending guarantee.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from deep_research.agent.structured_evidence import RunArtifacts
from deep_research.core.auth import UserIdentity
from deep_research.core.deps import get_chat_service
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity

pytestmark = pytest.mark.unit

_MOD = "deep_research.api.v1.messages"
_PERSIST = (
    "deep_research.agent.persistence.update_structured_output_independent"
)


@pytest.fixture
def mock_user() -> UserIdentity:
    return UserIdentity(
        user_id="test-user-123", email="t@example.com", display_name="T"
    )


@pytest.fixture
def mock_chat_service(mock_user: UserIdentity) -> MagicMock:
    svc = MagicMock()
    chat = MagicMock()
    chat.user_id = mock_user.user_id
    svc.get_for_user = AsyncMock(return_value=chat)
    return svc


@pytest.fixture
def client(mock_user: UserIdentity, mock_chat_service: MagicMock) -> Any:
    async def _override_user() -> UserIdentity:
        return mock_user

    app.dependency_overrides[get_current_user_identity] = _override_user
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _envelope(
    *,
    generated_at: str | None = None,
    slots: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": 2,
        "binding": "run",
        "agent_id": str(uuid4()),
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "data": {"comparison": [], "key_findings": []},
        "meta": {
            "slots": slots
            or {
                "comparison": {"status": "ok"},
                "key_findings": {"status": "ok"},
            },
            "sources": [],
        },
    }


def _artifacts(envelope: dict[str, Any] | None) -> RunArtifacts:
    return RunArtifacts(
        report="a report",
        claims=[],
        sources=[],
        research_session_id=uuid4(),
        envelope=envelope,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_restructure_all_slots_202_and_schedules_task(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    persist = AsyncMock(return_value=True)
    bg = AsyncMock()
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(_envelope()))),
        patch(_PERSIST, persist),
        patch(f"{_MOD}._restructure_in_background", bg),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert sorted(body["slots"]) == ["comparison", "key_findings"]

    # Both slots were marked pending in the persisted envelope.
    persist.assert_awaited_once()
    marked = persist.await_args.kwargs["envelope"]
    assert marked["meta"]["slots"]["comparison"] == {"status": "pending"}
    assert marked["meta"]["slots"]["key_findings"] == {"status": "pending"}

    # The background task was scheduled with the requested slots.
    bg.assert_awaited_once()
    assert bg.await_args.kwargs["requested"] == {"comparison", "key_findings"}


def test_restructure_subset_marks_only_requested(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    persist = AsyncMock(return_value=True)
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(_envelope()))),
        patch(_PERSIST, persist),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={"slots": ["comparison"]},
        )
    assert resp.status_code == 202
    marked = persist.await_args.kwargs["envelope"]
    assert marked["meta"]["slots"]["comparison"] == {"status": "pending"}
    assert marked["meta"]["slots"]["key_findings"] == {"status": "ok"}


def test_restructure_unknown_slot_400(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(_envelope()))),
        patch(_PERSIST, AsyncMock()),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={"slots": ["ghost"]},
        )
    assert resp.status_code == 400


def test_restructure_no_envelope_404(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(None))),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={},
        )
    assert resp.status_code == 404


def test_restructure_missing_run_404(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=None)),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={},
        )
    assert resp.status_code == 404


def test_restructure_not_owner_404(
    client: Any, mock_chat_service: MagicMock
) -> None:
    mock_chat_service.get_for_user = AsyncMock(return_value=None)
    chat_id, message_id = uuid4(), uuid4()
    with patch(f"{_MOD}._restructure_in_background", AsyncMock()):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={},
        )
    assert resp.status_code == 404


def test_restructure_conflict_when_pending_and_recent_409(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    envelope = _envelope(
        generated_at=datetime.now(UTC).isoformat(),
        slots={
            "comparison": {"status": "pending"},
            "key_findings": {"status": "ok"},
        },
    )
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(envelope))),
        patch(_PERSIST, AsyncMock()),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={"slots": ["comparison"]},
        )
    assert resp.status_code == 409


def test_restructure_stale_pending_is_allowed(client: Any) -> None:
    chat_id, message_id = uuid4(), uuid4()
    old = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    envelope = _envelope(
        generated_at=old,
        slots={
            "comparison": {"status": "pending"},
            "key_findings": {"status": "ok"},
        },
    )
    with (
        patch(f"{_MOD}.load_run_artifacts", AsyncMock(return_value=_artifacts(envelope))),
        patch(_PERSIST, AsyncMock()),
        patch(f"{_MOD}._restructure_in_background", AsyncMock()),
    ):
        resp = client.post(
            f"/api/v1/chats/{chat_id}/messages/{message_id}/restructure",
            json={"slots": ["comparison"]},
        )
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Background function — never-stuck-pending guarantee
# ---------------------------------------------------------------------------


async def test_background_runs_structure_and_update_for_runnable() -> None:
    from deep_research.api.v1.messages import _restructure_in_background

    marked = _envelope(
        slots={"comparison": {"status": "pending"}, "key_findings": {"status": "ok"}}
    )
    surface = {"components": [], "bindings": []}
    resolved = SimpleNamespace(slots={"comparison": object()})
    structure_mock = AsyncMock(return_value=marked)
    with (
        patch(
            "deep_research.agent.structured_surface.load_agent_surface",
            AsyncMock(return_value=(surface, "W/etag")),
        ),
        patch(
            "deep_research.surface.output_schema.collect_output_slots",
            return_value=MagicMock(),
        ),
        patch(
            "deep_research.surface.output_schema.resolve_binding_for_run",
            return_value=resolved,
        ),
        patch(
            "deep_research.agent.structured_surface.structure_and_update",
            structure_mock,
        ),
    ):
        await _restructure_in_background(
            chat_id=uuid4(),
            message_id=uuid4(),
            user_id="u",
            requested={"comparison"},
            storage_stack=None,
            llm=None,
            marked_envelope=marked,
            artifacts=_artifacts(marked),
        )
    structure_mock.assert_awaited_once()
    assert structure_mock.await_args.kwargs["only_slots"] == {"comparison"}
    assert structure_mock.await_args.kwargs["surface_etag"] == "W/etag"


async def test_background_writes_failed_when_surface_gone() -> None:
    from deep_research.api.v1.messages import _restructure_in_background

    marked = _envelope(slots={"comparison": {"status": "pending"}})
    persist = AsyncMock(return_value=True)
    with (
        patch(
            "deep_research.agent.structured_surface.load_agent_surface",
            AsyncMock(return_value=None),
        ),
        patch(_PERSIST, persist),
    ):
        await _restructure_in_background(
            chat_id=uuid4(),
            message_id=uuid4(),
            user_id="u",
            requested={"comparison"},
            storage_stack=None,
            llm=None,
            marked_envelope=marked,
            artifacts=_artifacts(marked),
        )
    persist.assert_awaited_once()
    written = persist.await_args.kwargs["envelope"]
    assert written["meta"]["slots"]["comparison"]["status"] == "failed"


async def test_background_exception_writes_failed_stub() -> None:
    from deep_research.api.v1.messages import _restructure_in_background

    marked = _envelope(slots={"comparison": {"status": "pending"}})
    resolved = SimpleNamespace(slots={"comparison": object()})
    persist = AsyncMock(return_value=True)
    with (
        patch(
            "deep_research.agent.structured_surface.load_agent_surface",
            AsyncMock(return_value=({}, None)),
        ),
        patch(
            "deep_research.surface.output_schema.collect_output_slots",
            return_value=MagicMock(),
        ),
        patch(
            "deep_research.surface.output_schema.resolve_binding_for_run",
            return_value=resolved,
        ),
        patch(
            "deep_research.agent.structured_surface.structure_and_update",
            AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch(_PERSIST, persist),
    ):
        await _restructure_in_background(
            chat_id=uuid4(),
            message_id=uuid4(),
            user_id="u",
            requested={"comparison"},
            storage_stack=None,
            llm=None,
            marked_envelope=marked,
            artifacts=_artifacts(marked),
        )
    persist.assert_awaited_once()
    written = persist.await_args.kwargs["envelope"]
    assert written["meta"]["slots"]["comparison"]["status"] == "failed"
    assert "boom" in written["meta"]["slots"]["comparison"]["error"]
