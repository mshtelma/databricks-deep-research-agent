"""Contract tests for ``CachedExportService`` (F-OTHER.5).

Exercises export_markdown, export_json, export_report_markdown,
export_provenance_markdown against the parametric ``stack`` fixture
from conftest.py (FakeBackend by default).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from deep_research.services.cached.export import CachedExportService

# ---------------------------------------------------------------------------
# Stub services
# ---------------------------------------------------------------------------


class _StubChatService:
    def __init__(self, chats: list) -> None:
        self._chats = chats

    async def get_for_user(self, chat_id: UUID, user_id: str):
        for c in self._chats:
            if c.id == chat_id and c.user_id == user_id:
                return c
        return None

    async def list(self, user_id: str, limit: int = 50, offset: int = 0, status=None, search=None):
        matching = [c for c in self._chats if c.user_id == user_id]
        return matching, len(matching)


class _StubMessageService:
    def __init__(self, messages_by_chat: dict) -> None:
        self._map = messages_by_chat

    async def list_messages(self, chat_id: UUID, limit: int = 100, offset: int = 0, before=None):
        msgs = self._map.get(chat_id, [])
        return msgs, len(msgs)


class _StubResearchSessionService:
    def __init__(self, sessions_by_message: dict) -> None:
        self._map = sessions_by_message

    async def get_by_message(self, message_id: UUID, *, chat_id: UUID):
        return self._map.get(message_id)


class _StubSourceService:
    def __init__(self, sources_by_session: dict) -> None:
        self._map = sources_by_session

    async def list_by_session(self, session_id: UUID, *, chat_id=None):
        return self._map.get(session_id, [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_role(value: str):
    return SimpleNamespace(value=value)


def _make_chat(user_id: str, title: str = "Test Chat") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        title=title,
        status=SimpleNamespace(value="active"),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_message(chat_id: UUID, role: str = "user", content: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        chat_id=chat_id,
        role=_make_role(role),
        content=content,
        created_at=datetime.now(UTC),
        is_edited=False,
        research_session_id=None,
    )


def _make_source(url: str, title: str = "Title") -> SimpleNamespace:
    return SimpleNamespace(url=url, title=title)


def _make_research_session(message_id: UUID, query: str = "test query", verification_data=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        message_id=message_id,
        query=query,
        research_depth="medium",
        verification_data=verification_data,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCachedExportServiceContract:
    """Export service — minimal chat lifecycle."""

    def _build_svc(self, chats, messages_by_chat, sessions_by_message=None, sources_by_session=None):
        return CachedExportService(
            chat_service=_StubChatService(chats),
            message_service=_StubMessageService(messages_by_chat),
            research_session_service=_StubResearchSessionService(sessions_by_message or {}),
            source_service=_StubSourceService(sources_by_session or {}),
        )

    @pytest.mark.asyncio
    async def test_export_markdown_basic(self) -> None:
        user_id = "user_a"
        chat = _make_chat(user_id, title="My Chat")
        msg_user = _make_message(chat.id, role="user", content="What is ML?")
        msg_agent = _make_message(chat.id, role="agent", content="ML is ...")

        svc = self._build_svc(
            chats=[chat],
            messages_by_chat={chat.id: [msg_user, msg_agent]},
        )
        result = await svc.export_markdown(chat.id, user_id)

        assert "# My Chat" in result
        assert "What is ML?" in result
        assert "ML is ..." in result

    @pytest.mark.asyncio
    async def test_export_markdown_not_found(self) -> None:
        svc = self._build_svc(chats=[], messages_by_chat={})
        with pytest.raises(ValueError, match="not found"):
            await svc.export_markdown(uuid4(), "user_x")

    @pytest.mark.asyncio
    async def test_export_json_shape(self) -> None:
        user_id = "user_b"
        chat = _make_chat(user_id, title="JSON Chat")
        msg = _make_message(chat.id, role="user", content="Hi")

        svc = self._build_svc(
            chats=[chat],
            messages_by_chat={chat.id: [msg]},
        )
        data = await svc.export_json(chat.id, user_id)

        assert data["id"] == str(chat.id)
        assert data["title"] == "JSON Chat"
        assert data["message_count"] == 1
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_export_report_markdown_not_found(self) -> None:
        """export_report_markdown raises ValueError if message not accessible."""
        svc = self._build_svc(chats=[], messages_by_chat={})
        with pytest.raises(ValueError, match="not found"):
            await svc.export_report_markdown(uuid4(), "user_z")

    @pytest.mark.asyncio
    async def test_export_report_markdown_with_session(self) -> None:
        user_id = "user_c"
        chat = _make_chat(user_id)
        msg = _make_message(chat.id, role="agent", content="Report body.")
        rs = _make_research_session(msg.id, query="Deep dive into AI")

        svc = self._build_svc(
            chats=[chat],
            messages_by_chat={chat.id: [msg]},
            sessions_by_message={msg.id: rs},
        )
        result = await svc.export_report_markdown(msg.id, user_id)

        assert "Deep dive into AI" in result
        assert "Report body." in result
        assert "# Deep dive into AI" in result

    @pytest.mark.asyncio
    async def test_export_provenance_markdown_no_verification_data(self) -> None:
        user_id = "user_d"
        chat = _make_chat(user_id)
        msg = _make_message(chat.id, role="agent", content="x")
        rs = _make_research_session(msg.id, verification_data=None)

        svc = self._build_svc(
            chats=[chat],
            messages_by_chat={chat.id: [msg]},
            sessions_by_message={msg.id: rs},
        )
        result = await svc.export_provenance_markdown(msg.id, user_id)

        assert "Verification Report" in result
        assert "No claims found" in result

    @pytest.mark.asyncio
    async def test_export_provenance_markdown_with_claims(self) -> None:
        user_id = "user_e"
        chat = _make_chat(user_id)
        msg = _make_message(chat.id, role="agent", content="x")
        verification_data = {
            "summary": {
                "total_claims": 2,
                "supported_count": 1,
                "partial_count": 0,
                "unsupported_count": 1,
                "contradicted_count": 0,
            },
            "claims": [
                {
                    "claim_text": "Claim A is true.",
                    "verification_verdict": "supported",
                    "evidence": {
                        "source_title": "Source 1",
                        "source_url": "https://example.com",
                        "quote_text": "Evidence quote.",
                    },
                },
            ],
        }
        rs = _make_research_session(msg.id, verification_data=verification_data)

        svc = self._build_svc(
            chats=[chat],
            messages_by_chat={chat.id: [msg]},
            sessions_by_message={msg.id: rs},
        )
        result = await svc.export_provenance_markdown(msg.id, user_id)

        assert "Verification Report" in result
        assert "SUPPORTED" in result
        assert "Claim A is true." in result
        assert "Source 1" in result

    @pytest.mark.asyncio
    async def test_make_export_service_factory_cached(self, stack) -> None:
        """make_export_service returns CachedExportService under cached impl."""
        from unittest.mock import MagicMock

        from deep_research.services._impl_factory import make_export_service
        from deep_research.services._protocols import IExportService

        settings = MagicMock()
        settings.storage_service_impl = "cached"

        svc = make_export_service(settings, stack)
        assert isinstance(svc, IExportService)
