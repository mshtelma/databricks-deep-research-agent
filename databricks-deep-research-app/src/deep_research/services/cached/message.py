"""Cache-backed `IMessageService`.

Storage model: `ChatState.messages[]` — every method is chat-scoped. The
cached impl keeps messages in the document alongside the research sessions
that reference them (single source of truth for a chat). Legacy callers
that used the global `messages` table are updated to pass `chat_id`
through; routes already have it in the URL path.

Return shape: `SimpleNamespace` mirroring the legacy `Message` ORM
attribute surface (`id`, `chat_id`, `role`, `content`, `created_at`,
`updated_at`, `is_edited`, `research_session_id`). Callers relying on
SQLAlchemy relationship traversal must migrate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IMessageService
from deep_research.storage.documents import Message as DocMessage

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)


def _role_value(role: Any) -> str:
    return getattr(role, "value", role)


def _msg_to_namespace(m: DocMessage, chat_id: UUID) -> SimpleNamespace:
    meta = dict(m.metadata or {})
    return SimpleNamespace(
        id=m.id,
        chat_id=chat_id,
        role=m.role,
        content=m.content,
        created_at=m.ts,
        updated_at=meta.get("updated_at", m.ts),
        is_edited=bool(meta.get("is_edited", False)),
        research_session_id=meta.get("research_session_id"),
        metadata=meta,
    )


class CachedMessageService(_CachedServiceBase, IMessageService):
    """`IMessageService` via `ChatState.messages[]`."""

    def __init__(self, stack: StorageStack) -> None:
        super().__init__(stack)

    async def create(
        self,
        chat_id: UUID,
        role: Any,
        content: str,
    ) -> SimpleNamespace:
        now = datetime.now(UTC)
        msg = DocMessage(
            role=_role_value(role),
            content=content,
            ts=now,
            metadata={},
        )

        def _apply(doc: Any) -> None:
            doc.state.add_message(msg)

        await self._mutate_chat(chat_id, _apply, dirty="both")
        logger.info("MESSAGE_CREATED chat_id=%s role=%s id=%s", chat_id, msg.role, msg.id)
        return _msg_to_namespace(msg, chat_id)

    async def get_with_chat(
        self,
        message_id: UUID,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        try:
            doc = await self._read_chat(chat_id)
        except ValueError:
            # Chat has never been hydrated and doesn't exist in backend.
            return None
        for m in doc.state.messages:
            if m.id == message_id:
                return _msg_to_namespace(m, chat_id)
        return None

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 100,
        offset: int = 0,
        before: datetime | None = None,
    ) -> tuple[list[SimpleNamespace], int]:
        doc = await self._read_chat(chat_id)
        msgs = list(doc.state.messages)
        if before is not None:
            msgs = [m for m in msgs if m.ts < before]
        total = len(msgs)
        slice_ = msgs[offset : offset + limit] if limit else msgs[offset:]
        return [_msg_to_namespace(m, chat_id) for m in slice_], total

    async def update_content(
        self,
        message_id: UUID,
        content: str,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        result: dict[str, DocMessage | None] = {"msg": None}

        def _apply(doc: Any) -> None:
            for m in doc.state.messages:
                if m.id == message_id:
                    m.content = content
                    now = datetime.now(UTC)
                    meta = dict(m.metadata or {})
                    meta["is_edited"] = True
                    meta["updated_at"] = now.isoformat()
                    m.metadata = meta
                    result["msg"] = m
                    return

        await self._mutate_chat(chat_id, _apply, dirty="state")
        m = result["msg"]
        if m is None:
            return None
        logger.info("MESSAGE_UPDATED chat_id=%s id=%s", chat_id, message_id)
        return _msg_to_namespace(m, chat_id)

    async def delete_subsequent(
        self,
        chat_id: UUID,
        after: datetime,
    ) -> int:
        counter: dict[str, int] = {"n": 0}

        def _apply(doc: Any) -> None:
            before = len(doc.state.messages)
            doc.state.messages = [m for m in doc.state.messages if m.ts <= after]
            counter["n"] = before - len(doc.state.messages)

        await self._mutate_chat(chat_id, _apply, dirty="state")
        n = counter["n"]
        if n:
            logger.info(
                "MESSAGES_TRUNCATED chat_id=%s deleted=%s after=%s",
                chat_id, n, after.isoformat(),
            )
        return n

    async def set_research_session(
        self,
        message_id: UUID,
        research_session_id: UUID,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        result: dict[str, DocMessage | None] = {"msg": None}

        def _apply(doc: Any) -> None:
            for m in doc.state.messages:
                if m.id == message_id:
                    meta = dict(m.metadata or {})
                    meta["research_session_id"] = str(research_session_id)
                    m.metadata = meta
                    result["msg"] = m
                    return

        await self._mutate_chat(chat_id, _apply, dirty="state")
        m = result["msg"]
        return _msg_to_namespace(m, chat_id) if m else None

    async def get_conversation_history(
        self,
        chat_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        doc = await self._read_chat(chat_id)
        msgs = [m for m in doc.state.messages if m.content]
        if limit and len(msgs) > limit:
            msgs = msgs[-limit:]
        return [{"role": _role_value(m.role), "content": m.content} for m in msgs]
