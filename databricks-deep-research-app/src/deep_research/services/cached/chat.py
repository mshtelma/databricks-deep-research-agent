"""Cache-backed `IChatService` — routes chat CRUD through `StorageStack`.

Write pattern: create/update/soft_delete all go through `_mutate_chat` which
applies an in-memory delta under the per-chat `asyncio.Lock` and enqueues a
`WriteQueue` flush. Reads return immediately from the in-memory cache.

Return shape: every method that historically returned a legacy `Chat` ORM object
now returns a `ChatView` dataclass. `ChatView` exposes the same attribute names
that the 17 call sites rely on (`id`, `user_id`, `title`, `status`, `chat_type`,
`created_at`, `updated_at`, `deleted_at`, `incognito_session_id`, `messages`,
`sources`, `research_sessions`). Callers are unaffected.

Thread-safety: concurrent calls on the same `chat_id` are serialized by the
per-chat lock inside `_mutate_chat`. Different chat IDs are fully concurrent.
"""

from __future__ import annotations

import builtins
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.agent.chat_title import derive_chat_title_from_query
from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._list_cache import _UserScopedLRU
from deep_research.services._protocols import IChatService
from deep_research.services.storage.surface_state import merge_surface_state
from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatMetaEmbed,
    ChatState,
)

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

# Process-global list cache shared across all CachedChatService instances.
_CHAT_LIST_CACHE: _UserScopedLRU = _UserScopedLRU(ttl_sec=2.0, max_entries=10_000)


# --- Legacy-compatible return object ----------------------------------------


@dataclass
class ChatView:
    """Read-only DTO mirroring the legacy `Chat` ORM attribute surface.

    Callers that previously received a SQLAlchemy `Chat` ORM object receive
    this instead. All attribute names are identical so no call-site changes
    are needed beyond switching to the cached service.

    Note: `messages`, `sources`, and `research_sessions` are always ``[]``
    on lightweight list/get paths. They are populated only by `get_full`
    (which returns a `ChatFullViewCached` wrapper instead).
    """

    id: UUID
    user_id: str
    title: str | None
    status: str
    chat_type: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    incognito_session_id: UUID | None = None
    messages: list[Any] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)
    research_sessions: list[Any] = field(default_factory=list)

    # Legacy ORM compat props used in a few call sites
    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @property
    def is_archived(self) -> bool:
        from deep_research.models.chat import ChatStatus
        return self.status == ChatStatus.ARCHIVED

    @property
    def is_incognito(self) -> bool:
        from deep_research.models.chat import ChatType
        return self.chat_type == ChatType.INCOGNITO


@dataclass
class ChatFullViewCached:
    """Full-hydration view from the cached backend.

    Returned by `CachedChatService.get_full`. The `chat` attribute is a
    `ChatView` (same interface as legacy `Chat`). The `messages`,
    `research_sessions`, and `sources` lists are populated from `ChatState`.

    `get_full` in `chats.py` iterates `chat.messages` — so we expose `messages`
    both on the `ChatView` and here. The chats.py endpoint accesses
    `chat.messages` directly, so we patch the `chat` object's messages list.
    """

    chat: ChatView
    messages: list[Any] = field(default_factory=list)
    research_sessions: list[Any] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)
    # Per-agent UI surface state persisted in ChatState.chat.metadata["surface_state"].
    surface_state: dict[str, Any] | None = None

    # Delegate attribute access for fields the endpoint reads from `chat`
    @property
    def id(self) -> UUID:
        return self.chat.id

    @property
    def title(self) -> str | None:
        return self.chat.title

    @property
    def status(self) -> str:
        return self.chat.status

    @property
    def chat_type(self) -> str:
        return self.chat.chat_type

    @property
    def created_at(self) -> datetime:
        return self.chat.created_at

    @property
    def updated_at(self) -> datetime:
        return self.chat.updated_at

    @property
    def deleted_at(self) -> datetime | None:
        return self.chat.deleted_at

    @property
    def user_id(self) -> str:
        return self.chat.user_id


# --- Helper -----------------------------------------------------------------


def _meta_to_view(meta: ChatMeta, state: ChatState | None = None) -> ChatView:
    """Build a `ChatView` from a `ChatMeta` + optional `ChatState`."""
    # Derive chat_type and incognito_session_id from embedded state
    from deep_research.models.chat import ChatStatus, ChatType
    chat_embed = state.chat if state is not None else None
    chat_type = ChatType.REGULAR
    incognito_session_id: UUID | None = None
    if chat_embed is not None:
        raw_type = getattr(chat_embed, "type", "native")
        if raw_type == "incognito":
            chat_type = ChatType.INCOGNITO
        incognito_session_id = getattr(chat_embed, "incognito_session_id", None)

    # Status: stored in ChatState.chat.metadata["status"] if present;
    # default to ACTIVE for non-deleted, DELETED for soft-deleted.
    status = ChatStatus.ACTIVE
    if meta.deleted_at is not None:
        status = ChatStatus.DELETED
    elif chat_embed is not None:
        raw_status = chat_embed.metadata.get("status")
        if raw_status:
            with suppress(ValueError):
                status = ChatStatus(raw_status)

    return ChatView(
        id=meta.chat_id,
        user_id=meta.user_id,
        title=meta.title or None,
        status=status,
        chat_type=chat_type,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        deleted_at=meta.deleted_at,
        incognito_session_id=incognito_session_id,
    )


def _doc_to_view(doc: ChatDocument) -> ChatView:
    return _meta_to_view(doc.meta, doc.state)


# --- Service ----------------------------------------------------------------


class CachedChatService(_CachedServiceBase, IChatService):
    """`IChatService` implementation over `StorageStack`.

    Thread-safety: see module docstring.
    """

    _service_name = "chat"

    def __init__(self, stack: StorageStack) -> None:
        super().__init__(stack)

    # -- Reads -----------------------------------------------------------

    async def get_for_user(self, chat_id: UUID, user_id: str) -> ChatView | None:
        """Return ChatView if chat exists, owned by user, and not deleted."""
        # Use _read_chat to hit the in-memory cache first (warm after create/mutate).
        # Falls back to backend.load_chat on cache miss.
        try:
            doc = await self._read_chat(chat_id)
        except Exception:
            return None
        if doc.meta.user_id != user_id:
            return None
        if doc.meta.deleted_at is not None:
            return None
        return _doc_to_view(doc)

    async def get_by_id(self, chat_id: UUID) -> ChatView | None:
        """Return ChatView by PK without ownership check; None if absent/deleted."""
        try:
            doc = await self._read_chat(chat_id)
        except Exception:
            return None
        if doc.meta.deleted_at is not None:
            return None
        return _doc_to_view(doc)

    async def get_full(self, chat_id: UUID, user_id: str) -> ChatFullViewCached | None:
        """Return ChatFullViewCached with messages/sessions/sources from state.

        Double-trip: backend for existence + ownership, then cache for freshness.
        The cache alone cannot gate existence — cache.get with user_id
        constructs a placeholder ChatDocument on backend miss.
        """
        backend_doc = await self._stack.backend.load_chat(chat_id)
        if backend_doc is None:
            return None
        if backend_doc.meta.user_id != user_id:
            return None
        if backend_doc.meta.deleted_at is not None:
            return None

        try:
            doc = await self._read_chat(chat_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "CHAT_GET_FULL_CACHE_READ_FAILED chat=%s err=%s",
                str(chat_id)[:8], str(exc)[:120],
            )
            doc = backend_doc

        if doc.meta.user_id != user_id:
            logger.error(
                "CHAT_GET_FULL_OWNERSHIP_MISMATCH chat=%s backend=%s cache=%s",
                str(chat_id)[:8], backend_doc.meta.user_id, doc.meta.user_id,
            )
            return None

        view = _doc_to_view(doc)
        messages = [_state_msg_to_view(m) for m in doc.state.messages]
        sources = [_state_source_to_view(s) for s in doc.state.sources]
        research_sessions = [_state_session_to_view(rs) for rs in doc.state.research_sessions]

        # Link each message to its research session so the /full endpoint can
        # surface inline claims/verification_data. Without this, every message
        # view keeps research_session=None and chats.py emits claims=[] for all
        # messages — every citation renders grey (see _link_sessions_to_messages).
        _link_sessions_to_messages(messages, research_sessions)

        # Patch view.messages so that `chat.messages` in chats.py works
        view.messages = messages
        view.sources = sources
        view.research_sessions = research_sessions

        raw_surface = doc.state.chat.metadata.get("surface_state")
        surface_state: dict[str, Any] | None = (
            dict(raw_surface) if isinstance(raw_surface, dict) else None
        )

        full = ChatFullViewCached(
            chat=view,
            messages=messages,
            research_sessions=research_sessions,
            sources=sources,
            surface_state=surface_state,
        )
        return full

    async def list(
        self,
        user_id: str,
        status: Any | None = None,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[ChatView], int]:
        """Return ``(chats, total_count)`` for user. Cache hit returns in <1 ms."""
        status_str: str | None = None
        if status is not None:
            status_str = status.value if hasattr(status, "value") else str(status)

        # Fast path: cache hit.
        cached = _CHAT_LIST_CACHE.get(user_id, status_str, search, limit, offset)
        if cached is not None:
            return cached

        # Slow path: acquire per-user lock, re-check, then hit backend.
        lock = await _CHAT_LIST_CACHE.user_lock(user_id)
        async with lock:
            cached = _CHAT_LIST_CACHE.get(user_id, status_str, search, limit, offset)
            if cached is not None:
                return cached

            metas = await self._stack.backend.list_chat_metas(
                user_id,
                include_deleted=False,
                limit=limit,
                offset=offset,
                search=search,
                status=status_str,
            )

            all_metas = await self._stack.backend.list_chat_metas(
                user_id,
                include_deleted=False,
                limit=100_000,  # effectively unbounded
                offset=0,
                search=search,
                status=status_str,
            )
            total = len(all_metas)

            views = [_meta_to_view(m) for m in metas]
            _CHAT_LIST_CACHE.set(user_id, status_str, search, limit, offset, views, total)
            return views, total

    # -- Writes ----------------------------------------------------------

    async def create(self, user_id: str, title: str | None = None) -> ChatView:
        """Create a new chat document and return a ChatView."""
        chat_id = uuid4()
        now = datetime.now(UTC)
        meta = ChatMeta(
            chat_id=chat_id,
            user_id=user_id,
            title=title or "",
            preview="",
            created_at=now,
            updated_at=now,
            version=0,
        )
        state = ChatState(chat=ChatMetaEmbed(title=title or ""))
        doc = ChatDocument(meta=meta, state=state)
        await self._stack.backend.write_chat(doc, expected_version=0)
        _CHAT_LIST_CACHE.invalidate_user(user_id)
        logger.info("Created chat %s for user %s", chat_id, user_id)
        return _doc_to_view(doc)

    async def update_chat(
        self,
        chat_id: UUID,
        user_id: str,
        title: str | None = None,
        status: Any | None = None,
        surface_state_patch: dict[str, Any] | None = None,
    ) -> ChatView | None:
        """Update title, status, and/or surface_state; return updated ChatView or None."""
        # Hydrate into cache first (required by _mutate_chat)
        doc = await self._read_chat(chat_id)
        if doc.meta.user_id != user_id or doc.meta.deleted_at is not None:
            return None

        def _apply(d: ChatDocument) -> None:
            if title is not None:
                d.meta.title = title
                d.state.chat.title = title
            if status is not None:
                status_str = status.value if hasattr(status, "value") else str(status)
                d.state.chat.metadata["status"] = status_str
            if surface_state_patch is not None:
                existing_surface: dict[str, Any] = dict(
                    d.state.chat.metadata.get("surface_state") or {}
                )
                d.state.chat.metadata["surface_state"] = merge_surface_state(
                    existing_surface, surface_state_patch
                )
            d.meta.updated_at = datetime.now(UTC)

        await self._mutate_chat(chat_id, _apply)
        _CHAT_LIST_CACHE.invalidate_user(user_id)
        # Return view from the in-memory doc (already updated)
        hydrated = await self._read_chat(chat_id)
        return _doc_to_view(hydrated)

    async def soft_delete(self, chat_id: UUID, user_id: str) -> bool:
        """Soft-delete; return True if found and owned, False otherwise."""
        # Hydrate first so _mutate_chat can find the entry
        try:
            doc = await self._read_chat(chat_id)
        except Exception:
            return False
        if doc.meta.user_id != user_id or doc.meta.deleted_at is not None:
            return False

        from deep_research.models.chat import ChatStatus

        def _apply(d: ChatDocument) -> None:
            now = datetime.now(UTC)
            d.meta.deleted_at = now
            d.meta.updated_at = now
            d.state.chat.metadata["status"] = ChatStatus.DELETED.value

        await self._mutate_chat(chat_id, _apply)
        _CHAT_LIST_CACHE.invalidate_user(user_id)
        # Flush immediately so `list_chat_metas` on the backend sees the
        # updated `deleted_at` without waiting for the queue's next tick.
        await self._stack.queue.flush_chat_now(chat_id)
        return True

    async def restore(self, chat_id: UUID, user_id: str) -> ChatView | None:
        """Restore soft-deleted chat; return restored ChatView or None."""
        # For restore, we need to find the deleted chat. load_chat filters
        # deleted_at IS NULL, so we use list_chat_metas with include_deleted.
        metas = await self._stack.backend.list_chat_metas(
            user_id, include_deleted=True, limit=100_000
        )
        meta = next((m for m in metas if m.chat_id == chat_id), None)
        if meta is None or meta.deleted_at is None:
            return None
        if meta.user_id != user_id:
            return None

        from deep_research.models.chat import ChatStatus

        # Build a document from the meta + empty state and write it with
        # deleted_at cleared — this undeletes the chat.
        state = ChatState()
        restored_meta = meta.model_copy(
            update={
                "deleted_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        restored_meta.model_fields_set.add("deleted_at")
        doc = ChatDocument(meta=restored_meta, state=state)
        doc.state.chat.metadata["status"] = ChatStatus.ACTIVE.value

        # Write directly via backend (bypassing _mutate_chat which requires
        # the chat to already be in the hot-path cache).
        await self._stack.backend.write_chat(doc, expected_version=meta.version)

        # Hydrate into cache so subsequent requests see the restored state.
        await self._read_chat(chat_id)

        # Re-read from backend to get the final view.
        updated = await self._stack.backend.load_chat(chat_id)
        if updated is None:
            return None
        return _doc_to_view(updated)

    async def update_title_from_message(
        self, chat_id: UUID, message_content: str
    ) -> None:
        """Set title from message body if title is currently empty.

        Uses the canonical derive_chat_title_from_query helper so every title
        writer produces identical output for the same input (byte-parity with
        framework_orchestrator's PersistenceCompletedEvent.chat_title).
        """
        try:
            doc = await self._read_chat(chat_id)
        except Exception:
            return
        if doc.meta.title:
            return

        derived_title = derive_chat_title_from_query(message_content)
        if not derived_title:
            return

        def _apply(d: ChatDocument) -> None:
            if not d.meta.title:
                d.meta.title = derived_title
                d.state.chat.title = derived_title
                d.meta.updated_at = datetime.now(UTC)

        await self._mutate_chat(chat_id, _apply)

    async def purge_deleted_chats(self, _days_old: int = 30) -> int:
        """Cold-path permanent deletion — not used on hot paths.

        The cached backend has no direct DELETE on the document table; we
        mark via soft-delete and leave hard-delete to a maintenance script.
        Returns 0 — the cleanup job should use the legacy service or a
        direct SQL statement for hard deletes.
        """
        logger.warning(
            "purge_deleted_chats called on CachedChatService — "
            "no-op; use legacy service or direct SQL for hard deletes"
        )
        return 0

    async def list_incognito_for_session(self, session_id: UUID) -> builtins.list[ChatView]:
        """Return incognito chats for a session. N+1 via list_rows + load_chat.

        N+1 note: incognito session listing is a rare, low-traffic path
        (incognito chats are ephemeral). A follow-up can batch this.
        """
        rows = await self._stack.backend.list_rows(
            "incognito_sessions", {"session_id": session_id}
        )
        views: list[ChatView] = []
        for row in rows:
            raw_chat_id = row.get("chat_id")
            if raw_chat_id is None:
                continue
            if isinstance(raw_chat_id, str):
                try:
                    cid = UUID(raw_chat_id)
                except ValueError:
                    continue
            else:
                cid = raw_chat_id
            doc = await self._stack.backend.load_chat(cid)
            if doc is not None:
                views.append(_doc_to_view(doc))
        return views

    # -- ORM-compat shims (used by legacy incognito flow in chats.py) ----

    async def add(self, chat: Any) -> Any:
        """Persist a pre-built Chat ORM object.

        The incognito path in `chats.py` constructs a raw `Chat(...)` ORM
        object and calls `chat_service.add(chat)`. When the cached service is
        active, we convert that ORM object into a `ChatDocument` and write it.
        """
        from deep_research.models.chat import ChatType

        chat_id = getattr(chat, "id", None) or uuid4()
        if isinstance(chat_id, str):
            chat_id = UUID(chat_id)
        now = datetime.now(UTC)

        chat_type_val = getattr(chat, "chat_type", ChatType.REGULAR)
        type_str = chat_type_val.value if hasattr(chat_type_val, "value") else str(chat_type_val)

        incognito_session_id = getattr(chat, "incognito_session_id", None)
        metadata: dict[str, Any] = {}
        status_val = getattr(chat, "status", None)
        if status_val is not None:
            metadata["status"] = status_val.value if hasattr(status_val, "value") else str(status_val)

        embed = ChatMetaEmbed(
            type=type_str,
            title=getattr(chat, "title", "") or "",
            incognito_session_id=incognito_session_id,
            metadata=metadata,
        )
        meta = ChatMeta(
            chat_id=chat_id,
            user_id=chat.user_id,
            title=getattr(chat, "title", "") or "",
            created_at=now,
            updated_at=now,
            version=0,
        )
        state = ChatState(chat=embed)
        doc = ChatDocument(meta=meta, state=state)
        await self._stack.backend.write_chat(doc, expected_version=0)
        # Update the ORM object's id so the caller can use it
        chat.id = chat_id
        return _doc_to_view(doc)

    async def update(self, chat: Any) -> Any:
        """Persist changes from a Chat-like object already loaded."""
        chat_id = UUID(str(chat.id))
        # Hydrate into cache first so _mutate_chat can find the entry
        with suppress(Exception):
            await self._read_chat(chat_id)

        def _apply(d: ChatDocument) -> None:
            new_title = getattr(chat, "title", None)
            if new_title is not None:
                d.meta.title = new_title
                d.state.chat.title = new_title

            chat_type_val = getattr(chat, "chat_type", None)
            if chat_type_val is not None:
                type_str = chat_type_val.value if hasattr(chat_type_val, "value") else str(chat_type_val)
                d.state.chat.type = type_str

            incognito_sid = getattr(chat, "incognito_session_id", None)
            d.state.chat.incognito_session_id = incognito_sid

            status_val = getattr(chat, "status", None)
            if status_val is not None:
                d.state.chat.metadata["status"] = (
                    status_val.value if hasattr(status_val, "value") else str(status_val)
                )

            deleted_at = getattr(chat, "deleted_at", None)
            d.meta.deleted_at = deleted_at
            d.meta.updated_at = datetime.now(UTC)

        await self._mutate_chat(chat_id, _apply)
        updated = await self._stack.backend.load_chat(chat_id)
        if updated is None:
            return chat
        return _doc_to_view(updated)


# --- State-object view adapters ---------------------------------------------
# These build lightweight SimpleNamespace-like objects from ChatState sub-models
# so the endpoint code in chats.py can iterate messages/research_sessions/sources
# with attribute access identical to the ORM objects.


def _state_msg_to_view(msg: Any) -> Any:
    """Adapt a `ChatState.Message` to a legacy-shaped message namespace."""
    from types import SimpleNamespace

    from deep_research.models.message import MessageRole

    role = msg.role
    # Normalize role string to enum if needed
    try:
        role_enum: MessageRole | str = MessageRole(role)
    except (ValueError, TypeError):
        role_enum = role

    return SimpleNamespace(
        id=msg.id,
        chat_id=None,  # chat_id not stored in state messages
        role=role_enum,
        content=msg.content,
        created_at=msg.ts,
        is_edited=False,
        research_session=None,  # sessions linked separately
        metadata=msg.metadata,
    )


def _state_source_to_view(src: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=src.id,
        url=src.url,
        title=src.title,
        snippet=None,
        relevance_score=None,
        source_type=src.source_type,
        source_metadata=src.metadata,
        is_cited=False,
    )


def _state_session_to_view(rs: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=rs.id,
        message_id=rs.message_id,
        status=rs.status,
        plan=rs.plan,
        observations=rs.observations,
        query_classification=rs.query_classification,
        verification_data=rs.verification_data,
        current_step_index=rs.current_step,
        plan_iterations=None,
        started_at=rs.started_at,
        completed_at=rs.completed_at,
        sources=[],
        reasoning_steps=[],
        current_agent=None,
        research_depth="auto",
        query=None,
        query_mode=None,
    )


def _session_supersedes(candidate: Any, incumbent: Any) -> bool:
    """True if ``candidate`` should replace ``incumbent`` as a message's session.

    Ranking: a session carrying ``verification_data`` wins over one without;
    ties break on the latest ``started_at``. This makes a regenerated/completed
    answer win over an earlier empty or in-progress attempt for the same message.
    """
    cand_has = bool(getattr(candidate, "verification_data", None))
    inc_has = bool(getattr(incumbent, "verification_data", None))
    if cand_has != inc_has:
        return cand_has
    cand_ts = getattr(candidate, "started_at", None)
    inc_ts = getattr(incumbent, "started_at", None)
    if cand_ts is None:
        return False
    if inc_ts is None:
        return True
    return bool(cand_ts > inc_ts)


def _link_sessions_to_messages(messages: list[Any], research_sessions: list[Any]) -> None:
    """Attach each message's ``research_session`` in place, by ``message_id``.

    ``_state_msg_to_view`` starts every message view with
    ``research_session=None`` ("linked separately"); the ``/chats/{id}/full``
    endpoint (``api/v1/chats.py``) reads claims and verification data
    EXCLUSIVELY from ``msg.research_session``. Without this link every message
    surfaces ``claims=[]`` and all citations render grey — and the frontend
    never fires ``/messages/{id}/claims`` either, because its
    ``latestAgentMessageIdForCitations`` gate requires ``m.researchSession``.

    When more than one session targets the same message (e.g. a regenerated
    answer), the highest-ranked session wins (see ``_session_supersedes``).
    """
    by_message: dict[Any, Any] = {}
    for rs in research_sessions:
        message_id = getattr(rs, "message_id", None)
        if message_id is None:
            continue
        incumbent = by_message.get(message_id)
        if incumbent is None or _session_supersedes(rs, incumbent):
            by_message[message_id] = rs
    for msg in messages:
        linked = by_message.get(getattr(msg, "id", None))
        if linked is not None:
            msg.research_session = linked
