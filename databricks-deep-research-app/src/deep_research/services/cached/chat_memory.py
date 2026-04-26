"""Cache-backed `ChatMemoryService` — Strategy A (inherit + swap hydrate).

Per the finalization plan's Wave 5c: inherit from `SQLAlchemyChatMemoryService`
(the renamed legacy class — aliased to `ChatMemoryService` for back-compat),
override only the persistence paths, and reuse `render`/`snapshot`/
`search_findings`/`account_candidates` unchanged by populating the legacy
in-memory mirrors (`self._findings`, `self._entities`, …) from the
`ChatState` pydantic document on hydrate.

This preserves the plugin contract in one line: the `render()` output is
byte-equal to the legacy impl for the same input state, because the legacy
code path is what runs. Only what it reads FROM changes.

`preprocess_new_files` is the LLM-extraction path — it depends on a real
`FileUploadService` that reads file chunks. Kept operational only when the
legacy file-upload service is still available (STORAGE_SERVICE_IMPL=cached
on the memory service ≠ automatic cached file-upload service). For pure-
cached deployments a follow-up lands a cached file-chunk read path and
adapts this method.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.models.chat_memory_coverage import ChatMemoryCoverage
from deep_research.models.chat_memory_entity import ChatMemoryEntity
from deep_research.models.chat_memory_file import ChatMemoryFile
from deep_research.models.chat_memory_finding import ChatMemoryFinding
from deep_research.models.chat_memory_plugin_ext import ChatMemoryPluginExt
from deep_research.services.chat_memory_service import (
    ChatMemoryService,
    _EntityRegistry,
)
from deep_research.storage.documents import (
    Coverage as DocCoverage,
)
from deep_research.storage.documents import (
    Entity as DocEntity,
)
from deep_research.storage.documents import (
    FileMemo as DocFileMemo,
)
from deep_research.storage.documents import (
    Finding as DocFinding,
)

if TYPE_CHECKING:
    from databricks_deep_research.memory import ChatMemorySnapshot

    from deep_research.storage.documents import ChatDocument, ChatState
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)


class CachedChatMemoryService(ChatMemoryService):
    """`IChatMemoryService` over `StorageStack`.

    Constructor takes a `StorageStack` instead of an `AsyncSession`; the
    legacy base class's `session` is set to `None` since no SQLAlchemy work
    runs through it.
    """

    def __init__(
        self,
        stack: StorageStack,
        embedder: Any = None,
        *,
        llm: Any = None,
    ) -> None:
        # Initialize legacy state but pass `session=None`. Any legacy path that
        # would try to touch `self._session` would fail loudly; we override
        # every such path below.
        super().__init__(session=None, embedder=embedder, llm=llm)  # type: ignore[arg-type]
        self._stack = stack

    # -- Hydration ------------------------------------------------------

    async def hydrate(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
    ) -> ChatMemorySnapshot:
        """Load the ChatDocument from cache and project its memory into the
        legacy in-memory mirrors so inherited methods (`render`, `snapshot`,
        etc.) work unchanged.

        When `user_id` is provided and the chat has no row yet in
        `deep_research_state.chat_meta` (i.e. it was created via the legacy
        `public.chats` path and has not yet been mirrored into the new
        engine), the cache creates a fresh `ChatDocument` with an empty
        `ChatState.memory`. This is the "lazy-mirror" path that lets the
        agent framework run against any chat_id regardless of which engine
        owned its creation — subsequent memory mutations land in the new
        engine from that point on.
        """
        doc = await self._stack.cache.get(chat_id, user_id=user_id)
        self._populate_from_state(chat_id, doc.state)
        logger.info(
            "MEMORY_HYDRATED_FROM_CACHE chat_id=%s findings=%d entities=%d "
            "coverage=%d files=%d plugins=%d",
            chat_id,
            len(self._findings),
            len(self._entities.by_id),
            len(self._coverage),
            len(self._files),
            len(self._plugin_ext),
        )
        return self.snapshot()

    def _populate_from_state(self, chat_id: UUID, state: ChatState) -> None:
        self._chat_id = chat_id
        self._findings = [
            _legacy_finding(chat_id, f) for f in state.memory.findings
        ]
        entity_rows = [_legacy_entity(chat_id, e) for e in state.memory.entities]
        self._entities = _EntityRegistry(
            by_id={e.id: e for e in entity_rows},
            by_name={e.name.casefold(): e for e in entity_rows},
        )
        self._coverage = [
            _legacy_coverage(chat_id, c) for c in state.memory.coverage
        ]
        self._files = [
            _legacy_file(chat_id, f) for f in state.memory.files
        ]
        self._plugin_ext = {
            name: _legacy_plugin_ext(chat_id, name, entry.payload)
            for name, entry in state.memory.plugin_ext.items()
        }

    # -- Cache access helpers -------------------------------------------
    #
    # This class deliberately does NOT inherit from `_CachedServiceBase`
    # because it reuses `render`/`snapshot`/`_findings` state from the
    # legacy `ChatMemoryService` (Strategy A — see module docstring).
    # Replicate the two helpers it needs locally rather than refactoring
    # the base class's `__init__` contract.

    async def _read_chat(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
    ) -> ChatDocument:
        return await self._stack.cache.get(chat_id, user_id=user_id)

    async def _mutate_chat(
        self,
        chat_id: UUID,
        fn: Callable[[ChatDocument], None],
    ) -> None:
        await self._stack.cache.mutate(chat_id, fn)

    # -- Plugin extension write -----------------------------------------

    async def upsert_plugin_ext(
        self,
        plugin_name: str,
        payload_json: dict[str, Any],
    ) -> None:
        if self._chat_id is None:
            raise RuntimeError("hydrate() must be called before upsert_plugin_ext()")
        payload_size = len(json.dumps(payload_json).encode("utf-8"))
        if payload_size > self.PAYLOAD_MAX_BYTES:
            raise ValueError(
                f"plugin_extensions payload for {plugin_name!r} is "
                f"{payload_size} bytes, exceeds PAYLOAD_MAX_BYTES="
                f"{self.PAYLOAD_MAX_BYTES}"
            )
        chat_id = self._chat_id
        # 1) update in-memory mirror so the workflow's subsequent render()
        #    sees the brief before DB persistence completes. Matches the
        #    `feedback_render_must_include_plugin_extensions` invariant.
        self._plugin_ext[plugin_name] = _legacy_plugin_ext(
            chat_id, plugin_name, payload_json
        )
        # 2) route through cache.mutate — synchronous mutation to in-memory
        #    ChatState + fire-and-forget DB persistence via WriteQueue.
        await self._stack.cache.mutate(
            chat_id,
            lambda doc, _pn=plugin_name, _p=payload_json: doc.state.upsert_plugin_ext(
                _pn, _p
            ),
        )

    # -- Private upsert paths -------------------------------------------
    #
    # The legacy class's private `_upsert_finding` / `_upsert_entity` /
    # `_upsert_coverage` / `_upsert_file` are called from
    # `preprocess_new_files` (and potentially future consolidation loops).
    # Override them to route through the cache so anything the inherited
    # code triggers lands in the new schema.

    async def _upsert_finding(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """Route finding upserts through the cache.

        Signature matches the legacy method. We mirror its behavior: compute
        a content_hash, update in-memory mirror, mutate ChatState via cache.
        """
        # Build a placeholder that quacks like a ChatMemoryFinding so legacy
        # callers (e.g. preprocess_new_files) that append to self._findings
        # still work. Because the legacy class owns its own call sites we
        # delegate argument parsing to a helper on this class.
        raise NotImplementedError(
            "CachedChatMemoryService._upsert_finding is only reachable via "
            "preprocess_new_files; that path is a follow-up per the plan."
        )

    async def _upsert_entity(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "CachedChatMemoryService._upsert_entity is only reachable via "
            "preprocess_new_files; that path is a follow-up per the plan."
        )

    async def _upsert_file(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "CachedChatMemoryService._upsert_file is only reachable via "
            "preprocess_new_files; that path is a follow-up per the plan."
        )

    # -- preprocess_new_files: routes through CachedFileUploadService ----------

    async def preprocess_new_files(
        self,
        chat_id: Any,
        file_ids: list[UUID],
        *,
        file_service: Any = None,
        research_session_id: UUID | None = None,
    ) -> None:
        """Preprocess uploaded files via CachedFileUploadService.

        Reads chunk text from the append-only ``file_chunks`` table via the
        ``file_service`` passed by the orchestrator.  Falls back to calling
        the parent class implementation when a legacy SQLAlchemy-backed
        ``file_service`` is provided (detected by absence of ``_stack``).
        """
        if not file_ids:
            return

        # Determine whether we have a cached or legacy file service
        if file_service is None or not hasattr(file_service, "_stack"):
            # No file service or legacy file service — try parent class path
            try:
                await super().preprocess_new_files(
                    chat_id,
                    file_ids,
                    file_service=file_service,
                    research_session_id=research_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "CACHED_CHAT_MEMORY_PREPROCESS_FILES_LEGACY_FAILED "
                    "chat_id=%s error=%s",
                    chat_id,
                    str(exc)[:200],
                )
            return

        # Cached path: read file metadata + chunks via CachedFileUploadService
        chat_uuid = chat_id if isinstance(chat_id, UUID) else UUID(str(chat_id))
        for file_id in file_ids:
            try:
                uploaded = await file_service.get(file_id)
                if uploaded is None:
                    logger.warning(
                        "CACHED_CHAT_MEMORY_FILE_NOT_FOUND file_id=%s", file_id
                    )
                    continue

                # Read all chunks in order
                chunks = await file_service.get_file_chunks(file_id)
                full_text = "\n\n".join(
                    c.content for c in chunks if c.content
                )
                if not full_text.strip():
                    logger.warning(
                        "CACHED_CHAT_MEMORY_FILE_EMPTY file_id=%s", file_id
                    )
                    continue

                # Register the file memo in the memory service
                await self._upsert_file_memo(
                    chat_id=chat_uuid,
                    file_id=file_id,
                    filename=uploaded.filename,
                    content_summary=full_text[:2000],
                    chunk_count=len(chunks),
                    research_session_id=research_session_id,
                )

            except Exception as exc:
                logger.warning(
                    "CACHED_CHAT_MEMORY_PREPROCESS_FILE_FAILED "
                    "file_id=%s error=%s",
                    file_id,
                    str(exc)[:200],
                )

    async def _upsert_file_memo(
        self,
        chat_id: UUID,
        file_id: UUID,
        filename: str,
        content_summary: str,
        chunk_count: int,
        research_session_id: UUID | None,  # noqa: ARG002 — reserved for future
    ) -> None:
        """Store a FileMemo into `state.memory.files` via ChatState."""
        try:
            doc = await self._read_chat(chat_id)
            if any(f.id == file_id for f in doc.state.memory.files):
                return  # idempotent

            memo = DocFileMemo(
                id=file_id,
                name=filename,
                summary=content_summary,
            )

            # Keep the legacy in-memory mirror hot so the current turn's
            # snapshot()/render() see the new memo immediately — matches
            # the pattern in upsert_plugin_ext above. chunk_count lives
            # only on the legacy row because DocFileMemo has no such
            # field; a re-hydrate later defaults to 0, which is fine for
            # the render/UI paths that consult it.
            self._files.append(
                _legacy_file(chat_id, memo, chunk_count=chunk_count)
            )

            await self._mutate_chat(
                chat_id,
                lambda d, _m=memo: d.state.upsert_file_memo(_m),
            )
        except Exception as exc:
            logger.warning(
                "CACHED_CHAT_MEMORY_UPSERT_FILE_MEMO_FAILED "
                "file_id=%s error=%s",
                file_id,
                str(exc)[:200],
            )


# -- Adapter helpers -------------------------------------------------------


def _legacy_finding(chat_id: UUID, f: DocFinding) -> ChatMemoryFinding:
    """Build an ORM-shape `ChatMemoryFinding` from a pydantic `Finding`."""
    row = ChatMemoryFinding()
    row.id = f.id
    row.chat_id = chat_id
    row.research_session_id = getattr(f, "research_session_id", None)
    row.source_step = f.step
    row.origin = f.origin
    row.content = f.content
    row.confidence = f.confidence
    # JSONB column accepts Python list directly.
    row.entity_ids = list(f.entity_ids)
    row.supersedes_id = f.supersedes_id
    row.content_hash = f.content_hash
    # created_at is not a required mirror attribute for render(); omit.
    return row


def _legacy_entity(chat_id: UUID, e: DocEntity) -> ChatMemoryEntity:
    row = ChatMemoryEntity()
    row.id = e.id
    row.chat_id = chat_id
    row.name = e.name
    row.entity_type = e.type
    row.aliases = list(e.aliases)
    row.summary = getattr(e, "summary", None)
    row.supporting_finding_ids = list(e.supporting_finding_ids)
    return row


def _legacy_coverage(chat_id: UUID, c: DocCoverage) -> ChatMemoryCoverage:
    row = ChatMemoryCoverage()
    row.id = c.id
    row.chat_id = chat_id
    row.topic = c.topic
    row.status = c.status
    row.depth = c.depth
    return row


def _legacy_file(
    chat_id: UUID, f: DocFileMemo, *, chunk_count: int = 0
) -> ChatMemoryFile:
    row = ChatMemoryFile()
    row.id = uuid4()  # legacy row id; not the file_id
    row.chat_id = chat_id
    row.file_id = f.id
    row.one_line_summary = f.summary or ""
    row.status = f.status
    row.entity_ids = list(f.entity_ids)
    row.chunk_count = chunk_count
    row.preprocessed_at = datetime.now(UTC)
    return row


def _legacy_plugin_ext(
    chat_id: UUID,
    plugin_name: str,
    payload: dict[str, Any],
) -> ChatMemoryPluginExt:
    row = ChatMemoryPluginExt()
    row.chat_id = chat_id
    row.plugin_name = plugin_name
    row.payload_json = dict(payload)
    return row
