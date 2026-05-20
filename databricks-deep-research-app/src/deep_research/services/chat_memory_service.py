"""Chat-scoped research memory service — parallel to ChatSourcePoolService.

This is the durable, cross-turn counterpart of the per-run ``PoolState``.
Findings, entities, coverage, and attached-file metadata live in
``chat_memory_*`` tables keyed by ``chat_id`` and are hydrated fresh at
the start of every turn.

Layering (user directive — see memory file
``feedback_memory_is_chat_scoped.md``):

- ``PoolState``           — per-run, in-memory, FIFO event stream. Harness internals only.
- ``ChatSourcePool``      — per-chat, DB-backed URL/content pool. Existing.
- ``ChatMemoryService``   — per-chat, DB-backed structured knowledge. This module.

Agents read a rendered projection of ``ChatMemorySnapshot`` via the
spotlighted system-prompt appendix; pools stay invisible to agents.

PR 1 scope: hydrate, regex-first file preprocessing, renderers, scoped
plugin writes, search scaffolding. PR 2 extends with the
``consolidate_from_pool`` LLM update loop and hybrid-search index
building.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from databricks_deep_research.memory import (
    ChatMemorySnapshot,
    CoverageEntry,
    EntityRecord,
    FileRef,
    KnowledgeFinding,
)
from databricks_deep_research.memory.llm_extractor import (
    DEFAULT_HEAD_CHARS,
    extract_file_content,
)
from databricks_deep_research.memory.spotlighting import (
    DEFAULT_SPOTLIGHTING_MODE,
    SpotlightingMode,
    wrap_attached_context,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.models.chat_memory_coverage import ChatMemoryCoverage
from deep_research.models.chat_memory_entity import ChatMemoryEntity
from deep_research.models.chat_memory_file import ChatMemoryFile
from deep_research.models.chat_memory_finding import ChatMemoryFinding
from deep_research.models.chat_memory_plugin_ext import ChatMemoryPluginExt
from deep_research.models.enums import EntityType, FindingOrigin
from deep_research.models.uploaded_file import FileChunk, UploadedFile

if TYPE_CHECKING:
    from databricks_deep_research.llm.client import FrameworkLLMClient

    from deep_research.services.file_upload_service import FileUploadService
    from deep_research.services.llm.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class _EntityRegistry:
    """Dual-keyed entity lookup built during hydrate().

    Per Architect iter-3 advisory: findings reference entities via
    ``entity_ids: list[UUID]``; rendering those by name requires an O(1)
    lookup by ID, not a linear scan. Hence the two-key index.
    """

    by_id: dict[UUID, ChatMemoryEntity]
    by_name: dict[str, ChatMemoryEntity]

    @classmethod
    def empty(cls) -> _EntityRegistry:
        return cls(by_id={}, by_name={})


class ChatMemoryService:
    """Chat-scoped research memory store.

    Usage (orchestrator at turn start):

        memory = ChatMemoryService(session, embedder=embedder)
        await memory.hydrate(chat_id)
        await memory.preprocess_new_files(
            chat_id, file_ids, file_service=file_upload_service,
        )
        # run ContextEnricher plugins, then workflow execution.
    """

    PAYLOAD_MAX_BYTES = 64 * 1024
    """Advisory size cap on ``chat_memory_plugin_ext.payload_json``.
    Enforced at the service layer; the DB column itself is unbounded JSONB."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder | None = None,
        *,
        llm: FrameworkLLMClient | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._llm = llm
        # search_findings() degrades to BM25-only when self._embedder is None,
        # matching ChatSourcePoolService at chat_source_pool_service.py:230-231.
        # Implementers MUST support the degraded path; embeddings are an
        # optimisation, not a requirement.
        # preprocess_new_files() fails open when self._llm is None — the file
        # still appears in the appendix by filename.

        self._chat_id: UUID | None = None
        self._findings: list[ChatMemoryFinding] = []
        self._entities = _EntityRegistry.empty()
        self._coverage: list[ChatMemoryCoverage] = []
        self._files: list[ChatMemoryFile] = []
        self._plugin_ext: dict[str, ChatMemoryPluginExt] = {}

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    async def hydrate(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,  # noqa: ARG002 — mirrors cached override signature
    ) -> ChatMemorySnapshot:
        """Load all chat-memory rows for ``chat_id`` and build a projection.

        The `user_id` kwarg is ignored by the legacy path — it exists only
        for signature parity with `CachedChatMemoryService.hydrate`, which
        uses it to lazy-create a mirror row in `deep_research_state.chat_meta`
        for chats that originated under the legacy `public.chats` path.
        """
        self._chat_id = chat_id

        # Four indexed queries in sequence (AsyncSession discourages shared
        # connection reuse for parallel execution). Each returns O(1..100)
        # rows for typical chats; cost is dominated by network RTT, not query.
        self._findings = list(
            (
                await self._session.execute(
                    select(ChatMemoryFinding).where(
                        ChatMemoryFinding.chat_id == chat_id
                    ).order_by(ChatMemoryFinding.source_step, ChatMemoryFinding.created_at)
                )
            ).scalars().all()
        )

        entity_rows = list(
            (
                await self._session.execute(
                    select(ChatMemoryEntity).where(
                        ChatMemoryEntity.chat_id == chat_id
                    )
                )
            ).scalars().all()
        )
        self._entities = _EntityRegistry(
            by_id={e.id: e for e in entity_rows},
            by_name={e.name.casefold(): e for e in entity_rows},
        )

        self._coverage = list(
            (
                await self._session.execute(
                    select(ChatMemoryCoverage).where(
                        ChatMemoryCoverage.chat_id == chat_id
                    )
                )
            ).scalars().all()
        )

        self._files = list(
            (
                await self._session.execute(
                    select(ChatMemoryFile).where(
                        ChatMemoryFile.chat_id == chat_id
                    ).order_by(ChatMemoryFile.preprocessed_at)
                )
            ).scalars().all()
        )

        plugin_ext_rows = list(
            (
                await self._session.execute(
                    select(ChatMemoryPluginExt).where(
                        ChatMemoryPluginExt.chat_id == chat_id
                    )
                )
            ).scalars().all()
        )
        self._plugin_ext = {row.plugin_name: row for row in plugin_ext_rows}

        logger.info(
            "MEMORY_HYDRATED_FROM_DB chat_id=%s findings=%d entities=%d coverage=%d files=%d plugins=%d",
            chat_id, len(self._findings), len(entity_rows),
            len(self._coverage), len(self._files), len(self._plugin_ext),
        )

        return self.snapshot()

    def snapshot(self) -> ChatMemorySnapshot:
        """Return the current in-memory projection."""
        assert self._chat_id is not None, "hydrate() must be called first"
        return ChatMemorySnapshot(
            chat_id=self._chat_id,
            files=[self._file_to_ref(f) for f in self._files],
            entities=[self._entity_to_record(e) for e in self._entities.by_id.values()],
            findings=[self._finding_to_projection(f) for f in self._findings],
            coverage=[
                CoverageEntry(topic=c.topic, status=c.status, depth=c.depth)  # type: ignore[arg-type]
                for c in self._coverage
            ],
            plugin_extensions={
                name: dict(row.payload_json or {})
                for name, row in self._plugin_ext.items()
            },
        )

    # ------------------------------------------------------------------
    # File preprocessing — universal LLM extractor (no regex, no profile)
    # ------------------------------------------------------------------

    async def preprocess_new_files(
        self,
        chat_id: UUID,
        file_ids: Iterable[UUID],
        *,
        file_service: FileUploadService | None = None,  # noqa: ARG002 — reserved for PR 2
        head_chars: int = DEFAULT_HEAD_CHARS,
        research_session_id: UUID | None = None,
    ) -> list[ChatMemoryFile]:
        """Extract entities + facts + summaries for uploaded files not yet in memory.

        Idempotent across turns: files already preprocessed whose
        ``UploadedFile.updated_at`` has not advanced past
        ``ChatMemoryFile.preprocessed_at`` are skipped.

        One cheap-tier LLM call per new file runs ``extract_file_content``
        (see ``databricks_deep_research.memory.llm_extractor``) and writes:
        - ``chat_memory_files`` (file_purpose + one_line_summary)
        - ``chat_memory_entities`` (one row per extracted entity)
        - ``chat_memory_findings`` (one row per key_fact, source_step=0,
          origin=FILE, confidence from the LLM output)

        Fails open when ``self._llm is None``: the file appears in memory
        by filename only (no entities, no findings), so downstream agents
        still know it exists even on LLM-less deployments.
        """
        if self._chat_id is None:
            raise RuntimeError("hydrate() must be called before preprocess_new_files()")
        if self._chat_id != chat_id:
            raise RuntimeError("preprocess_new_files() chat_id mismatch")

        existing = {f.file_id: f for f in self._files}
        newly_preprocessed: list[ChatMemoryFile] = []

        for file_id in file_ids:
            existing_cmf = existing.get(file_id)
            uf = (
                await self._session.execute(
                    select(UploadedFile).where(UploadedFile.id == file_id)
                )
            ).scalar_one_or_none()
            if uf is None:
                logger.warning("FILE_PREPROCESS_SKIP_MISSING file_id=%s", file_id)
                continue

            # Staleness check: rebuild if the file was re-chunked since last
            # preprocess, otherwise skip.
            if (
                existing_cmf is not None
                and uf.updated_at is not None
                and uf.updated_at <= existing_cmf.preprocessed_at
            ):
                continue

            logger.info(
                "FILE_PREPROCESS_START file_id=%s filename=%s",
                file_id, uf.filename,
            )

            head_text = await self._fetch_head_text(file_id, head_chars)
            extraction = await extract_file_content(
                filename=uf.filename,
                content_head=head_text,
                llm=self._llm,
                head_chars=head_chars,
            )

            # Resolve name -> ChatMemoryEntity via upsert, keep a map so
            # key_facts' related_entity can be linked.
            entity_ids: list[UUID] = []
            entity_name_to_id: dict[str, UUID] = {}
            for ent in extraction.entities:
                try:
                    etype = EntityType(ent.entity_type)
                except ValueError:
                    etype = EntityType.OTHER
                row = await self._upsert_entity(
                    chat_id=chat_id,
                    name=ent.name,
                    entity_type=etype,
                    aliases=list(ent.aliases),
                    summary=ent.role or "",
                )
                entity_ids.append(row.id)
                entity_name_to_id[ent.name.casefold()] = row.id

            # Promote each key_fact into a ChatMemoryFinding (source_step=0,
            # origin=FILE). This is what makes the render at synthesizer-scope
            # useful without any plugin.
            for fact in extraction.key_facts:
                linked_entity_ids: list[UUID] = []
                if fact.related_entity:
                    rid = entity_name_to_id.get(fact.related_entity.casefold())
                    if rid is not None:
                        linked_entity_ids.append(rid)
                await self._upsert_finding(
                    chat_id=chat_id,
                    research_session_id=research_session_id,
                    content=fact.content,
                    confidence=fact.confidence,
                    entity_ids=linked_entity_ids,
                    source_step=0,
                    origin=FindingOrigin.FILE,
                    category=fact.category,
                )

            # Upsert the chat_memory_files row with the LLM-derived summary.
            cmf = await self._upsert_file(
                chat_id=chat_id,
                file_id=file_id,
                one_line_summary=extraction.one_line_summary,
                entity_ids=entity_ids,
                chunk_count=uf.chunk_count,
            )
            newly_preprocessed.append(cmf)

            logger.info(
                "FILE_PREPROCESS_COMPLETE file_id=%s purpose=%r entities=%d facts=%d notes=%r",
                file_id, extraction.file_purpose, len(extraction.entities),
                len(extraction.key_facts), extraction.notes[:120],
            )

        if newly_preprocessed:
            # Refresh in-memory state so subsequent render() sees the new rows.
            await self.hydrate(chat_id)

        return newly_preprocessed

    async def _fetch_head_text(self, file_id: UUID, head_chars: int) -> str:
        """Concatenate the first few chunks for LLM extraction."""
        rows = list(
            (
                await self._session.execute(
                    select(FileChunk)
                    .where(FileChunk.file_id == file_id)
                    .order_by(FileChunk.chunk_index)
                    .limit(4)
                )
            ).scalars().all()
        )
        joined = "\n\n".join(row.content or "" for row in rows)
        return joined[:head_chars]

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    async def _upsert_entity(
        self,
        chat_id: UUID,
        name: str,
        entity_type: EntityType,
        *,
        aliases: list[str] | None = None,
        summary: str = "",
    ) -> ChatMemoryEntity:
        """Upsert a ChatMemoryEntity by (chat_id, name). Returns the row.

        New aliases are merged with any previously stored aliases (set
        union, bounded at 10). ``summary`` is updated only when non-empty
        to avoid blowing away better copy with a weaker one.
        """
        aliases_list = list(aliases or [])
        stmt = (
            pg_insert(ChatMemoryEntity)
            .values(
                chat_id=chat_id,
                name=name,
                entity_type=entity_type.value,
                summary=summary,
                aliases=aliases_list,
                supporting_finding_ids=[],
            )
            .on_conflict_do_update(
                index_elements=["chat_id", "name"],
                set_={
                    "entity_type": entity_type.value,
                    # Preserve existing summary if new one is empty.
                    "summary": (
                        summary if summary
                        else ChatMemoryEntity.summary
                    ),
                    "updated_at": datetime.utcnow(),
                },
            )
            .returning(ChatMemoryEntity)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()

        # Merge aliases client-side (pg_insert can't union JSONB arrays
        # inline). Do a follow-up write when we added anything new.
        if aliases_list:
            current = set(row.aliases or [])
            merged = current | set(aliases_list)
            if merged != current:
                row.aliases = list(merged)[:10]
                await self._session.flush()

        await self._session.flush()
        self._entities.by_id[row.id] = row
        self._entities.by_name[name.casefold()] = row
        logger.info(
            "ENTITY_REGISTRY_UPSERT chat_id=%s name=%r type=%s aliases=%d",
            chat_id, name, entity_type.value, len(aliases_list),
        )
        return row

    async def _upsert_finding(
        self,
        chat_id: UUID,
        content: str,
        *,
        research_session_id: UUID | None = None,
        confidence: str = "medium",
        entity_ids: list[UUID] | None = None,
        source_step: int = 0,
        origin: FindingOrigin = FindingOrigin.FILE,
        category: str = "other",
    ) -> ChatMemoryFinding:
        """Upsert a ChatMemoryFinding by (chat_id, content_hash).

        Called from ``preprocess_new_files`` for file-derived key_facts and
        (future) from ``consolidate_from_pool`` for research-derived
        findings. Dedup is content-hash based so re-processing a file
        produces stable rows.
        """
        hashed = self.content_hash(content)
        entity_id_strs = [str(eid) for eid in (entity_ids or [])]
        # ``category`` is currently carried via the content prefix so that
        # downstream projections (e.g. AccountBrief) can key off it without
        # adding a column. Future PR 2 will add a dedicated column if
        # category-filtering becomes hot.
        display_content = (
            content if category == "other" else f"[{category}] {content}"
        )

        stmt = (
            pg_insert(ChatMemoryFinding)
            .values(
                chat_id=chat_id,
                research_session_id=research_session_id,
                source_step=source_step,
                origin=origin.value if isinstance(origin, FindingOrigin) else origin,
                content=display_content,
                confidence=confidence,
                entity_ids=entity_id_strs,
                content_hash=hashed,
            )
            .on_conflict_do_update(
                index_elements=["chat_id", "content_hash"],
                set_={
                    "confidence": confidence,
                    "entity_ids": entity_id_strs,
                    "updated_at": datetime.utcnow(),
                },
            )
            .returning(ChatMemoryFinding)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        return row

    async def _upsert_file(
        self,
        chat_id: UUID,
        file_id: UUID,
        one_line_summary: str,
        entity_ids: list[UUID],
        chunk_count: int,
    ) -> ChatMemoryFile:
        entity_ids_json = [str(eid) for eid in entity_ids]
        stmt = (
            pg_insert(ChatMemoryFile)
            .values(
                chat_id=chat_id,
                file_id=file_id,
                one_line_summary=one_line_summary,
                entity_ids=entity_ids_json,
                chunk_count=chunk_count,
            )
            .on_conflict_do_update(
                index_elements=["chat_id", "file_id"],
                set_={
                    "one_line_summary": one_line_summary,
                    "entity_ids": entity_ids_json,
                    "chunk_count": chunk_count,
                    "preprocessed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                },
            )
            .returning(ChatMemoryFile)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        return row

    async def upsert_plugin_ext(
        self,
        plugin_name: str,
        payload_json: dict[str, Any],
    ) -> None:
        """Write a plugin's namespaced extension payload.

        Size-capped at ``PAYLOAD_MAX_BYTES``. Callers exceeding the cap
        see ``ValueError`` and should summarise further.
        """
        if self._chat_id is None:
            raise RuntimeError("hydrate() must be called before upsert_plugin_ext()")
        payload_size = len(json.dumps(payload_json).encode("utf-8"))
        if payload_size > self.PAYLOAD_MAX_BYTES:
            raise ValueError(
                f"plugin_extensions payload for {plugin_name!r} is "
                f"{payload_size} bytes, exceeds PAYLOAD_MAX_BYTES="
                f"{self.PAYLOAD_MAX_BYTES}"
            )
        stmt = (
            pg_insert(ChatMemoryPluginExt)
            .values(
                chat_id=self._chat_id,
                plugin_name=plugin_name,
                payload_json=payload_json,
            )
            .on_conflict_do_update(
                index_elements=["chat_id", "plugin_name"],
                set_={"payload_json": payload_json, "updated_at": datetime.utcnow()},
            )
            .returning(ChatMemoryPluginExt)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        self._plugin_ext[plugin_name] = row

    def enrich_scope(self, plugin_name: str) -> _PluginScope:
        """Return a scoped writer that only touches ``plugin_extensions[plugin_name]``."""
        return _PluginScope(service=self, plugin_name=plugin_name)

    # ------------------------------------------------------------------
    # Rendering — per-role markdown projection for prompt injection
    # ------------------------------------------------------------------

    def render(
        self,
        agent_type: str = "coordinator",
        max_chars: int = 3500,
    ) -> str:
        """Render a spotlighting-ready attached-context block.

        The caller (orchestrator) wraps this with ``wrap_attached_context``
        and seeds it into ``WorkflowState`` under
        ``CHAT_MEMORY_APPENDIX_STATE_KEY``. Agents never see this method
        directly — only the rendered result via the system-prompt
        appendix. Returns empty string when memory is empty (so the
        appendix injection becomes a no-op).
        """
        if not (
            self._files
            or self._entities.by_id
            or self._findings
            or self._coverage
            or self._plugin_ext
        ):
            return ""

        parts: list[str] = []

        if self._files:
            parts.append("## Attached files")
            for f in self._files[:20]:
                parts.append(
                    f"- **{self._file_filename(f.file_id)}** — {f.one_line_summary}"
                )

        accounts = [e for e in self._entities.by_id.values() if e.entity_type == EntityType.ACCOUNT.value]
        if accounts:
            parts.append("")
            parts.append("## Known accounts (from attached content)")
            for e in accounts[:10]:
                aliases = ", ".join(e.aliases) if e.aliases else ""
                line = f"- **{e.name}**"
                if aliases:
                    line += f" (aka {aliases})"
                if e.summary:
                    line += f" — {e.summary}"
                parts.append(line)

        people = [e for e in self._entities.by_id.values() if e.entity_type == EntityType.PERSON.value]
        if people and agent_type in {"coordinator", "researcher", "attendee_research", "crm_context", "synthesizer"}:
            parts.append("")
            parts.append("## Known people")
            for e in people[:15]:
                parts.append(f"- {e.name}" + (f" — {e.summary}" if e.summary else ""))

        competitors = [e for e in self._entities.by_id.values() if e.entity_type == EntityType.COMPETITOR.value]
        if competitors:
            parts.append("")
            parts.append("## Known competitors")
            for e in competitors[:10]:
                parts.append(f"- {e.name}" + (f" — {e.summary}" if e.summary else ""))

        # Coverage + findings are PR 2; render them if populated, skip otherwise.
        if self._coverage and agent_type in {"researcher", "reflector", "planner", "synthesizer"}:
            parts.append("")
            parts.append("## Research coverage")
            for c in self._coverage[:20]:
                parts.append(f"- {c.topic} — status={c.status}, depth={c.depth}")

        if self._findings and agent_type == "synthesizer":
            parts.append("")
            parts.append("## Consolidated findings")
            for fi in self._findings[:30]:
                parts.append(f"- [{fi.confidence}] {fi.content}")

        # Plugin-contributed structured briefs (e.g. sapresalesbot's AccountBrief).
        # Plugins write a pre-rendered markdown body under
        # ``payload_json["account_brief_markdown"]`` (or any ``*_markdown`` key);
        # surfacing it here makes the structured brief reachable by agents
        # without requiring plugin-specific template-var plumbing.
        if self._plugin_ext and agent_type in {
            "coordinator", "crm_context", "researcher", "synthesizer"
        }:
            for plugin_name, row in self._plugin_ext.items():
                payload = row.payload_json or {}
                rendered_fragments: list[str] = []
                for key, value in payload.items():
                    if isinstance(value, str) and key.endswith("_markdown") and value.strip():
                        rendered_fragments.append(value.strip())
                if rendered_fragments:
                    parts.append("")
                    parts.append(f"## Plugin context — {plugin_name}")
                    parts.extend(rendered_fragments)

        rendered = "\n".join(parts).strip()
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 1] + "…"
        return rendered

    def render_appendix_block(
        self,
        agent_type: str = "coordinator",
        max_chars: int = 3500,
        mode: SpotlightingMode = DEFAULT_SPOTLIGHTING_MODE,
    ) -> str:
        """Render + spotlighting-wrap, ready for the system-prompt appendix."""
        content = self.render(agent_type=agent_type, max_chars=max_chars)
        return wrap_attached_context(content, mode=mode)

    # ------------------------------------------------------------------
    # Entity lookup helpers (for extraction disambiguation)
    # ------------------------------------------------------------------

    def account_candidates(self) -> list[str]:
        """Return canonical account names known from attached files.

        Used by ``extraction.py`` to disambiguate ``company_name`` when the
        user's prompt mentions an ambiguous name like "Sagacity" and the
        attached files name "Sagacity Corp".
        """
        return [
            e.name
            for e in self._entities.by_id.values()
            if e.entity_type == EntityType.ACCOUNT.value
        ]

    # ------------------------------------------------------------------
    # Search (PR 2 hook — PR 1 returns empty)
    # ------------------------------------------------------------------

    async def search_findings(self, query: str, k: int = 5) -> list[KnowledgeFinding]:
        """Hybrid BM25 + embeddings search over chat_memory_findings.

        PR 1 scaffolding: when findings are empty (PR 1 state), returns
        empty list. PR 2 builds the BM25 + embeddings index during hydrate
        and implements ranking. Embedder-optional: degrades to BM25-only
        when ``self._embedder is None``.
        """
        if not self._findings:
            return []
        # Minimal BM25-ish keyword filter for PR 1. PR 2 swaps in the full
        # HybridSearchIndex used by ChatSourcePoolService.
        q = query.casefold()
        matches = [f for f in self._findings if q in (f.content or "").casefold()]
        matches.sort(key=lambda f: f.created_at or datetime.min, reverse=True)
        return [self._finding_to_projection(f) for f in matches[:k]]

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _file_filename(self, file_id: UUID) -> str:
        """Best-effort filename lookup; falls back to UUID when absent."""
        cmf = next((f for f in self._files if f.file_id == file_id), None)
        if cmf is None:
            return str(file_id)
        # We don't join uploaded_files in hydrate to keep the query count
        # low; the file name is available through FileRef.filename when the
        # orchestrator builds it. For the render helper we fall back to the
        # id. The orchestrator should prefer passing FileRef-populated
        # projections when filename matters (see _file_to_ref).
        return str(file_id)

    def _file_to_ref(self, f: ChatMemoryFile) -> FileRef:
        # We do not join uploaded_files here to avoid 1+N query patterns;
        # the orchestrator can enrich FileRef with live filename via its
        # own FileUploadService lookup if needed for non-prompt UI.
        return FileRef(
            id=f.file_id,
            filename=str(f.file_id),
            file_type="unknown",
            size=0,
            chunk_count=f.chunk_count,
            status="ready",
            one_line_summary=f.one_line_summary,
        )

    def _entity_to_record(self, e: ChatMemoryEntity) -> EntityRecord:
        return EntityRecord(
            id=e.id,
            name=e.name,
            entity_type=e.entity_type,
            summary=e.summary or "",
            aliases=list(e.aliases or []),
            supporting_finding_ids=[
                UUID(sid) if isinstance(sid, str) else sid
                for sid in (e.supporting_finding_ids or [])
            ],
        )

    def _finding_to_projection(self, f: ChatMemoryFinding) -> KnowledgeFinding:
        return KnowledgeFinding(
            id=f.id,
            content=f.content,
            confidence=f.confidence,  # type: ignore[arg-type]
            source_step=f.source_step,
            origin=f.origin,
            entity_ids=[
                UUID(eid) if isinstance(eid, str) else eid
                for eid in (f.entity_ids or [])
            ],
            supersedes_id=f.supersedes_id,
            content_hash=f.content_hash,
            created_at=f.created_at,
        )

    @staticmethod
    def content_hash(content: str) -> str:
        """Compute the dedup hash for a finding. PR 2 uses this in upserts."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class _PluginScope:
    """Scoped writer returned by ``enrich_scope``.

    Enforces the namespacing contract: plugins write only under their own
    ``plugin_extensions[plugin_name]`` key, never another plugin's.
    """

    service: ChatMemoryService
    plugin_name: str

    async def upsert(self, payload: dict[str, Any]) -> None:
        await self.service.upsert_plugin_ext(self.plugin_name, payload)
