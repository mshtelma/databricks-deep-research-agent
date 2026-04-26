"""Cache-backed `ISourceService`.

Storage model: `ChatState.sources[]` with dedup-by-URL. Legacy callers
historically scoped sources by `research_session_id`; under the cached
impl sources are **chat-scoped** and the research_session_id is
preserved as metadata on each Source entry for per-session filters.

Return shape: `SimpleNamespace` mirroring the legacy `Source` ORM surface
(`id`, `url`, `title`, `snippet`, `content`, `relevance_score`,
`source_type`, `source_metadata`, `research_session_id`, `fetched_at`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.models.enums import SourceType
from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import ISourceService
from deep_research.storage.documents import Source as DocSource

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)

MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50_000


def _truncate_title(title: str | None) -> str | None:
    if title and len(title) > MAX_TITLE_LENGTH:
        return title[: MAX_TITLE_LENGTH - 3] + "..."
    return title


def _truncate_content(content: str | None) -> str | None:
    if content and len(content) > MAX_CONTENT_LENGTH:
        return content[:MAX_CONTENT_LENGTH]
    return content


def _source_to_namespace(s: DocSource, chat_id: UUID | None = None) -> SimpleNamespace:
    meta = dict(s.metadata or {})
    return SimpleNamespace(
        id=s.id,
        chat_id=chat_id,
        research_session_id=_parse_uuid(meta.get("research_session_id")),
        url=s.url,
        title=s.title,
        snippet=meta.get("snippet"),
        content=meta.get("content"),
        relevance_score=meta.get("relevance_score"),
        source_type=s.source_type,
        source_metadata=meta.get("source_metadata") or meta.get("metadata"),
        fetched_at=_parse_ts(meta.get("fetched_at")),
        last_used_step=s.last_used_step,
        metadata=meta,
    )


def _parse_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except Exception:  # noqa: BLE001
        return None


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:  # noqa: BLE001
        return None


class CachedSourceService(_CachedServiceBase, ISourceService):
    """`ISourceService` via `ChatState.sources[]`."""

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    async def create(
        self,
        research_session_id: UUID,
        url: str,
        title: str | None = None,
        snippet: str | None = None,
        content: str | None = None,
        relevance_score: float | None = None,
        source_type: str = SourceType.WEB.value,
        source_metadata: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace:
        src_id = uuid4()
        now = datetime.now(UTC)
        meta: dict[str, Any] = {
            "research_session_id": str(research_session_id),
            "snippet": snippet,
            "content": _truncate_content(content),
            "relevance_score": relevance_score,
            "source_metadata": source_metadata,
            "fetched_at": now.isoformat(),
        }
        src = DocSource(
            id=src_id,
            url=url,
            title=_truncate_title(title),
            source_type=source_type,
            metadata=meta,
        )

        def _apply(doc: Any) -> None:
            doc.state.add_source(src)

        await self._mutate_chat(chat_id, _apply, dirty="state")
        logger.debug("SOURCE_CREATED chat_id=%s session_id=%s url=%s", chat_id, research_session_id, url[:80])
        return _source_to_namespace(src, chat_id)

    async def create_many(
        self,
        research_session_id: UUID,
        sources: list[dict[str, Any]],
        *,
        chat_id: UUID,
    ) -> list[SimpleNamespace]:
        now_iso = datetime.now(UTC).isoformat()
        built: list[DocSource] = []
        for data in sources:
            src_id = uuid4()
            meta: dict[str, Any] = {
                "research_session_id": str(research_session_id),
                "snippet": data.get("snippet"),
                "content": _truncate_content(data.get("content")),
                "relevance_score": data.get("relevance_score"),
                "source_metadata": data.get("metadata") or data.get("source_metadata"),
                "fetched_at": now_iso,
            }
            built.append(
                DocSource(
                    id=src_id,
                    url=str(data.get("url", "")),
                    title=_truncate_title(data.get("title")),
                    source_type=data.get("type", data.get("source_type", SourceType.WEB.value)),
                    metadata=meta,
                )
            )

        def _apply(doc: Any) -> None:
            for s in built:
                doc.state.add_source(s)

        await self._mutate_chat(chat_id, _apply, dirty="state")
        logger.info("SOURCES_CREATED chat_id=%s count=%d session_id=%s", chat_id, len(built), research_session_id)
        return [_source_to_namespace(s, chat_id) for s in built]

    async def list_by_session(
        self,
        research_session_id: UUID,
        limit: int = 100,
        *,
        chat_id: UUID | None = None,
    ) -> list[SimpleNamespace]:
        if chat_id is None:
            logger.warning("LIST_BY_SESSION_NO_CHAT_ID session_id=%s — returning empty list", research_session_id)
            return []
        doc = await self._read_chat(chat_id)
        target = str(research_session_id)
        matches = [
            s for s in doc.state.sources
            if (s.metadata or {}).get("research_session_id") == target
        ]
        matches.sort(
            key=lambda s: (s.metadata or {}).get("relevance_score") or 0.0,
            reverse=True,
        )
        return [_source_to_namespace(s, chat_id) for s in matches[:limit]]

    async def update_content(
        self,
        source_id: UUID,
        content: str,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        result: dict[str, DocSource | None] = {"src": None}
        truncated = _truncate_content(content)

        def _apply(doc: Any) -> None:
            for s in doc.state.sources:
                if s.id == source_id:
                    meta = dict(s.metadata or {})
                    meta["content"] = truncated
                    s.metadata = meta
                    result["src"] = s
                    return

        await self._mutate_chat(chat_id, _apply, dirty="state")
        s = result["src"]
        return _source_to_namespace(s, chat_id) if s else None

    async def get_by_url(
        self,
        research_session_id: UUID,
        url: str,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        doc = await self._read_chat(chat_id)
        target = str(research_session_id)
        for s in doc.state.sources:
            if s.url == url and (s.metadata or {}).get("research_session_id") == target:
                return _source_to_namespace(s, chat_id)
        return None
