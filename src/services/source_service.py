"""Source service - manages research sources."""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.source import Source

logger = logging.getLogger(__name__)

# Database constraint: sources.title is VARCHAR(500)
MAX_TITLE_LENGTH = 500


def _truncate_title(title: str | None) -> str | None:
    """Truncate title to fit VARCHAR(500) constraint.

    Args:
        title: Source title from API.

    Returns:
        Truncated title with ellipsis if over limit, original otherwise.
    """
    if title and len(title) > MAX_TITLE_LENGTH:
        return title[: MAX_TITLE_LENGTH - 3] + "..."
    return title


class SourceService:
    """Service for managing research sources."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize source service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(
        self,
        research_session_id: UUID,
        url: str,
        title: str | None = None,
        snippet: str | None = None,
        content: str | None = None,
        relevance_score: float | None = None,
    ) -> Source:
        """Create a new source.

        Args:
            research_session_id: Research session ID.
            url: Source URL.
            title: Source title.
            snippet: Source snippet.
            content: Full content.
            relevance_score: Relevance score.

        Returns:
            Created source.
        """
        source = Source(
            research_session_id=research_session_id,
            url=url,
            title=_truncate_title(title),
            snippet=snippet,
            content=content,
            relevance_score=relevance_score,
        )
        self._session.add(source)
        await self._session.flush()
        await self._session.refresh(source)
        logger.debug(f"Created source {source.id} for session {research_session_id}")
        return source

    async def create_many(
        self,
        research_session_id: UUID,
        sources: list[dict[str, str | float | None]],
    ) -> list[Source]:
        """Create multiple sources.

        Args:
            research_session_id: Research session ID.
            sources: List of source data dicts.

        Returns:
            Created sources.
        """
        created = []
        for source_data in sources:
            source = Source(
                research_session_id=research_session_id,
                url=str(source_data.get("url", "")),
                title=_truncate_title(source_data.get("title")),  # type: ignore[arg-type]
                snippet=source_data.get("snippet"),
                content=source_data.get("content"),
                relevance_score=source_data.get("relevance_score"),
            )
            self._session.add(source)
            created.append(source)

        await self._session.flush()
        for source in created:
            await self._session.refresh(source)

        logger.info(
            f"Created {len(created)} sources for session {research_session_id}"
        )
        return created

    async def get(self, source_id: UUID) -> Source | None:
        """Get a source by ID.

        Args:
            source_id: Source ID.

        Returns:
            Source if found, None otherwise.
        """
        result = await self._session.execute(
            select(Source).where(Source.id == source_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(
        self,
        research_session_id: UUID,
        limit: int = 100,
    ) -> list[Source]:
        """List sources for a research session.

        Args:
            research_session_id: Research session ID.
            limit: Maximum number of sources.

        Returns:
            List of sources.
        """
        query = (
            select(Source)
            .where(Source.research_session_id == research_session_id)
            .order_by(Source.relevance_score.desc().nullslast())
            .limit(limit)
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def update_content(
        self,
        source_id: UUID,
        content: str,
    ) -> Source | None:
        """Update source content.

        Args:
            source_id: Source ID.
            content: Full content.

        Returns:
            Updated source or None if not found.
        """
        source = await self.get(source_id)
        if not source:
            return None

        source.content = content
        await self._session.flush()
        await self._session.refresh(source)
        return source

    async def get_by_url(
        self,
        research_session_id: UUID,
        url: str,
    ) -> Source | None:
        """Get source by URL within a session.

        Args:
            research_session_id: Research session ID.
            url: Source URL.

        Returns:
            Source if found, None otherwise.
        """
        result = await self._session.execute(
            select(Source).where(
                Source.research_session_id == research_session_id,
                Source.url == url,
            )
        )
        return result.scalar_one_or_none()
