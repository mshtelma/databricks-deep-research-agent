"""EvidenceSpan service - CRUD operations for evidence spans."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.evidence_span import EvidenceSpan

logger = logging.getLogger(__name__)


class EvidenceSpanService:
    """Service for managing evidence spans.

    Provides CRUD operations for evidence spans extracted from sources.
    Part of Stage 1 (Evidence Pre-Selection) of the citation pipeline.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize evidence span service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(
        self,
        source_id: UUID,
        quote_text: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        section_heading: str | None = None,
        relevance_score: float | None = None,
        has_numeric_content: bool = False,
    ) -> EvidenceSpan:
        """Create a new evidence span.

        Args:
            source_id: Parent source ID.
            quote_text: The exact supporting quote.
            start_offset: Start position in source content.
            end_offset: End position in source content.
            section_heading: Section/page context.
            relevance_score: Relevance ranking score (0.0-1.0).
            has_numeric_content: Whether span contains numeric data.

        Returns:
            Created evidence span.
        """
        span = EvidenceSpan(
            source_id=source_id,
            quote_text=quote_text,
            start_offset=start_offset,
            end_offset=end_offset,
            section_heading=section_heading,
            relevance_score=relevance_score,
            has_numeric_content=has_numeric_content,
        )
        self._session.add(span)
        await self._session.flush()
        await self._session.refresh(span)
        logger.info(f"Created evidence span {span.id} for source {source_id}")
        return span

    async def create_batch(
        self,
        source_id: UUID,
        spans: list[dict],
    ) -> list[EvidenceSpan]:
        """Create multiple evidence spans for a source.

        Args:
            source_id: Parent source ID.
            spans: List of span dicts with quote_text and optional fields.

        Returns:
            List of created evidence spans.
        """
        created = []
        for span_data in spans:
            span = EvidenceSpan(
                source_id=source_id,
                quote_text=span_data["quote_text"],
                start_offset=span_data.get("start_offset"),
                end_offset=span_data.get("end_offset"),
                section_heading=span_data.get("section_heading"),
                relevance_score=span_data.get("relevance_score"),
                has_numeric_content=span_data.get("has_numeric_content", False),
            )
            self._session.add(span)
            created.append(span)

        await self._session.flush()
        for span in created:
            await self._session.refresh(span)

        logger.info(f"Created {len(created)} evidence spans for source {source_id}")
        return created

    async def get(self, span_id: UUID) -> EvidenceSpan | None:
        """Get an evidence span by ID.

        Args:
            span_id: Evidence span ID.

        Returns:
            Evidence span if found, None otherwise.
        """
        result = await self._session.execute(
            select(EvidenceSpan).where(EvidenceSpan.id == span_id)
        )
        return result.scalar_one_or_none()

    async def get_with_source(self, span_id: UUID) -> EvidenceSpan | None:
        """Get an evidence span with its source eagerly loaded.

        Args:
            span_id: Evidence span ID.

        Returns:
            Evidence span with source if found, None otherwise.
        """
        result = await self._session.execute(
            select(EvidenceSpan)
            .options(selectinload(EvidenceSpan.source))
            .where(EvidenceSpan.id == span_id)
        )
        return result.scalar_one_or_none()

    async def list_by_source(
        self,
        source_id: UUID,
        min_relevance: float | None = None,
        numeric_only: bool = False,
    ) -> list[EvidenceSpan]:
        """List all evidence spans for a source.

        Args:
            source_id: Source ID.
            min_relevance: Optional minimum relevance score filter.
            numeric_only: Only return spans with numeric content.

        Returns:
            List of evidence spans ordered by relevance (descending).
        """
        conditions = [EvidenceSpan.source_id == source_id]

        if min_relevance is not None:
            conditions.append(EvidenceSpan.relevance_score >= min_relevance)

        if numeric_only:
            conditions.append(EvidenceSpan.has_numeric_content == True)  # noqa: E712

        query = (
            select(EvidenceSpan)
            .where(and_(*conditions))
            .order_by(EvidenceSpan.relevance_score.desc().nulls_last())
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def list_by_sources(
        self,
        source_ids: list[UUID],
        min_relevance: float | None = None,
        limit: int | None = None,
    ) -> list[EvidenceSpan]:
        """List evidence spans from multiple sources.

        Args:
            source_ids: List of source IDs.
            min_relevance: Optional minimum relevance score filter.
            limit: Maximum number of spans to return.

        Returns:
            List of evidence spans ordered by relevance (descending).
        """
        if not source_ids:
            return []

        conditions = [EvidenceSpan.source_id.in_(source_ids)]

        if min_relevance is not None:
            conditions.append(EvidenceSpan.relevance_score >= min_relevance)

        query = (
            select(EvidenceSpan)
            .options(selectinload(EvidenceSpan.source))
            .where(and_(*conditions))
            .order_by(EvidenceSpan.relevance_score.desc().nulls_last())
        )

        if limit:
            query = query.limit(limit)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def update_relevance(
        self,
        span_id: UUID,
        relevance_score: float,
    ) -> EvidenceSpan | None:
        """Update evidence span relevance score.

        Args:
            span_id: Evidence span ID.
            relevance_score: New relevance score (0.0-1.0).

        Returns:
            Updated evidence span or None if not found.
        """
        span = await self.get(span_id)
        if not span:
            return None

        span.relevance_score = relevance_score
        await self._session.flush()
        await self._session.refresh(span)
        logger.info(f"Updated relevance for span {span_id}: {relevance_score}")
        return span

    async def get_count_by_source(self, source_id: UUID) -> int:
        """Get count of evidence spans for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of evidence spans.
        """
        result = await self._session.execute(
            select(func.count(EvidenceSpan.id)).where(
                EvidenceSpan.source_id == source_id
            )
        )
        return result.scalar() or 0

    async def delete(self, span_id: UUID) -> bool:
        """Delete an evidence span.

        Args:
            span_id: Evidence span ID.

        Returns:
            True if deleted, False if not found.
        """
        span = await self.get(span_id)
        if not span:
            return False

        await self._session.delete(span)
        await self._session.flush()
        logger.info(f"Deleted evidence span {span_id}")
        return True

    async def delete_by_source(self, source_id: UUID) -> int:
        """Delete all evidence spans for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of deleted spans.
        """
        from sqlalchemy import delete as sql_delete

        result = await self._session.execute(
            sql_delete(EvidenceSpan).where(EvidenceSpan.source_id == source_id)
        )
        count = result.rowcount
        await self._session.flush()
        logger.info(f"Deleted {count} evidence spans for source {source_id}")
        return count
