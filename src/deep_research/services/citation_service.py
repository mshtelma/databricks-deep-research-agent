"""Citation service - CRUD operations for citations."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from deep_research.models.citation import Citation

logger = logging.getLogger(__name__)


class CitationService:
    """Service for managing citations.

    Provides CRUD operations for citation links between claims and evidence.
    Part of Stage 2 (Interleaved Generation) of the citation pipeline.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize citation service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(
        self,
        claim_id: UUID,
        evidence_span_id: UUID,
        confidence_score: float | None = None,
        is_primary: bool = True,
    ) -> Citation:
        """Create a new citation link.

        Args:
            claim_id: Claim ID.
            evidence_span_id: Evidence span ID.
            confidence_score: Confidence in the attribution (0.0-1.0).
            is_primary: Whether this is the primary citation.

        Returns:
            Created citation.
        """
        citation = Citation(
            claim_id=claim_id,
            evidence_span_id=evidence_span_id,
            confidence_score=confidence_score,
            is_primary=is_primary,
        )
        self._session.add(citation)
        await self._session.flush()
        await self._session.refresh(citation)
        logger.info(
            f"Created citation {citation.id} linking claim {claim_id} to span {evidence_span_id}"
        )
        return citation

    async def get(self, citation_id: UUID) -> Citation | None:
        """Get a citation by ID.

        Args:
            citation_id: Citation ID.

        Returns:
            Citation if found, None otherwise.
        """
        result = await self._session.execute(
            select(Citation).where(Citation.id == citation_id)
        )
        return result.scalar_one_or_none()

    async def get_by_claim_and_evidence(
        self,
        claim_id: UUID,
        evidence_span_id: UUID,
    ) -> Citation | None:
        """Get a citation by claim and evidence span IDs.

        Args:
            claim_id: Claim ID.
            evidence_span_id: Evidence span ID.

        Returns:
            Citation if found, None otherwise.
        """
        result = await self._session.execute(
            select(Citation).where(
                and_(
                    Citation.claim_id == claim_id,
                    Citation.evidence_span_id == evidence_span_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_claim(
        self,
        claim_id: UUID,
        include_evidence: bool = False,
    ) -> list[Citation]:
        """List all citations for a claim.

        Args:
            claim_id: Claim ID.
            include_evidence: Whether to eagerly load evidence spans.

        Returns:
            List of citations ordered by primary first, then confidence.
        """
        query = select(Citation).where(Citation.claim_id == claim_id)

        if include_evidence:
            query = query.options(
                selectinload(Citation.evidence_span).selectinload("source")
            )

        query = query.order_by(
            Citation.is_primary.desc(),
            Citation.confidence_score.desc().nulls_last(),
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def list_by_evidence(self, evidence_span_id: UUID) -> list[Citation]:
        """List all citations for an evidence span.

        Args:
            evidence_span_id: Evidence span ID.

        Returns:
            List of citations.
        """
        query = (
            select(Citation)
            .where(Citation.evidence_span_id == evidence_span_id)
            .order_by(Citation.created_at)
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def get_primary_citation(self, claim_id: UUID) -> Citation | None:
        """Get the primary citation for a claim.

        Args:
            claim_id: Claim ID.

        Returns:
            Primary citation if found, None otherwise.
        """
        result = await self._session.execute(
            select(Citation)
            .options(selectinload(Citation.evidence_span).selectinload("source"))
            .where(
                and_(
                    Citation.claim_id == claim_id,
                    Citation.is_primary == True,  # noqa: E712
                )
            )
        )
        return result.scalar_one_or_none()

    async def update_confidence(
        self,
        citation_id: UUID,
        confidence_score: float,
    ) -> Citation | None:
        """Update citation confidence score.

        Args:
            citation_id: Citation ID.
            confidence_score: New confidence score (0.0-1.0).

        Returns:
            Updated citation or None if not found.
        """
        citation = await self.get(citation_id)
        if not citation:
            return None

        citation.confidence_score = confidence_score
        await self._session.flush()
        await self._session.refresh(citation)
        logger.info(
            f"Updated confidence for citation {citation_id}: {confidence_score}"
        )
        return citation

    async def set_primary(
        self,
        claim_id: UUID,
        citation_id: UUID,
    ) -> Citation | None:
        """Set a citation as primary for a claim.

        Unsets any existing primary citation.

        Args:
            claim_id: Claim ID.
            citation_id: Citation ID to set as primary.

        Returns:
            Updated citation or None if not found.
        """
        # First, unset any existing primary
        existing_primary = await self.get_primary_citation(claim_id)
        if existing_primary and existing_primary.id != citation_id:
            existing_primary.is_primary = False

        # Set the new primary
        citation = await self.get(citation_id)
        if not citation or citation.claim_id != claim_id:
            return None

        citation.is_primary = True
        await self._session.flush()
        await self._session.refresh(citation)
        logger.info(f"Set citation {citation_id} as primary for claim {claim_id}")
        return citation

    async def get_count_by_claim(self, claim_id: UUID) -> int:
        """Get count of citations for a claim.

        Args:
            claim_id: Claim ID.

        Returns:
            Number of citations.
        """
        result = await self._session.execute(
            select(func.count(Citation.id)).where(Citation.claim_id == claim_id)
        )
        return result.scalar() or 0

    async def delete(self, citation_id: UUID) -> bool:
        """Delete a citation.

        Args:
            citation_id: Citation ID.

        Returns:
            True if deleted, False if not found.
        """
        citation = await self.get(citation_id)
        if not citation:
            return False

        await self._session.delete(citation)
        await self._session.flush()
        logger.info(f"Deleted citation {citation_id}")
        return True

    async def delete_by_claim(self, claim_id: UUID) -> int:
        """Delete all citations for a claim.

        Args:
            claim_id: Claim ID.

        Returns:
            Number of deleted citations.
        """
        from sqlalchemy import delete as sql_delete

        result = await self._session.execute(
            sql_delete(Citation).where(Citation.claim_id == claim_id)
        )
        count = result.rowcount
        await self._session.flush()
        logger.info(f"Deleted {count} citations for claim {claim_id}")
        return count

    async def replace_citation(
        self,
        claim_id: UUID,
        old_evidence_span_id: UUID,
        new_evidence_span_id: UUID,
        confidence_score: float | None = None,
    ) -> Citation | None:
        """Replace a citation's evidence span.

        Used during Stage 5 (Citation Correction).

        Args:
            claim_id: Claim ID.
            old_evidence_span_id: Original evidence span ID.
            new_evidence_span_id: New evidence span ID.
            confidence_score: Optional new confidence score.

        Returns:
            Updated citation or None if not found.
        """
        citation = await self.get_by_claim_and_evidence(claim_id, old_evidence_span_id)
        if not citation:
            return None

        citation.evidence_span_id = new_evidence_span_id
        if confidence_score is not None:
            citation.confidence_score = confidence_score

        await self._session.flush()
        await self._session.refresh(citation)
        logger.info(
            f"Replaced citation for claim {claim_id}: {old_evidence_span_id} -> {new_evidence_span_id}"
        )
        return citation
