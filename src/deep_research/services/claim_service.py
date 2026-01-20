"""Claim service - CRUD operations for claims."""

import logging
from uuid import UUID

from sqlalchemy import and_, func, select

from deep_research.models.claim import Claim
from deep_research.models.enums import ClaimType, ConfidenceLevel, VerificationVerdict
from deep_research.services.base import BaseRepository
from deep_research.services.loading import CLAIM_WITH_CITATIONS_OPTIONS

logger = logging.getLogger(__name__)


class ClaimService(BaseRepository[Claim]):
    """Service for managing claims.

    Provides CRUD operations for claims extracted from agent messages.
    Part of the claim-level citation verification pipeline.

    Extends BaseRepository[Claim] for standard CRUD operations.
    """

    model = Claim

    async def create(
        self,
        message_id: UUID,
        claim_text: str,
        claim_type: ClaimType | str,
        position_start: int,
        position_end: int,
        confidence_level: ConfidenceLevel | str | None = None,
        verification_verdict: VerificationVerdict | str | None = None,
        verification_reasoning: str | None = None,
        abstained: bool = False,
        citation_key: str | None = None,
        citation_keys: list[str] | None = None,
    ) -> Claim:
        """Create a new claim.

        Args:
            message_id: Parent message ID.
            claim_text: The extracted claim text.
            claim_type: Type of claim ('general' or 'numeric').
            position_start: Start character offset in message.
            position_end: End character offset in message.
            confidence_level: HaluGate-style confidence level.
            verification_verdict: Verification result.
            verification_reasoning: Explanation for verdict.
            abstained: Whether verification was abstained.
            citation_key: Primary citation key (e.g., "Arxiv", "Zhipu").
            citation_keys: All citation keys for multi-marker sentences.

        Returns:
            Created claim.
        """
        # Convert enums to strings if needed
        type_value = claim_type.value if isinstance(claim_type, ClaimType) else claim_type
        conf_value = (
            confidence_level.value
            if isinstance(confidence_level, ConfidenceLevel)
            else confidence_level
        )
        verdict_value = (
            verification_verdict.value
            if isinstance(verification_verdict, VerificationVerdict)
            else verification_verdict
        )

        claim = Claim(
            message_id=message_id,
            claim_text=claim_text,
            claim_type=type_value,
            position_start=position_start,
            position_end=position_end,
            confidence_level=conf_value,
            verification_verdict=verdict_value,
            verification_reasoning=verification_reasoning,
            abstained=abstained,
            citation_key=citation_key,
            citation_keys=citation_keys,
        )
        claim = await self.add(claim)
        logger.info(f"Created claim {claim.id} for message {message_id}")
        return claim

    async def get_with_citations(self, claim_id: UUID) -> Claim | None:
        """Get a claim with its citations eagerly loaded.

        Args:
            claim_id: Claim ID.

        Returns:
            Claim with citations if found, None otherwise.
        """
        # Use shared eager-loading options from loading.py
        result = await self._session.execute(
            select(Claim)
            .options(*CLAIM_WITH_CITATIONS_OPTIONS)
            .where(Claim.id == claim_id)
        )
        return result.scalar_one_or_none()

    async def list_by_message(
        self,
        message_id: UUID,
        include_citations: bool = False,
    ) -> list[Claim]:
        """List all claims for a message.

        Args:
            message_id: Message ID.
            include_citations: Whether to eagerly load citations.

        Returns:
            List of claims ordered by position.
        """
        query = select(Claim).where(Claim.message_id == message_id)

        if include_citations:
            # Use shared eager-loading options from loading.py
            query = query.options(*CLAIM_WITH_CITATIONS_OPTIONS)

        query = query.order_by(Claim.position_start)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def update_verification(
        self,
        claim_id: UUID,
        verdict: VerificationVerdict | str,
        reasoning: str | None = None,
        abstained: bool = False,
    ) -> Claim | None:
        """Update claim verification result.

        Args:
            claim_id: Claim ID.
            verdict: Verification verdict.
            reasoning: Explanation for the verdict.
            abstained: Whether verification was abstained.

        Returns:
            Updated claim or None if not found.
        """
        claim = await self.get(claim_id)
        if not claim:
            return None

        verdict_value = (
            verdict.value if isinstance(verdict, VerificationVerdict) else verdict
        )

        claim.verification_verdict = verdict_value
        claim.verification_reasoning = reasoning
        claim.abstained = abstained
        claim = await self.update(claim)
        logger.info(f"Updated verification for claim {claim_id}: {verdict_value}")
        return claim

    async def update_confidence(
        self,
        claim_id: UUID,
        confidence_level: ConfidenceLevel | str,
    ) -> Claim | None:
        """Update claim confidence level.

        Args:
            claim_id: Claim ID.
            confidence_level: New confidence level.

        Returns:
            Updated claim or None if not found.
        """
        claim = await self.get(claim_id)
        if not claim:
            return None

        conf_value = (
            confidence_level.value
            if isinstance(confidence_level, ConfidenceLevel)
            else confidence_level
        )

        claim.confidence_level = conf_value
        claim = await self.update(claim)
        logger.info(f"Updated confidence for claim {claim_id}: {conf_value}")
        return claim

    async def get_verdict_counts(self, message_id: UUID) -> dict[str, int]:
        """Get verdict counts for a message.

        Args:
            message_id: Message ID.

        Returns:
            Dict with counts by verdict.
        """
        result = await self._session.execute(
            select(Claim.verification_verdict, func.count(Claim.id))
            .where(Claim.message_id == message_id)
            .group_by(Claim.verification_verdict)
        )

        counts = {
            "supported": 0,
            "partial": 0,
            "unsupported": 0,
            "contradicted": 0,
            "unverified": 0,
        }

        for verdict, count in result.all():
            if verdict is None:
                counts["unverified"] = count
            else:
                counts[verdict] = count

        return counts

    async def get_abstained_count(self, message_id: UUID) -> int:
        """Get count of abstained claims for a message.

        Args:
            message_id: Message ID.

        Returns:
            Number of abstained claims.
        """
        result = await self._session.execute(
            select(func.count(Claim.id)).where(
                and_(
                    Claim.message_id == message_id,
                    Claim.abstained == True,  # noqa: E712
                )
            )
        )
        return result.scalar() or 0

    async def delete_claim(self, claim_id: UUID) -> bool:
        """Delete a claim by ID.

        Args:
            claim_id: Claim ID.

        Returns:
            True if deleted, False if not found.
        """
        deleted = await self.delete_by_id(claim_id)
        if deleted:
            logger.info(f"Deleted claim {claim_id}")
        return deleted
