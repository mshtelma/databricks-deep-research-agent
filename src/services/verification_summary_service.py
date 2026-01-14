"""VerificationSummary service - aggregated verification statistics."""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.app_config import get_app_config
from src.models.citation_correction import CitationCorrection
from src.models.claim import Claim
from src.models.verification_summary import VerificationSummary

logger = logging.getLogger(__name__)


class VerificationSummaryService:
    """Service for managing verification summaries.

    Provides computation and caching of verification statistics for messages.
    Created after all claims in a message have been verified (Stage 4).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize verification summary service.

        Args:
            session: Database session.
        """
        self._session = session
        self._config = get_app_config().citation_verification

    async def get(self, message_id: UUID) -> VerificationSummary | None:
        """Get verification summary for a message.

        Args:
            message_id: Message ID.

        Returns:
            VerificationSummary if found, None otherwise.
        """
        result = await self._session.execute(
            select(VerificationSummary).where(
                VerificationSummary.message_id == message_id
            )
        )
        return result.scalar_one_or_none()

    async def compute_summary(self, message_id: UUID) -> VerificationSummary:
        """Compute verification summary for a message.

        Aggregates verdict counts from all claims and creates/updates
        the cached summary.

        Args:
            message_id: Message ID.

        Returns:
            Created or updated VerificationSummary.
        """
        # Get verdict counts
        verdict_result = await self._session.execute(
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

        for verdict, count in verdict_result.all():
            if verdict is None:
                counts["unverified"] = count
            else:
                counts[verdict] = count

        # Get abstained count
        abstained_result = await self._session.execute(
            select(func.count(Claim.id)).where(
                Claim.message_id == message_id,
                Claim.abstained == True,  # noqa: E712
            )
        )
        abstained_count = abstained_result.scalar() or 0

        # Get correction count
        correction_result = await self._session.execute(
            select(func.count(CitationCorrection.id))
            .join(Claim, CitationCorrection.claim_id == Claim.id)
            .where(
                Claim.message_id == message_id,
                CitationCorrection.correction_type != "keep",
            )
        )
        correction_count = correction_result.scalar() or 0

        # Compute rates
        total_claims = sum(counts.values())
        verified_count = total_claims - abstained_count

        unsupported_rate = (
            counts["unsupported"] / verified_count if verified_count > 0 else 0.0
        )
        contradicted_rate = (
            counts["contradicted"] / verified_count if verified_count > 0 else 0.0
        )

        # Check warning thresholds
        warning = (
            unsupported_rate > self._config.unsupported_claim_warning_threshold
            or contradicted_rate > 0.05
        )

        # Check if summary already exists
        existing = await self.get(message_id)

        if existing:
            # Update existing summary
            existing.total_claims = total_claims
            existing.supported_count = counts["supported"]
            existing.partial_count = counts["partial"]
            existing.unsupported_count = counts["unsupported"]
            existing.contradicted_count = counts["contradicted"]
            existing.abstained_count = abstained_count
            existing.unsupported_rate = unsupported_rate
            existing.contradicted_rate = contradicted_rate
            existing.warning = warning
            existing.citation_corrections = correction_count
            await self._session.flush()
            await self._session.refresh(existing)
            logger.info(f"Updated verification summary for message {message_id}")
            return existing

        # Create new summary
        summary = VerificationSummary(
            message_id=message_id,
            total_claims=total_claims,
            supported_count=counts["supported"],
            partial_count=counts["partial"],
            unsupported_count=counts["unsupported"],
            contradicted_count=counts["contradicted"],
            abstained_count=abstained_count,
            unsupported_rate=unsupported_rate,
            contradicted_rate=contradicted_rate,
            warning=warning,
            citation_corrections=correction_count,
        )
        self._session.add(summary)
        await self._session.flush()
        await self._session.refresh(summary)
        logger.info(f"Created verification summary for message {message_id}")
        return summary

    async def check_warning_threshold(self, message_id: UUID) -> bool:
        """Check if message exceeds warning thresholds.

        Args:
            message_id: Message ID.

        Returns:
            True if warning threshold is exceeded.
        """
        summary = await self.get(message_id)
        if not summary:
            summary = await self.compute_summary(message_id)

        return summary.warning

    async def get_or_compute(self, message_id: UUID) -> VerificationSummary:
        """Get existing summary or compute a new one.

        Args:
            message_id: Message ID.

        Returns:
            VerificationSummary (existing or newly computed).
        """
        summary = await self.get(message_id)
        if summary:
            return summary
        return await self.compute_summary(message_id)

    async def invalidate(self, message_id: UUID) -> bool:
        """Invalidate (delete) cached summary for a message.

        Call this when claims are added, modified, or deleted.

        Args:
            message_id: Message ID.

        Returns:
            True if deleted, False if not found.
        """
        summary = await self.get(message_id)
        if not summary:
            return False

        await self._session.delete(summary)
        await self._session.flush()
        logger.info(f"Invalidated verification summary for message {message_id}")
        return True

    async def get_correction_metrics(self, message_id: UUID) -> dict[str, int]:
        """Get citation correction metrics for a message.

        Args:
            message_id: Message ID.

        Returns:
            Dict with correction counts by type.
        """
        result = await self._session.execute(
            select(CitationCorrection.correction_type, func.count(CitationCorrection.id))
            .join(Claim, CitationCorrection.claim_id == Claim.id)
            .where(Claim.message_id == message_id)
            .group_by(CitationCorrection.correction_type)
        )

        metrics = {
            "keep": 0,
            "replace": 0,
            "remove": 0,
            "add_alternate": 0,
            "total": 0,
        }

        for correction_type, count in result.all():
            metrics[correction_type] = count
            metrics["total"] += count

        return metrics
