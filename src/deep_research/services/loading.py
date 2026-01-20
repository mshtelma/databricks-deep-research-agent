"""Shared SQLAlchemy eager-loading options.

This module centralizes the repeated selectinload chains used across
services to prevent N+1 query problems with relationships.

Usage:
    from deep_research.services.loading import CLAIM_WITH_CITATIONS_OPTIONS

    query = select(Claim).options(*CLAIM_WITH_CITATIONS_OPTIONS).where(...)
"""

from sqlalchemy.orm import selectinload

from deep_research.models.citation import Citation
from deep_research.models.claim import Claim
from deep_research.models.evidence_span import EvidenceSpan
from deep_research.models.message import Message

# Full claim graph: claim -> citations -> evidence_span -> source
# Also includes numeric_detail and corrections
# Used by ClaimService.get_with_citations() and list_by_message()
CLAIM_WITH_CITATIONS_OPTIONS = (
    selectinload(Claim.citations)
    .selectinload(Citation.evidence_span)
    .selectinload(EvidenceSpan.source),
    selectinload(Claim.numeric_detail),
    selectinload(Claim.corrections),
)

# Message with chat relationship for authorization checks
# Used by verify_message_ownership() in authorization.py
MESSAGE_WITH_CHAT_OPTIONS = (selectinload(Message.chat),)

# Message with research session for list responses
# Used by MessageService for message list with session metadata
MESSAGE_WITH_SESSION_OPTIONS = (selectinload(Message.research_session),)

# Evidence span with source for displaying quotes with source metadata
EVIDENCE_WITH_SOURCE_OPTIONS = (selectinload(EvidenceSpan.source),)
