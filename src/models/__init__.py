"""Database models package."""

from src.models.audit_log import AuditAction, AuditLog
from src.models.chat import Chat, ChatStatus
from src.models.citation import Citation
from src.models.citation_correction import CitationCorrection
from src.models.claim import Claim
from src.models.enums import (
    ClaimType,
    ConfidenceLevel,
    CorrectionType,
    DerivationType,
    VerificationVerdict,
)
from src.models.evidence_span import EvidenceSpan
from src.models.message import Message, MessageRole
from src.models.message_feedback import FeedbackRating, MessageFeedback
from src.models.numeric_claim import NumericClaim
from src.models.research_event import ResearchEvent
from src.models.research_session import (
    ResearchDepth,
    ResearchSession,
    ResearchSessionStatus,
    ResearchStatus,
)
from src.models.source import Source
from src.models.user_preferences import UserPreferences
from src.models.verification_summary import VerificationSummary

__all__ = [
    # Chat
    "Chat",
    "ChatStatus",
    # Message
    "Message",
    "MessageRole",
    # Research
    "ResearchSession",
    "ResearchSessionStatus",
    "ResearchStatus",
    "ResearchDepth",
    "ResearchEvent",
    # Source
    "Source",
    # User Preferences
    "UserPreferences",
    # Feedback
    "MessageFeedback",
    "FeedbackRating",
    # Audit
    "AuditLog",
    "AuditAction",
    # Citation verification enums
    "ClaimType",
    "VerificationVerdict",
    "ConfidenceLevel",
    "CorrectionType",
    "DerivationType",
    # Citation verification models
    "Claim",
    "EvidenceSpan",
    "Citation",
    "NumericClaim",
    "CitationCorrection",
    "VerificationSummary",
]
