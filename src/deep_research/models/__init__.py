"""Database models package.

JSONB Migration (Migration 011):
Citation verification models (Claim, Citation, EvidenceSpan, NumericClaim,
CitationCorrection, VerificationSummary) have been removed. This data is now
stored in the verification_data JSONB column on research_sessions.

The enum types (ClaimType, VerificationVerdict, etc.) are kept for use in schemas.
"""

from deep_research.models.audit_log import AuditAction, AuditLog
from deep_research.models.chat import Chat, ChatStatus
from deep_research.models.enums import (
    ClaimType,
    ConfidenceLevel,
    CorrectionType,
    DerivationType,
    VerificationVerdict,
)
from deep_research.models.message import Message, MessageRole
from deep_research.models.message_feedback import FeedbackRating, MessageFeedback
from deep_research.models.research_event import ResearchEvent
from deep_research.models.research_session import (
    ResearchDepth,
    ResearchSession,
    ResearchSessionStatus,
    ResearchStatus,
)
from deep_research.models.source import Source
from deep_research.models.user_preferences import UserPreferences

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
    # Citation verification enums (kept for schema compatibility)
    "ClaimType",
    "VerificationVerdict",
    "ConfidenceLevel",
    "CorrectionType",
    "DerivationType",
]
