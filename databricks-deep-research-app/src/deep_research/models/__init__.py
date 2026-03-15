"""Database models package.

JSONB Migration (Migration 011):
Citation verification models (Claim, Citation, EvidenceSpan, NumericClaim,
CitationCorrection, VerificationSummary) have been removed. This data is now
stored in the verification_data JSONB column on research_sessions.

The enum types (ClaimType, VerificationVerdict, etc.) are kept for use in schemas.
"""

from deep_research.models.audit_log import AuditAction, AuditLog
from deep_research.models.chat import Chat, ChatStatus, ChatType
from deep_research.models.custom_agent import (
    AgentOutputFormat,
    AgentPresetStep,
    AgentResearchDepth,
    AgentSourceScope,
    AgentVisibility,
    AgentWorkflowMode,
    CustomAgent,
)
from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
    DataSourceVisibility,
    UserDataSource,
)
from deep_research.models.enums import (
    ClaimType,
    ConfidenceLevel,
    CorrectionType,
    DerivationType,
    VerificationVerdict,
)
from deep_research.models.incognito_session import IncognitoSession
from deep_research.models.message import Message, MessageRole
from deep_research.models.message_feedback import FeedbackRating, MessageFeedback
from deep_research.models.prompt_template import (
    PromptTemplate,
    TemplateType,
    TemplateVisibility,
)
from deep_research.models.research_event import ResearchEvent
from deep_research.models.research_session import (
    ResearchDepth,
    ResearchSession,
    ResearchSessionStatus,
    ResearchStatus,
)
from deep_research.models.source import Source
from deep_research.models.uploaded_file import (
    FileChunk,
    FileProcessingStatus,
    FileType,
    UploadedFile,
)
from deep_research.models.user_preferences import UserPreferences

__all__ = [
    # Chat
    "Chat",
    "ChatStatus",
    "ChatType",
    # Data Source (007-enterprise-data-sources)
    "UserDataSource",
    "DataSourceType",
    "DataSourceVisibility",
    "DataSourceValidationStatus",
    # Incognito Session
    "IncognitoSession",
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
    # Uploaded File (US7)
    "UploadedFile",
    "FileChunk",
    "FileType",
    "FileProcessingStatus",
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
    # Prompt Template (US5)
    "PromptTemplate",
    "TemplateType",
    "TemplateVisibility",
    # Custom Agent (US6)
    "CustomAgent",
    "AgentPresetStep",
    "AgentVisibility",
    "AgentSourceScope",
    "AgentWorkflowMode",
    "AgentOutputFormat",
    "AgentResearchDepth",
]
