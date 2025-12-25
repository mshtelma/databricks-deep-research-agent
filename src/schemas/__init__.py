"""Pydantic schemas package."""

from src.schemas.agent import (
    AgentQueryRequest,
    AgentQueryResponse,
    ContextMessage,
    QueryClassification,
    SourceResponse,
)
from src.schemas.chat import (
    ChatCreate,
    ChatListResponse,
    ChatResponse,
    ChatUpdate,
)
from src.schemas.common import ErrorResponse, HealthResponse, PaginatedResponse
from src.schemas.feedback import FeedbackRequest, FeedbackResponse
from src.schemas.message import (
    EditMessageRequest,
    EditMessageResponse,
    MessageListResponse,
    MessageResponse,
    RegenerateResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from src.schemas.preferences import (
    UpdatePreferencesRequest,
    UserPreferencesResponse,
)
from src.schemas.research import (
    CancelResearchResponse,
    ResearchPlan,
    ResearchSession,
)
from src.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    ClarificationNeededEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    ResearchCompletedEvent,
    StepCompletedEvent,
    StepStartedEvent,
    StreamErrorEvent,
    StreamEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
)

__all__ = [
    # Common
    "ErrorResponse",
    "HealthResponse",
    "PaginatedResponse",
    # Chat
    "ChatCreate",
    "ChatUpdate",
    "ChatResponse",
    "ChatListResponse",
    # Message
    "SendMessageRequest",
    "SendMessageResponse",
    "EditMessageRequest",
    "EditMessageResponse",
    "RegenerateResponse",
    "MessageResponse",
    "MessageListResponse",
    # Agent
    "AgentQueryRequest",
    "AgentQueryResponse",
    "ContextMessage",
    "QueryClassification",
    "SourceResponse",
    # Research
    "ResearchSession",
    "ResearchPlan",
    "CancelResearchResponse",
    # Streaming
    "StreamEvent",
    "AgentStartedEvent",
    "AgentCompletedEvent",
    "ClarificationNeededEvent",
    "PlanCreatedEvent",
    "StepStartedEvent",
    "StepCompletedEvent",
    "ReflectionDecisionEvent",
    "SynthesisStartedEvent",
    "SynthesisProgressEvent",
    "ResearchCompletedEvent",
    "StreamErrorEvent",
    # Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Preferences
    "UpdatePreferencesRequest",
    "UserPreferencesResponse",
]
