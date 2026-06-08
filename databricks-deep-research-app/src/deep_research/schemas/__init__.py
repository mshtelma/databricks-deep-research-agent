"""Pydantic schemas package."""

from deep_research.schemas.agent import (
    AgentQueryRequest,
    AgentQueryResponse,
    ContextMessage,
    QueryClassification,
    SourceResponse,
)
from deep_research.schemas.chat import (
    ChatCreate,
    ChatListResponse,
    ChatResponse,
    ChatUpdate,
)
from deep_research.schemas.common import ErrorResponse, HealthResponse, PaginatedResponse
from deep_research.schemas.deployment import (
    ActiveDeploymentsErrorResponse,
    ActiveDeploymentSummary,
    BatchDeploymentConfig,
    CanRunFastResponse,
    CanRunSlowResponse,
    CreateDeploymentRequest,
    DeploymentConfig,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentStatusResponse,
    InAppDeploymentConfig,
    MlflowAgentDeploymentConfig,
    ShellAppDeploymentConfig,
)
from deep_research.schemas.feedback import FeedbackRequest, FeedbackResponse
from deep_research.schemas.message import (
    EditMessageRequest,
    EditMessageResponse,
    MessageListResponse,
    MessageResponse,
    RegenerateResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from deep_research.schemas.preferences import (
    UpdatePreferencesRequest,
    UserPreferencesResponse,
)
from deep_research.schemas.research import (
    CancelResearchResponse,
    ResearchPlan,
    ResearchSession,
)
from deep_research.schemas.research_request import (
    ResearchRequest,
    ResearchRequestWithChat,
)
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    ClarificationNeededEvent,
    PlanCreatedEvent,
    PlanForReview,
    PlanReviewEvent,
    PlanReviewResponseEvent,
    PlanReviewTimeoutEvent,
    PlanStepForReview,
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
    # Plan Review (007-enterprise-data-sources)
    "PlanReviewEvent",
    "PlanReviewResponseEvent",
    "PlanReviewTimeoutEvent",
    "PlanForReview",
    "PlanStepForReview",
    # Research Request (007-enterprise-data-sources)
    "ResearchRequest",
    "ResearchRequestWithChat",
    # Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Preferences
    "UpdatePreferencesRequest",
    "UserPreferencesResponse",
    # Deployment (Phase 1 backend)
    "ActiveDeploymentSummary",
    "ActiveDeploymentsErrorResponse",
    "BatchDeploymentConfig",
    "CanRunFastResponse",
    "CanRunSlowResponse",
    "CreateDeploymentRequest",
    "DeploymentConfig",
    "DeploymentListResponse",
    "DeploymentResponse",
    "DeploymentStatusResponse",
    "InAppDeploymentConfig",
    "MlflowAgentDeploymentConfig",
    "ShellAppDeploymentConfig",
]
