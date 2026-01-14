"""LLM service package."""

from src.services.llm.auth import (
    # Legacy exports (backwards compatibility)
    LLMCredential,
    LLMCredentialProvider,
    TOKEN_LIFETIME,
    TOKEN_REFRESH_BUFFER,
    # New centralized auth exports
    DatabricksAuth,
    OAuthCredential,
    get_databricks_auth,
)
from src.services.llm.client import LLMClient
from src.services.llm.config import ModelConfig
from src.services.llm.truncation import (
    get_context_window_for_request,
    truncate_messages,
    truncate_text,
)
from src.services.llm.types import (
    EndpointHealth,
    LLMRequest,
    LLMResponse,
    ModelEndpoint,
    ModelRole,
    ModelTier,
    ReasoningEffort,
    SelectionStrategy,
)

__all__ = [
    # Client
    "LLMClient",
    "ModelConfig",
    # Types
    "ModelTier",
    "ModelRole",
    "ModelEndpoint",
    "EndpointHealth",
    "ReasoningEffort",
    "SelectionStrategy",
    "LLMRequest",
    "LLMResponse",
    # Centralized auth (preferred)
    "DatabricksAuth",
    "OAuthCredential",
    "get_databricks_auth",
    # Legacy auth (backwards compatibility)
    "LLMCredential",
    "LLMCredentialProvider",
    "TOKEN_LIFETIME",
    "TOKEN_REFRESH_BUFFER",
    # Truncation utilities
    "truncate_messages",
    "truncate_text",
    "get_context_window_for_request",
]
