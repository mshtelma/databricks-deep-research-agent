"""Pydantic models for AIditor API."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Edit Types
# =============================================================================


class EditType(str, Enum):
    """Types of edit operations."""

    REMOVE = "REMOVE"
    LESS = "LESS"
    MORE = "MORE"
    CUSTOM = "CUSTOM"


class EditInstruction(BaseModel):
    """A single edit instruction for text modification."""

    type: EditType
    text: str = Field(..., description="The text to be modified")
    instruction: Optional[str] = Field(
        None, description="Custom instruction for CUSTOM type edits"
    )


# =============================================================================
# Chat / LLM Models
# =============================================================================


class ChatRequest(BaseModel):
    """Request payload for LLM processing."""

    model: str = Field(..., description="Model endpoint name")
    original_markdown: str = Field(..., description="The original markdown document")
    edits: list[EditInstruction] = Field(..., description="List of edit instructions")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: Optional[int] = None


class DebugInfo(BaseModel):
    """Debug information for the LLM request/response."""

    system_prompt: str = Field(..., description="System prompt sent to the LLM")
    user_message: str = Field(..., description="User message sent to the LLM")
    raw_response: str = Field(..., description="Raw content returned by the LLM (before stripping)")


class ChatResponse(BaseModel):
    """Response from LLM processing."""

    content: str = Field(..., description="The edited markdown document")
    model: str = Field(..., description="Model used for processing")
    usage: Optional[UsageInfo] = None
    debug: Optional[DebugInfo] = None


# =============================================================================
# Model Info
# =============================================================================


class ModelStatus(str, Enum):
    """Serving endpoint status."""

    READY = "READY"
    NOT_READY = "NOT_READY"
    PENDING = "PENDING"


class ModelInfo(BaseModel):
    """Information about a serving endpoint model."""

    name: str = Field(..., description="Endpoint name")
    display_name: str = Field(..., description="Human-readable name")
    status: ModelStatus = Field(..., description="Endpoint status")
    task: str = Field(default="llm/v1/chat", description="Task type")


class ModelsResponse(BaseModel):
    """Response containing available models."""

    models: list[ModelInfo]
    default_model: Optional[str] = None


# =============================================================================
# MCP Endpoint Types
# =============================================================================


class MCPEndpointType(str, Enum):
    """Types of MCP endpoints."""

    GENIE = "genie"
    VECTOR = "vector"
    EXTERNAL = "external"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"


# =============================================================================
# Genie Space Models
# =============================================================================


class GenieSpaceInfo(BaseModel):
    """Information about a Genie Space."""

    id: str = Field(..., description="Genie Space ID")
    name: str = Field(..., description="Display name")
    tables: list[str] = Field(default_factory=list, description="Associated tables")
    status: str = Field(default="active", description="Status")


class GenieQueryRequest(BaseModel):
    """Request to query a Genie Space."""

    space_id: str = Field(..., description="Genie Space ID")
    query: str = Field(..., description="Natural language query")


class GenieQueryResponse(BaseModel):
    """Response from Genie Space query."""

    status: str = Field(..., description="Query status")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    columns: list[str] = Field(default_factory=list, description="Column names")
    data: list[list[Any]] = Field(default_factory=list, description="Query results")
    markdown_table: str = Field(..., description="Results as markdown table")
    error: Optional[str] = None


# =============================================================================
# Vector Search Models
# =============================================================================


class VectorIndexInfo(BaseModel):
    """Information about a Vector Search index."""

    name: str = Field(..., description="Index name")
    endpoint: str = Field(..., description="Endpoint name")
    num_docs: int = Field(default=0, description="Number of documents")


class VectorSearchRequest(BaseModel):
    """Request for Vector Search query."""

    index_name: str = Field(..., description="Index name")
    query: str = Field(..., description="Search query text")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results")


class VectorSearchResult(BaseModel):
    """A single vector search result."""

    text: str = Field(..., description="Document text")
    source: Optional[str] = Field(None, description="Source document path")
    score: float = Field(..., ge=0, le=1, description="Relevance score")


class VectorSearchResponse(BaseModel):
    """Response from Vector Search query."""

    results: list[VectorSearchResult]
    markdown_list: str = Field(..., description="Results as markdown list")
    error: Optional[str] = None


# =============================================================================
# External Connection Models (Tavily)
# =============================================================================


class ExternalConnectionInfo(BaseModel):
    """Information about an external MCP connection."""

    name: str = Field(..., description="Connection name")
    type: str = Field(..., description="Connection type (e.g., tavily)")
    status: str = Field(default="active", description="Connection status")


class ExternalQueryRequest(BaseModel):
    """Request for external API query."""

    connection_name: str = Field(..., description="Connection name")
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=3, ge=1, le=10, description="Max results")


class ExternalSearchResult(BaseModel):
    """A single external search result."""

    title: str
    url: str
    snippet: str


class ExternalQueryResponse(BaseModel):
    """Response from external API query."""

    results: list[ExternalSearchResult]
    markdown_summary: str = Field(..., description="Results as markdown")
    error: Optional[str] = None


# =============================================================================
# Knowledge Assistant Models
# =============================================================================


class KnowledgeAssistantInfo(BaseModel):
    """Information about a Knowledge Assistant."""

    tile_id: str = Field(..., description="KA tile ID")
    name: str = Field(..., description="Display name")
    endpoint_name: str = Field(..., description="Model serving endpoint name")
    status: str = Field(default="active", description="Status (ONLINE, PROVISIONING, etc.)")


class KnowledgeAssistantQueryRequest(BaseModel):
    """Request to query a Knowledge Assistant."""

    tile_id: str = Field(..., description="KA tile ID")
    query: str = Field(..., description="Natural language query")


class KnowledgeAssistantQueryResponse(BaseModel):
    """Response from Knowledge Assistant query."""

    status: str = Field(..., description="Query status")
    answer: str = Field(..., description="The KA's response")
    sources: list[str] = Field(default_factory=list, description="Source documents referenced")
    error: Optional[str] = None


# =============================================================================
# Combined MCP Endpoints
# =============================================================================


class MCPEndpoints(BaseModel):
    """All available MCP endpoints."""

    genie_spaces: list[GenieSpaceInfo] = Field(default_factory=list)
    vector_indexes: list[VectorIndexInfo] = Field(default_factory=list)
    external_connections: list[ExternalConnectionInfo] = Field(default_factory=list)
    knowledge_assistants: list[KnowledgeAssistantInfo] = Field(default_factory=list)
