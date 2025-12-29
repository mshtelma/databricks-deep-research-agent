"""LLM service type definitions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ModelTier(str, Enum):
    """Model capability tiers."""

    SIMPLE = "simple"  # Fast, low-latency, low-cost
    ANALYTICAL = "analytical"  # Balanced reasoning, moderate cost
    COMPLEX = "complex"  # Extended thinking, higher cost


class ReasoningEffort(str, Enum):
    """Reasoning effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionStrategy(str, Enum):
    """Endpoint selection strategy."""

    PRIORITY = "priority"  # Use first healthy endpoint
    ROUND_ROBIN = "round_robin"  # Rotate through endpoints


@dataclass
class ModelEndpoint:
    """Configuration for a model endpoint."""

    id: str
    endpoint_identifier: str
    max_context_window: int
    tokens_per_minute: int

    # Optional overrides (None means inherit from role)
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = None
    supports_structured_output: bool = False
    # Some models (e.g., GPT-5) don't support temperature parameter
    supports_temperature: bool = True


@dataclass
class ModelRole:
    """Configuration for a model role (tier)."""

    name: str
    endpoints: list[str]  # Endpoint IDs in priority order
    temperature: float = 0.7
    max_tokens: int = 8000
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = None
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True


@dataclass
class EndpointHealth:
    """Runtime health state for an endpoint (in-memory only)."""

    endpoint_id: str
    is_healthy: bool = True
    last_success_at: datetime | None = None
    consecutive_errors: int = 0
    rate_limited_until: datetime | None = None
    tokens_used_this_minute: int = 0
    minute_started_at: datetime | None = None

    def mark_success(self) -> None:
        """Mark endpoint as successful."""
        self.is_healthy = True
        self.last_success_at = datetime.now(UTC)
        self.consecutive_errors = 0
        self.rate_limited_until = None

    def mark_failure(self, rate_limited: bool = False) -> None:
        """Mark endpoint as failed."""
        import random
        from datetime import timedelta

        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:
            self.is_healthy = False

        if rate_limited:
            # Exponential backoff with jitter
            delay = min(60, 2**self.consecutive_errors) + random.uniform(0, 1)
            self.rate_limited_until = datetime.now(UTC) + timedelta(seconds=delay)

    def record_tokens(self, tokens: int) -> None:
        """Record token usage for rate limiting."""
        now = datetime.now(UTC)
        if self.minute_started_at is None or (now - self.minute_started_at).total_seconds() >= 60:
            self.minute_started_at = now
            self.tokens_used_this_minute = 0
        self.tokens_used_this_minute += tokens

    def can_handle_request(self, estimated_tokens: int, tokens_per_minute: int) -> bool:
        """Check if endpoint can handle request within rate limit."""
        if self.rate_limited_until and datetime.now(UTC) < self.rate_limited_until:
            return False
        return (self.tokens_used_this_minute + estimated_tokens) <= tokens_per_minute


class LLMRequest(BaseModel):
    """Request to LLM service."""

    messages: list[dict[str, str]]
    tier: ModelTier
    temperature: float | None = None
    max_tokens: int | None = None
    structured_output: type[BaseModel] | None = None


@dataclass
class LLMResponse:
    """Response from LLM service."""

    content: str
    usage: dict[str, int]
    endpoint_id: str
    duration_ms: float
    structured: Any | None = None


@dataclass
class ToolCall:
    """A tool call from an LLM response (OpenAI format)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallChunk:
    """Accumulated streaming data for a single tool call."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_json: str = ""

    def is_complete(self) -> bool:
        """Check if we have all required fields."""
        return bool(self.id and self.name and self.arguments_json)


@dataclass
class StreamWithToolsChunk:
    """A chunk from streaming with tools - either content or tool calls."""

    # Content text (may be empty)
    content: str = ""
    # Tool calls detected in this chunk
    tool_calls: list[ToolCall] | None = None
    # Is this the final chunk (LLM finished)?
    is_done: bool = False
