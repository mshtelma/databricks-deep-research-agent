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
    BULK_ANALYSIS = "bulk_analysis"  # Large-context analysis (Gemini), NLI, extraction
    FAST = "fast"  # Non-structured quick tasks (GPT 5.2)


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
    # Claude models support prompt caching via cache_control parameter
    supports_prompt_caching: bool = False


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


# Minimum time (seconds) before retrying an unhealthy endpoint
# This ensures provider rate limits have time to clear
MIN_RECOVERY_SECONDS = 60


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
            # Exponential backoff with jitter, but enforce minimum floor
            # to let provider rate limits clear before retry
            exponential_delay = min(180, 2**self.consecutive_errors)
            delay = max(MIN_RECOVERY_SECONDS, exponential_delay) + random.uniform(0, 1)
            self.rate_limited_until = datetime.now(UTC) + timedelta(seconds=delay)

    def reset_if_recovered(self) -> bool:
        """Reset health if rate limit has expired.

        This enables time-based recovery: when an endpoint was marked unhealthy
        due to rate limits, it can be given another chance after the rate limit
        timeout expires.

        Returns:
            True if the endpoint was reset to healthy, False otherwise.
        """
        if not self.is_healthy and self.rate_limited_until:
            if datetime.now(UTC) >= self.rate_limited_until:
                # Give the endpoint another chance
                self.is_healthy = True
                self.consecutive_errors = 0
                self.rate_limited_until = None
                return True
        return False

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

    def get_budget_reset_wait(self) -> float:
        """Seconds until token budget window resets.

        The budget window is 60 seconds from minute_started_at (see record_tokens()).
        After 60s, tokens_used_this_minute resets to 0.

        Returns:
            0.0 if no tracking started yet or window has already reset.
            Otherwise, seconds remaining until the 60s window expires.
        """
        if self.minute_started_at is None:
            return 0.0
        now = datetime.now(UTC)
        elapsed = (now - self.minute_started_at).total_seconds()
        if elapsed >= 60.0:
            return 0.0
        return 60.0 - elapsed


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
