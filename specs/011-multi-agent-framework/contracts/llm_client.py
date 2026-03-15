"""
Contract: LLM Client.

The framework wraps openai.AsyncOpenAI directly. No Protocol abstraction.
Databricks standardizes on OpenAI-compatible endpoints.

The app provides the AsyncOpenAI instance + model mapping via llm_adapter.py.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from openai import AsyncOpenAI


class ModelTier(str, Enum):
    """Model tier for routing to appropriate endpoints."""

    simple = "simple"
    analytical = "analytical"
    complex = "complex"


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    # usage keys: prompt_tokens, completion_tokens, total_tokens
    model: str = ""
    finish_reason: str = "stop"
    structured: Any | None = None  # Parsed structured output (Pydantic model, dict, etc.)


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    function_name: str
    arguments: str  # JSON string


@dataclass
class EndpointHealth:
    """Per-endpoint runtime health state.

    Tracks consecutive errors, rate limiting windows, and token usage
    for intelligent endpoint selection and failover.
    """

    is_healthy: bool = True
    consecutive_errors: int = 0
    rate_limited_until: float = 0.0
    tokens_used_this_minute: int = 0
    minute_started_at: float = 0.0

    def mark_success(self) -> None:
        """Reset error counters on successful request."""
        ...

    def mark_failure(self, rate_limited: bool = False) -> None:
        """Increment error counter, optionally set rate limit window."""
        ...

    def can_handle_request(self, estimated_tokens: int, tpm_limit: int) -> bool:
        """Check if endpoint can handle request within TPM budget."""
        ...


@dataclass(frozen=True)
class ModelTierConfig:
    """Rich model tier config with multiple endpoints and fallback.

    Replaces simple str mapping when rate limiting / fallback is needed.
    Backward compatible — str values in model_mapping are auto-wrapped.
    """

    endpoints: list[str]  # Priority-ordered endpoint names
    fallback_on_429: bool = True
    rotation_strategy: Literal["PRIORITY", "ROUND_ROBIN"] = "PRIORITY"
    tokens_per_minute: int = 0  # 0 = unlimited


class FrameworkLLMClient:
    """Thin wrapper around AsyncOpenAI for framework use.

    Maps ModelTier to concrete model names and provides structured output support.
    This is NOT a Protocol — the framework depends on openai directly.

    Usage (standalone):
        client = FrameworkLLMClient(
            openai_client=AsyncOpenAI(api_key="sk-..."),
            model_mapping={"simple": "gpt-4o-mini", "analytical": "gpt-4o", "complex": "o1"},
        )
        response = await client.complete(messages, tier=ModelTier.analytical)

    Usage (Deep Research app via adapter):
        openai_client, model_mapping = llm_adapter.adapt(app_llm_client)
        client = FrameworkLLMClient(openai_client, model_mapping)
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_mapping: dict[str, str | ModelTierConfig],
        *,
        embedding_model: str | None = None,
    ) -> None:
        self._client = openai_client
        self._models = model_mapping
        self._embedding_model = embedding_model
        self._endpoint_health: dict[str, EndpointHealth] = {}

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Access the underlying AsyncOpenAI client."""
        return self._client

    def resolve_model(self, tier: str | ModelTier) -> str:
        """Resolve a model tier to a concrete model name."""
        tier_str = tier.value if isinstance(tier, ModelTier) else tier
        model = self._models.get(tier_str)
        if model is None:
            raise ValueError(f"Unknown model tier: {tier_str}. Available: {list(self._models.keys())}")
        return model

    def _select_endpoint(self, tier: str) -> str:
        """Select best endpoint for tier based on health + rotation strategy."""
        ...

    def _find_fallback(self, tier: str, failed_endpoint: str) -> str | None:
        """Find a fallback endpoint after 429/failure on the primary."""
        ...

    @property
    def supports_embeddings(self) -> bool:
        """Whether an embedding model is configured."""
        return self._embedding_model is not None

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Batch embed texts via OpenAI embeddings.create().

        Args:
            texts: List of strings to embed.
            model: Override embedding model. If None, uses configured embedding_model.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If no embedding model is configured and none provided.
        """
        ...  # pragma: no cover

    async def embed_single(self, text: str) -> list[float]:
        """Convenience for embedding a single text.

        Args:
            text: String to embed.

        Returns:
            Embedding vector.
        """
        ...  # pragma: no cover

    async def _retry_with_backoff(
        self, func: Any, *, max_retries: int = 3
    ) -> Any:
        """Retry with exponential backoff and jitter on transient failures."""
        ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tier: str | ModelTier = ModelTier.analytical,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: Any | None = None,
        structured_output: type | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response.

        When a ModelTierConfig is provided for the tier, the client will:
        1. Select the best healthy endpoint via _select_endpoint()
        2. Track token usage against TPM limits
        3. On 429 errors, automatically try fallback endpoints if fallback_on_429=True
        4. Use exponential backoff with jitter for transient failures
        5. Mark endpoints healthy/unhealthy based on success/failure

        Args:
            messages: Conversation messages (OpenAI format).
            tier: Model tier hint — resolved to concrete model via model_mapping.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            tools: Tool definitions for function calling (OpenAI format).
            response_format: OpenAI response_format parameter.
            structured_output: Pydantic model class for structured output parsing.

        Returns:
            LLMResponse with content, optional tool calls, usage stats, and
            optional structured output.
        """
        ...  # pragma: no cover

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tier: str | ModelTier = ModelTier.analytical,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str | ToolCall, None]:
        """Stream response tokens and tool calls.

        Retry and fallback behavior mirrors complete() — on 429, the client
        automatically tries fallback endpoints before raising.

        Yields:
            str chunks for content, ToolCall objects for tool calls.
        """
        ...  # pragma: no cover
