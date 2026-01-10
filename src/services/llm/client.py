"""Unified LLM client using OpenAI SDK with Databricks endpoints."""

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import mlflow
import openai
from mlflow.entities import SpanType
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ValidationError as PydanticValidationError

from src.core.config import get_settings
from src.core.exceptions import LLMError, RateLimitError
from src.core.logging_utils import (
    get_logger,
    log_llm_error,
    log_llm_fallback,
    log_llm_prompt,
    log_llm_request,
    log_llm_response,
)
from src.services.llm.auth import LLMCredentialProvider
from src.services.llm.config import ModelConfig
from src.services.llm.types import (
    EndpointHealth,
    LLMResponse,
    ModelEndpoint,
    ModelRole,
    ModelTier,
    StreamWithToolsChunk,
    ToolCall,
    ToolCallChunk,
)

logger = get_logger(__name__)


def _normalize_content(raw_content: str | list[Any] | None) -> str:
    """Normalize LLM response content to a string.

    Some models (e.g., Databricks Gemini) return content as a list of
    content parts instead of a plain string. This function handles both
    formats uniformly.

    Args:
        raw_content: The raw content from message.content - can be:
            - str: Plain text response
            - list: List of content parts [{"type": "text", "text": "..."}]
            - None: No content

    Returns:
        Normalized string content.
    """
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        # Handle list of content parts (Gemini-style response)
        parts: list[str] = []
        for part in raw_content:
            if isinstance(part, dict):
                # ContentPart format: {"type": "text", "text": "..."}
                text = part.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
            else:
                # Fallback: convert to string
                parts.append(str(part))
        return " ".join(parts)
    # Fallback for unexpected types
    return str(raw_content)


def normalize_messages_for_logging(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize messages for MLflow span logging.

    Converts message content from list format (used by cache_control or
    Gemini responses) to plain strings to avoid Pydantic serialization
    warnings when MLflow logs the messages.

    Args:
        messages: List of message dicts, potentially with list content.

    Returns:
        List of message dicts with all content normalized to strings.
    """
    normalized: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        normalized_msg: dict[str, str] = {
            "role": str(msg.get("role", "user")),
            "content": _normalize_content(content) if not isinstance(content, str) else content,
        }
        # Preserve tool_call_id if present (for tool response messages)
        if "tool_call_id" in msg:
            normalized_msg["tool_call_id"] = str(msg["tool_call_id"])
        if "name" in msg:
            normalized_msg["name"] = str(msg["name"])
        normalized.append(normalized_msg)
    return normalized


class LLMClient:
    """Unified LLM client with model routing and health tracking."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize LLM client.

        Args:
            config: Model configuration. If None, creates new instance.
        """
        settings = get_settings()
        self._config = config or ModelConfig()
        self._health: dict[str, EndpointHealth] = {}

        # Auth mode tracking
        self._credential_provider: LLMCredentialProvider | None = None
        self._current_token: str | None = None

        # Auth retry tracking (prevents infinite loops on persistent auth failures)
        self._auth_retry_count: int = 0
        self._refresh_lock = asyncio.Lock()

        # Get token - either from env or from WorkspaceClient (with refresh support)
        token = settings.databricks_token
        self._base_url = f"{settings.databricks_host}/serving-endpoints"

        if not token and settings.databricks_config_profile:
            # Profile-based OAuth auth with refresh support
            self._credential_provider = LLMCredentialProvider(
                profile=settings.databricks_config_profile
            )
            credential = self._credential_provider.get_credential()
            token = credential.token
            self._base_url = self._credential_provider.get_base_url()

        if not token:
            raise ValueError("No Databricks token available")

        self._current_token = token

        # Initialize OpenAI client for Databricks
        self._client = AsyncOpenAI(
            api_key=token,
            base_url=self._base_url,
        )

    def _ensure_fresh_client(self) -> None:
        """Ensure the OpenAI client has a fresh OAuth token.

        For profile-based OAuth auth, checks if token is expired and
        refreshes if needed. For direct token auth, this is a no-op.
        """
        if self._credential_provider is None:
            # Direct token auth - no refresh needed
            return

        credential = self._credential_provider.get_credential()

        if credential.token != self._current_token:
            # Token was refreshed - recreate client
            logger.info("LLM_TOKEN_REFRESHED", provider="oauth")
            self._current_token = credential.token
            self._client = AsyncOpenAI(
                api_key=credential.token,
                base_url=self._base_url,
            )

    async def _force_refresh_token(self) -> bool:
        """Force refresh OAuth token after authentication error.

        Uses async lock to prevent race conditions when multiple
        concurrent requests encounter auth errors simultaneously.

        Returns:
            True if token was refreshed and retry should proceed, False otherwise.
        """
        if self._credential_provider is None:
            return False

        async with self._refresh_lock:
            # Prevent infinite auth retry loop (max 2 retries)
            self._auth_retry_count += 1
            if self._auth_retry_count > 2:
                logger.error("LLM_AUTH_RETRY_EXHAUSTED", max_retries=2)
                self._auth_retry_count = 0
                return False

            # Check if another coroutine already refreshed while we waited
            credential = self._credential_provider.get_credential()
            if credential.token != self._current_token:
                # Token was already refreshed by another coroutine
                logger.info("LLM_TOKEN_ALREADY_REFRESHED", by="concurrent_request")
                self._current_token = credential.token
                self._client = AsyncOpenAI(
                    api_key=credential.token,
                    base_url=self._base_url,
                )
                return True

            # Force refresh the token
            logger.warning(
                "LLM_TOKEN_FORCE_REFRESH",
                reason="auth_error",
                attempt=self._auth_retry_count,
            )
            credential = self._credential_provider.get_credential(force_refresh=True)
            self._current_token = credential.token
            self._client = AsyncOpenAI(
                api_key=credential.token,
                base_url=self._base_url,
            )
            return True

    def _reset_auth_retry_count(self) -> None:
        """Reset auth retry count after successful request."""
        self._auth_retry_count = 0

    def _get_health(self, endpoint_id: str) -> EndpointHealth:
        """Get or create health state for endpoint."""
        if endpoint_id not in self._health:
            self._health[endpoint_id] = EndpointHealth(endpoint_id=endpoint_id)
        return self._health[endpoint_id]

    def _select_endpoint(
        self,
        role: ModelRole,
        estimated_tokens: int,
    ) -> tuple[ModelEndpoint, EndpointHealth]:
        """Select best endpoint for request.

        Includes time-based recovery: endpoints marked unhealthy due to rate
        limits can be retried after their rate_limited_until timestamp expires.
        """
        endpoints = self._config.get_endpoints_for_role(role.name)

        for endpoint in endpoints:
            health = self._get_health(endpoint.id)

            # Check if endpoint should be considered (includes time-based recovery)
            if not health.is_healthy:
                # Try time-based recovery for rate-limited endpoints
                if health.reset_if_recovered():
                    logger.info(
                        "ENDPOINT_RECOVERED",
                        endpoint=endpoint.id,
                        reason="rate_limit_expired",
                    )
                else:
                    continue  # Still unhealthy, skip

            if health.can_handle_request(estimated_tokens, endpoint.tokens_per_minute):
                return endpoint, health

        # No healthy endpoint available
        raise RateLimitError(retry_after=30)

    def _merge_config(
        self,
        role: ModelRole,
        endpoint: ModelEndpoint,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Merge role and endpoint configuration.

        Note: Some models (e.g., GPT-5) don't support temperature parameter.
        The supports_temperature flag controls whether it's included.
        """
        config: dict[str, Any] = {
            "max_tokens": max_tokens or endpoint.max_tokens or role.max_tokens,
        }

        # Only include temperature if the endpoint supports it
        if endpoint.supports_temperature:
            config["temperature"] = temperature or endpoint.temperature or role.temperature

        return config

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        endpoint: ModelEndpoint,
    ) -> list[dict[str, Any]]:
        """Transform messages to include cache_control for supported endpoints.

        This is transparent to higher layers - agents send plain messages,
        this method transforms them internally when caching is enabled.

        Converts system message from string format:
            {"role": "system", "content": "Your prompt..."}

        To content-array format with cache_control:
            {"role": "system", "content": [
                {"type": "text", "text": "Your prompt...", "cache_control": {"type": "ephemeral"}}
            ]}

        Args:
            messages: Original messages from agent.
            endpoint: Selected endpoint (checked for caching support).

        Returns:
            Messages with cache_control applied if applicable, otherwise unchanged.
        """
        # Check if endpoint supports caching
        if not endpoint.supports_prompt_caching:
            return messages

        # Get prompt caching config from app config
        caching_config = self._config.app_config.prompt_caching
        if not caching_config.enabled:
            return messages

        transformed: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and caching_config.cache_system_prompt:
                content = msg.get("content", "")
                # Handle content that's already in array format
                if isinstance(content, list):
                    transformed.append(msg)
                    continue

                # Only cache if above minimum threshold (~4 chars per token)
                estimated_tokens = len(content) // 4
                if estimated_tokens >= caching_config.min_tokens_threshold:
                    transformed.append({
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": caching_config.cache_type}
                        }]
                    })
                    if caching_config.log_cache_usage:
                        logger.debug(
                            "CACHE_CONTROL_APPLIED",
                            role="system",
                            estimated_tokens=estimated_tokens,
                            endpoint=endpoint.id,
                        )
                else:
                    transformed.append(msg)
            else:
                transformed.append(msg)

        return transformed

    def _get_earliest_endpoint_available(self, role: ModelRole) -> float:
        """Get seconds until the earliest endpoint becomes available.

        Returns 0 if an endpoint is already available (including endpoints
        that can be recovered via time-based recovery), otherwise the
        shortest wait time until any endpoint's rate limit expires.

        Args:
            role: The model role to check endpoints for.

        Returns:
            Seconds to wait (0 if an endpoint is available now).
        """
        now = datetime.now(UTC)
        min_wait = float("inf")

        for endpoint_id in role.endpoints:
            health = self._get_health(endpoint_id)

            # Check for healthy endpoints available now
            if health.is_healthy:
                if not health.rate_limited_until or health.rate_limited_until <= now:
                    return 0.0  # Available now

            # Check for unhealthy endpoints that can be recovered
            # (rate limit expired, so reset_if_recovered() will succeed)
            elif health.rate_limited_until and health.rate_limited_until <= now:
                return 0.0  # Will recover on next _select_endpoint() call

            # Check when rate limit expires (for both healthy and unhealthy)
            if health.rate_limited_until and health.rate_limited_until > now:
                wait = (health.rate_limited_until - now).total_seconds()
                min_wait = min(min_wait, wait)

        return min_wait if min_wait != float("inf") else 0.0

    async def complete(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier,
        temperature: float | None = None,
        max_tokens: int | None = None,
        structured_output: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Complete a chat request with automatic retry on rate limits.

        Args:
            messages: Chat messages.
            tier: Model tier (simple, analytical, complex).
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            structured_output: Optional Pydantic model for structured output.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            RateLimitError: If all retries are exhausted.
            LLMError: For other LLM errors.
        """
        from src.core.app_config import get_app_config

        rate_config = get_app_config().rate_limiting
        role = self._config.get_role(tier.value)

        last_error: Exception | None = None

        for attempt in range(rate_config.max_retries + 1):
            try:
                return await self._complete_impl(
                    messages, tier, role, temperature, max_tokens, structured_output
                )
            except RateLimitError as e:
                last_error = e

                if attempt >= rate_config.max_retries:
                    # Max retries exhausted
                    logger.warning(
                        "LLM_RATE_LIMIT_EXHAUSTED",
                        tier=tier.value,
                        attempts=attempt + 1,
                    )
                    raise

                # Calculate wait time using configured strategy
                endpoint_wait = self._get_earliest_endpoint_available(role)
                backoff_wait = rate_config.calculate_delay(attempt)
                delay = max(endpoint_wait, backoff_wait)

                if rate_config.jitter:
                    delay += random.uniform(0, 1)

                logger.warning(
                    "LLM_RATE_LIMIT_RETRY",
                    tier=tier.value,
                    attempt=attempt + 1,
                    max_retries=rate_config.max_retries,
                    delay_seconds=round(delay, 2),
                    endpoint_wait=round(endpoint_wait, 2),
                )

                await asyncio.sleep(delay)

        # Should not reach here, but satisfy type checker
        raise last_error or RateLimitError(retry_after=30)

    async def _complete_impl(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier,
        role: ModelRole,
        temperature: float | None = None,
        max_tokens: int | None = None,
        structured_output: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Internal implementation of complete (called by retry wrapper).

        Args:
            messages: Chat messages.
            tier: Model tier (simple, analytical, complex).
            role: Model role configuration.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            structured_output: Optional Pydantic model for structured output.

        Returns:
            LLMResponse with content and metadata.
        """
        # Ensure fresh OAuth token before request
        self._ensure_fresh_client()

        # Estimate tokens (rough: ~4 chars per token)
        estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
        estimated_tokens = estimated_input + (max_tokens or role.max_tokens)

        # Select endpoint
        endpoint, health = self._select_endpoint(role, estimated_tokens)
        config = self._merge_config(role, endpoint, temperature, max_tokens)

        # Log the request
        log_llm_request(
            logger,
            endpoint=endpoint.endpoint_identifier,
            tier=tier.value,
            messages=messages,
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            estimated_tokens=estimated_tokens,
        )

        # Log prompts at DEBUG level (+ MLflow trace)
        system_msg = next(
            (m.get("content") for m in messages if m.get("role") == "system"), None
        )
        user_msg = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        log_llm_prompt(logger, system_prompt=system_msg, user_prompt=user_msg or "")

        # Wrap LLM call in MLflow span for tracing
        with mlflow.start_span(
            name=f"llm_{tier.value}", span_type=SpanType.CHAT_MODEL
        ) as span:
            span.set_attributes({
                "llm.tier": tier.value,
                "llm.endpoint": endpoint.endpoint_identifier,
                "llm.message_count": len(messages),
                "llm.max_tokens": config.get("max_tokens"),
                "llm.temperature": config.get("temperature"),
            })
            # Capture input messages for tracing visibility
            # Normalize to avoid Pydantic serialization warnings with list content
            span.set_inputs({"messages": normalize_messages_for_logging(messages)})

            start_time = time.perf_counter()

            try:
                # Apply cache control for supported endpoints (transparent to callers)
                cached_messages = self._apply_cache_control(messages, endpoint)

                # Build request
                request_kwargs: dict[str, Any] = {
                    "model": endpoint.endpoint_identifier,
                    "messages": cached_messages,
                    **config,
                }

                # Add structured output if requested and supported
                if structured_output and endpoint.supports_structured_output:
                    request_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": structured_output.__name__,
                            "schema": structured_output.model_json_schema(),
                        },
                    }
                    logger.debug(
                        "Using structured output",
                        schema=structured_output.__name__,
                    )

                response = await self._client.chat.completions.create(**request_kwargs)

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Extract content with bounds check
                if not response.choices:
                    raise LLMError(
                        "Empty response from model (no choices)", endpoint=endpoint.id
                    )
                # Normalize content: handles both string and list formats (Gemini)
                content = _normalize_content(response.choices[0].message.content)

                # Parse structured output if requested
                structured = None
                if structured_output:
                    try:
                        structured = structured_output.model_validate_json(content)
                    except (json.JSONDecodeError, ValueError, PydanticValidationError) as e:
                        logger.warning(
                            "Structured output parse failed, attempting repair",
                            error=str(e)[:100],
                            schema=structured_output.__name__,
                        )
                        # Try with json-repair
                        from json_repair import repair_json

                        repaired = repair_json(content)
                        structured = structured_output.model_validate(
                            json.loads(repaired)
                        )
                        logger.debug("JSON repair successful")

                # Update health
                usage = {
                    "prompt_tokens": (
                        response.usage.prompt_tokens if response.usage else 0
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "total_tokens": (
                        response.usage.total_tokens if response.usage else 0
                    ),
                }
                health.mark_success()
                health.record_tokens(usage["total_tokens"])

                # Add response metrics to span
                span.set_attributes({
                    "llm.input_tokens": usage["prompt_tokens"],
                    "llm.output_tokens": usage["completion_tokens"],
                    "llm.total_tokens": usage["total_tokens"],
                    "llm.latency_ms": duration_ms,
                })

                # Log the response
                log_llm_response(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    duration_ms=duration_ms,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    content=content,
                    structured_type=(
                        structured_output.__name__ if structured_output else None
                    ),
                )

                # Capture output for tracing visibility
                span.set_outputs({"content": content, "usage": usage})

                # Reset auth retry count on successful request
                self._reset_auth_retry_count()

                return LLMResponse(
                    content=content,
                    usage=usage,
                    endpoint_id=endpoint.id,
                    duration_ms=duration_ms,
                    structured=structured,
                )

            except RateLimitError:
                # Re-raise our own RateLimitError to be handled by retry wrapper
                raise

            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # OAuth token invalid/expired - force refresh and retry once
                # Note: Databricks returns 403 PermissionDeniedError for expired tokens
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=False)

                span.set_attributes({
                    "llm.error": True,
                    "llm.error_type": type(e).__name__,
                    "llm.auth_retry": True,
                })

                logger.warning(
                    "LLM_AUTH_ERROR",
                    endpoint=endpoint.endpoint_identifier,
                    status_code=getattr(e, "status_code", None),
                    will_retry=self._credential_provider is not None,
                )

                if await self._force_refresh_token():
                    # Retry with refreshed token
                    return await self._complete_impl(
                        messages,
                        tier,
                        role,
                        temperature,
                        max_tokens,
                        structured_output,
                    )

                raise LLMError(
                    f"Authentication failed: {e}", endpoint=endpoint.id
                ) from e

            except openai.RateLimitError as e:
                # Proper rate limit detection via exception type
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=True)

                span.set_attributes({
                    "llm.error": True,
                    "llm.error_type": "RateLimitError",
                    "llm.is_rate_limit": True,
                })

                # Check if we can fallback to another endpoint
                can_fallback = False
                fallback_endpoint_id = None
                if role.fallback_on_429:
                    for fallback_endpoint in self._config.get_endpoints_for_role(
                        role.name
                    ):
                        if fallback_endpoint.id == endpoint.id:
                            continue
                        fallback_health = self._get_health(fallback_endpoint.id)
                        if fallback_health.is_healthy:
                            can_fallback = True
                            fallback_endpoint_id = fallback_endpoint.id
                            break

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=True,
                    will_fallback=can_fallback,
                )

                if can_fallback and fallback_endpoint_id:
                    log_llm_fallback(
                        logger,
                        from_endpoint=endpoint.endpoint_identifier,
                        to_endpoint=fallback_endpoint_id,
                        reason="rate_limit",
                    )
                    return await self._complete_impl(
                        messages,
                        tier,
                        role,
                        temperature,
                        max_tokens,
                        structured_output,
                    )

                raise RateLimitError(retry_after=30) from e

            except openai.APIStatusError as e:
                # Other API errors with status codes
                duration_ms = (time.perf_counter() - start_time) * 1000
                is_rate_limit = e.status_code == 429
                health.mark_failure(rate_limited=is_rate_limit)

                span.set_attributes({
                    "llm.error": True,
                    "llm.error_type": type(e).__name__,
                    "llm.status_code": e.status_code,
                    "llm.is_rate_limit": is_rate_limit,
                })

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=is_rate_limit,
                    will_fallback=False,
                )

                if is_rate_limit:
                    raise RateLimitError(retry_after=30) from e
                raise LLMError(str(e), endpoint=endpoint.id) from e

            except openai.APIConnectionError as e:
                # Connection errors (network, timeout)
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=False)

                span.set_attributes({
                    "llm.error": True,
                    "llm.error_type": "APIConnectionError",
                })

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(
                    f"Connection error: {e}", endpoint=endpoint.id
                ) from e

            except Exception as e:
                # Fallback for unexpected errors
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=False)

                span.set_attributes({
                    "llm.error": True,
                    "llm.error_type": type(e).__name__,
                })

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(str(e), endpoint=endpoint.id) from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion with automatic retry on rate limits.

        Args:
            messages: Chat messages.
            tier: Model tier.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Yields:
            Content chunks as they arrive.

        Raises:
            RateLimitError: If all retries are exhausted.
            LLMError: For other LLM errors.
        """
        from src.core.app_config import get_app_config

        rate_config = get_app_config().rate_limiting
        role = self._config.get_role(tier.value)

        last_error: Exception | None = None

        for attempt in range(rate_config.max_retries + 1):
            try:
                async for chunk in self._stream_impl(
                    messages, tier, role, temperature, max_tokens
                ):
                    yield chunk
                return  # Success - exit retry loop
            except RateLimitError as e:
                last_error = e

                if attempt >= rate_config.max_retries:
                    logger.warning(
                        "LLM_RATE_LIMIT_EXHAUSTED",
                        tier=tier.value,
                        attempts=attempt + 1,
                        streaming=True,
                    )
                    raise

                # Calculate wait time using configured strategy
                endpoint_wait = self._get_earliest_endpoint_available(role)
                backoff_wait = rate_config.calculate_delay(attempt)
                delay = max(endpoint_wait, backoff_wait)

                if rate_config.jitter:
                    delay += random.uniform(0, 1)

                logger.warning(
                    "LLM_RATE_LIMIT_RETRY",
                    tier=tier.value,
                    attempt=attempt + 1,
                    max_retries=rate_config.max_retries,
                    delay_seconds=round(delay, 2),
                    streaming=True,
                )

                await asyncio.sleep(delay)

        raise last_error or RateLimitError(retry_after=30)

    async def _stream_impl(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier,
        role: ModelRole,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Internal implementation of stream (called by retry wrapper).

        Args:
            messages: Chat messages.
            tier: Model tier.
            role: Model role configuration.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Yields:
            Content chunks as they arrive.
        """
        from src.services.llm.tracing import traced_stream

        # Ensure fresh OAuth token before request
        self._ensure_fresh_client()

        estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        endpoint, health = self._select_endpoint(role, estimated_tokens + 4000)
        config = self._merge_config(role, endpoint, temperature, max_tokens)

        # Apply cache control for supported endpoints (transparent to callers)
        cached_messages = self._apply_cache_control(messages, endpoint)

        # Log stream start
        logger.info(
            "LLM_STREAM_START",
            endpoint=endpoint.endpoint_identifier,
            tier=tier.value,
            messages=len(messages),
            est_tokens=estimated_tokens,
        )

        # Log prompts at DEBUG level (+ MLflow trace)
        system_msg = next(
            (m.get("content") for m in messages if m.get("role") == "system"), None
        )
        user_msg = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        log_llm_prompt(logger, system_prompt=system_msg, user_prompt=user_msg or "")

        # Inner generator for actual streaming
        async def _inner_stream() -> AsyncIterator[str]:
            start_time = time.perf_counter()
            chunk_count = 0
            total_content_len = 0

            try:
                stream = cast(
                    AsyncStream[ChatCompletionChunk],
                    await self._client.chat.completions.create(
                        model=endpoint.endpoint_identifier,
                        messages=cast(Any, cached_messages),
                        stream=True,
                        **config,
                    ),
                )

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        # Normalize content: handles both string and list formats (Gemini)
                        content = _normalize_content(chunk.choices[0].delta.content)
                        if content:
                            chunk_count += 1
                            total_content_len += len(content)
                            yield content

                health.mark_success()
                self._reset_auth_retry_count()

                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "LLM_STREAM_END",
                    endpoint=endpoint.endpoint_identifier,
                    duration_ms=round(duration_ms, 1),
                    chunks=chunk_count,
                    content_len=total_content_len,
                )

            except RateLimitError:
                # Re-raise our own RateLimitError to be handled by retry wrapper
                raise

            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # OAuth token invalid/expired - force refresh and retry
                # Note: Databricks returns 403 PermissionDeniedError for expired tokens
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=False)

                logger.warning(
                    "LLM_STREAM_AUTH_ERROR",
                    endpoint=endpoint.endpoint_identifier,
                    status_code=getattr(e, "status_code", None),
                    will_retry=self._credential_provider is not None,
                )

                if await self._force_refresh_token():
                    # Retry by re-yielding from a new stream
                    async for retry_chunk in self._stream_impl(
                        messages, tier, role, temperature, max_tokens
                    ):
                        yield retry_chunk
                    return

                raise LLMError(
                    f"Authentication failed: {e}", endpoint=endpoint.id
                ) from e

            except openai.RateLimitError as e:
                # Proper rate limit detection via exception type
                health.mark_failure(rate_limited=True)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=True,
                    will_fallback=False,
                )

                raise RateLimitError(retry_after=30) from e

            except openai.APIStatusError as e:
                # Other API errors with status codes
                is_rate_limit = e.status_code == 429
                health.mark_failure(rate_limited=is_rate_limit)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=is_rate_limit,
                    will_fallback=False,
                )

                if is_rate_limit:
                    raise RateLimitError(retry_after=30) from e
                raise LLMError(str(e), endpoint=endpoint.id) from e

            except openai.APIConnectionError as e:
                # Connection errors (network, timeout)
                health.mark_failure(rate_limited=False)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(
                    f"Connection error: {e}", endpoint=endpoint.id
                ) from e

            except Exception as e:
                # Fallback for unexpected errors
                health.mark_failure(rate_limited=False)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(str(e), endpoint=endpoint.id) from e

        # Wrap with MLflow tracing
        traced = traced_stream(
            inner=_inner_stream(),
            tier=tier,
            endpoint=endpoint.endpoint_identifier,
            message_count=len(messages),
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature"),
            messages=messages,
        )

        # Yield from traced iterator
        async for chunk in traced:
            yield chunk

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tier: ModelTier,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamWithToolsChunk]:
        """Stream a chat completion with tool calling support.

        This method supports the OpenAI function calling format for ReAct-style
        agent loops. The LLM can decide to call tools or output content.

        Args:
            messages: Chat messages (may include tool role for results).
            tools: OpenAI-format tool definitions.
            tier: Model tier.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Yields:
            StreamWithToolsChunk with content and/or tool calls.

        Raises:
            RateLimitError: If all retries are exhausted.
            LLMError: For other LLM errors.
        """
        from src.core.app_config import get_app_config

        rate_config = get_app_config().rate_limiting
        role = self._config.get_role(tier.value)

        last_error: Exception | None = None

        for attempt in range(rate_config.max_retries + 1):
            try:
                async for chunk in self._stream_with_tools_impl(
                    messages, tools, tier, role, temperature, max_tokens
                ):
                    yield chunk
                return  # Success
            except RateLimitError as e:
                last_error = e

                if attempt >= rate_config.max_retries:
                    logger.warning(
                        "LLM_RATE_LIMIT_EXHAUSTED",
                        tier=tier.value,
                        attempts=attempt + 1,
                        streaming=True,
                        with_tools=True,
                    )
                    raise

                endpoint_wait = self._get_earliest_endpoint_available(role)
                backoff_wait = rate_config.calculate_delay(attempt)
                delay = max(endpoint_wait, backoff_wait)

                if rate_config.jitter:
                    delay += random.uniform(0, 1)

                logger.warning(
                    "LLM_RATE_LIMIT_RETRY",
                    tier=tier.value,
                    attempt=attempt + 1,
                    max_retries=rate_config.max_retries,
                    delay_seconds=round(delay, 2),
                    streaming=True,
                    with_tools=True,
                )

                await asyncio.sleep(delay)

        raise last_error or RateLimitError(retry_after=30)

    async def _stream_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tier: ModelTier,
        role: ModelRole,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamWithToolsChunk]:
        """Internal implementation of stream_with_tools.

        Handles streaming responses with tool calls, accumulating tool call
        arguments across chunks.

        Args:
            messages: Chat messages.
            tools: OpenAI-format tool definitions.
            tier: Model tier.
            role: Model role configuration.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Yields:
            StreamWithToolsChunk with content and/or tool calls.
        """
        from src.services.llm.tracing import traced_stream_with_tools

        # Ensure fresh OAuth token before request
        self._ensure_fresh_client()

        estimated_tokens = sum(
            len(str(m.get("content", ""))) for m in messages
        ) // 4
        endpoint, health = self._select_endpoint(role, estimated_tokens + 4000)
        config = self._merge_config(role, endpoint, temperature, max_tokens)

        # Apply cache control for supported endpoints (transparent to callers)
        cached_messages = self._apply_cache_control(messages, endpoint)

        logger.info(
            "LLM_STREAM_WITH_TOOLS_START",
            endpoint=endpoint.endpoint_identifier,
            tier=tier.value,
            messages=len(messages),
            tools_count=len(tools),
            est_tokens=estimated_tokens,
        )

        # Inner generator for actual streaming with tools
        async def _inner_stream() -> AsyncIterator[StreamWithToolsChunk]:
            start_time = time.perf_counter()

            # Track tool call accumulation across chunks
            tool_call_chunks: dict[int, ToolCallChunk] = {}

            try:
                stream = cast(
                    AsyncStream[ChatCompletionChunk],
                    await self._client.chat.completions.create(
                        model=endpoint.endpoint_identifier,
                        messages=cast(Any, cached_messages),
                        tools=cast(Any, tools if tools else None),
                        stream=True,
                        **config,
                    ),
                )

                accumulated_content = ""

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    # Handle content
                    # Normalize content: handles both string and list formats (Gemini)
                    content = _normalize_content(delta.content) if delta.content else ""
                    if content:
                        accumulated_content += content

                    # Handle tool calls (accumulated across chunks)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index

                            # Initialize or update tool call accumulator
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = ToolCallChunk(index=idx)

                            tcc = tool_call_chunks[idx]

                            if tc.id:
                                tcc.id = tc.id
                            if tc.function and tc.function.name:
                                tcc.name = tc.function.name
                            if tc.function and tc.function.arguments:
                                tcc.arguments_json += tc.function.arguments

                    # Yield content chunks as they come
                    if content:
                        yield StreamWithToolsChunk(content=content)

                    # Check if stream is done
                    if finish_reason:
                        # Finalize tool calls
                        completed_tool_calls = []
                        for tcc in tool_call_chunks.values():
                            if tcc.id and tcc.name:
                                try:
                                    args = (
                                        json.loads(tcc.arguments_json)
                                        if tcc.arguments_json
                                        else {}
                                    )
                                except json.JSONDecodeError:
                                    # Try to repair JSON
                                    from json_repair import repair_json

                                    try:
                                        repaired = repair_json(tcc.arguments_json)
                                        args = json.loads(repaired)
                                    except Exception:
                                        args = {}
                                        logger.warning(
                                            "TOOL_CALL_ARGS_PARSE_FAILED",
                                            tool=tcc.name,
                                            raw_args=tcc.arguments_json[:100],
                                        )

                                completed_tool_calls.append(
                                    ToolCall(
                                        id=tcc.id,
                                        name=tcc.name,
                                        arguments=args,
                                    )
                                )

                        # Yield final chunk with tool calls (if any)
                        yield StreamWithToolsChunk(
                            content="",
                            tool_calls=completed_tool_calls if completed_tool_calls else None,
                            is_done=True,
                        )

                        health.mark_success()
                        self._reset_auth_retry_count()

                        duration_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(
                            "LLM_STREAM_WITH_TOOLS_END",
                            endpoint=endpoint.endpoint_identifier,
                            duration_ms=round(duration_ms, 1),
                            content_len=len(accumulated_content),
                            tool_calls=len(completed_tool_calls),
                            finish_reason=finish_reason,
                        )

                        return  # Stream is done

            except RateLimitError:
                # Re-raise our own RateLimitError to be handled by retry wrapper
                raise

            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # OAuth token invalid/expired - force refresh and retry
                # Note: Databricks returns 403 PermissionDeniedError for expired tokens
                duration_ms = (time.perf_counter() - start_time) * 1000
                health.mark_failure(rate_limited=False)

                logger.warning(
                    "LLM_STREAM_WITH_TOOLS_AUTH_ERROR",
                    endpoint=endpoint.endpoint_identifier,
                    status_code=getattr(e, "status_code", None),
                    will_retry=self._credential_provider is not None,
                )

                if await self._force_refresh_token():
                    # Retry by re-yielding from a new stream
                    async for retry_chunk in self._stream_with_tools_impl(
                        messages, tools, tier, role, temperature, max_tokens
                    ):
                        yield retry_chunk
                    return

                raise LLMError(
                    f"Authentication failed: {e}", endpoint=endpoint.id
                ) from e

            except openai.RateLimitError as e:
                # Proper rate limit detection via exception type
                health.mark_failure(rate_limited=True)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=True,
                    will_fallback=False,
                )

                raise RateLimitError(retry_after=30) from e

            except openai.APIStatusError as e:
                # Other API errors with status codes
                is_rate_limit = e.status_code == 429
                health.mark_failure(rate_limited=is_rate_limit)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=is_rate_limit,
                    will_fallback=False,
                )

                if is_rate_limit:
                    raise RateLimitError(retry_after=30) from e
                raise LLMError(str(e), endpoint=endpoint.id) from e

            except openai.APIConnectionError as e:
                # Connection errors (network, timeout)
                health.mark_failure(rate_limited=False)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(
                    f"Connection error: {e}", endpoint=endpoint.id
                ) from e

            except Exception as e:
                # Fallback for unexpected errors
                health.mark_failure(rate_limited=False)

                log_llm_error(
                    logger,
                    endpoint=endpoint.endpoint_identifier,
                    error=e,
                    is_rate_limit=False,
                    will_fallback=False,
                )

                raise LLMError(str(e), endpoint=endpoint.id) from e

        # Wrap with MLflow tracing
        traced = traced_stream_with_tools(
            inner=_inner_stream(),
            tier=tier,
            endpoint=endpoint.endpoint_identifier,
            message_count=len(messages),
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature"),
            messages=messages,
        )

        # Yield from traced iterator
        async for chunk in traced:
            yield chunk

    async def close(self) -> None:
        """Close the underlying HTTP client.

        Must be called before the event loop closes to avoid
        'Event loop is closed' errors during cleanup.
        """
        await self._client.close()
