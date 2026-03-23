"""LLM Client: thin wrapper around AsyncOpenAI with tiered model routing.

Maps ModelTier to concrete model names, provides structured output support,
rate-limit-aware endpoint selection, and automatic failover.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import random
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from databricks_deep_research.tracing import get_current_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class ModelTier(StrEnum):
    """Model tier for routing to appropriate endpoints."""

    simple = "simple"
    analytical = "analytical"
    complex = "complex"


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    function_name: str
    arguments: str  # JSON string


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
        self.consecutive_errors = 0
        self.is_healthy = True

    def mark_failure(self, rate_limited: bool = False) -> None:
        """Increment error counter, optionally set rate limit window."""
        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:
            self.is_healthy = False
        if rate_limited:
            # Back off for 60 seconds on rate-limit.
            self.rate_limited_until = time.monotonic() + 60.0

    def can_handle_request(self, estimated_tokens: int, tpm_limit: int) -> bool:
        """Check if endpoint can handle request within TPM budget."""
        now = time.monotonic()

        # Endpoint is rate-limited — not available yet.
        if now < self.rate_limited_until:
            return False

        # Endpoint has had too many consecutive errors.
        if not self.is_healthy:
            return False

        # No TPM limit configured — always OK.
        if tpm_limit <= 0:
            return True

        # Reset the per-minute window if >60 s have elapsed.
        if now - self.minute_started_at >= 60.0:
            self.tokens_used_this_minute = 0
            self.minute_started_at = now

        return self.tokens_used_this_minute + estimated_tokens <= tpm_limit


@dataclass(frozen=True)
class ModelTierConfig:
    """Rich model tier config with multiple endpoints and fallback.

    Replaces simple str mapping when rate limiting / fallback is needed.
    Backward compatible -- str values in model_mapping are auto-wrapped.
    """

    endpoints: list[str]  # Priority-ordered endpoint names
    fallback_on_429: bool = True
    rotation_strategy: Literal["PRIORITY", "ROUND_ROBIN"] = "PRIORITY"
    tokens_per_minute: int = 0  # 0 = unlimited


# ---------------------------------------------------------------------------
# Model config parsing
# ---------------------------------------------------------------------------


def parse_model_config(
    raw: dict[str, Any],
) -> dict[str, str | ModelTierConfig]:
    """Parse a raw dict (from YAML or Python) into a model tier mapping.

    Accepts both simple strings and rich endpoint dicts::

        {"simple": "model-name"}
        {"analytical": {"endpoints": ["a", "b"], "fallback_on_429": true}}

    Raises ValueError for invalid values or rotation strategies.
    """
    _VALID_STRATEGIES = {"PRIORITY", "ROUND_ROBIN"}
    result: dict[str, str | ModelTierConfig] = {}
    for tier, value in raw.items():
        if isinstance(value, str):
            result[tier] = value
        elif isinstance(value, dict):
            strategy = value.get("rotation_strategy", "PRIORITY")
            if isinstance(strategy, str):
                strategy = strategy.upper()
            if strategy not in _VALID_STRATEGIES:
                raise ValueError(
                    f"Invalid rotation_strategy '{strategy}' for tier '{tier}'. "
                    f"Valid: {', '.join(sorted(_VALID_STRATEGIES))}"
                )
            result[tier] = ModelTierConfig(
                endpoints=value["endpoints"],
                fallback_on_429=value.get("fallback_on_429", True),
                rotation_strategy=strategy,
                tokens_per_minute=value.get("tokens_per_minute", 0),
            )
        else:
            raise ValueError(
                f"Invalid model config for tier '{tier}': "
                f"expected str or dict, got {type(value).__name__}"
            )
    return result


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_DEFAULT_ESTIMATED_TOKENS = 4096


class FrameworkLLMClient:
    """Thin wrapper around AsyncOpenAI for framework use.

    Maps ModelTier to concrete model names and provides structured output
    support. This is NOT a Protocol -- the framework depends on openai
    directly.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_mapping: dict[str, str | ModelTierConfig],
        *,
        embedding_model: str | None = None,
        client_provider: Callable[[], AsyncOpenAI] | None = None,
    ) -> None:
        self._client = openai_client
        self._client_provider = client_provider
        self._models = model_mapping
        self._embedding_model = embedding_model
        self._endpoint_health: dict[str, EndpointHealth] = {}
        self._round_robin_index: dict[str, int] = {}
        self._closed = False

    # -- Properties ---------------------------------------------------------

    def _get_client(self) -> AsyncOpenAI:
        """Return the active OpenAI client.

        A configured ``client_provider`` is reserved for explicit client refreshes
        (for example after an auth failure), not for every request. Recreating the
        underlying ``AsyncOpenAI`` client on each call leaks transport resources and
        can surface noisy "Event loop is closed" exceptions during test teardown.
        """
        return self._client

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Access the underlying AsyncOpenAI client."""
        return self._get_client()

    @property
    def supports_embeddings(self) -> bool:
        """Whether an embedding model is configured."""
        return self._embedding_model is not None

    async def aclose(self) -> None:
        """Close the underlying OpenAI client when supported."""
        if self._closed:
            return
        client = self._client
        try:
            aclose = getattr(client, "aclose", None)
            if callable(aclose):
                await aclose()
                self._closed = True
                return

            close = getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
                self._closed = True
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise
            logger.info("LLM_CLIENT_ACLOSE_SKIPPED loop_closed=true")
            self._closed = True

    def close(self) -> None:
        """Best-effort synchronous close for non-async callers."""
        if self._closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.aclose())
            except RuntimeError as exc:
                if "Event loop is closed" not in str(exc):
                    raise
            return

        logger.warning("LLM_CLIENT_CLOSE_DEPRECATED active_event_loop=true use_aclose=true")
        return

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_databricks(
        cls,
        *,
        model: str = "databricks-claude-haiku-4-5",
        model_mapping: dict[str, str | ModelTierConfig] | None = None,
    ) -> FrameworkLLMClient:
        """Create a client authenticated against Databricks serving endpoints.

        Auth chain (tried in order):

        1. **Direct token** — ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` env vars.
        2. **SDK auto-detect** — ``WorkspaceClient()`` with no args (covers
           profiles, Azure MSI, and all other SDK-supported auth methods).
           Uses a ``client_provider`` callback so OAuth tokens refresh
           automatically on long-running workflows.

        Parameters
        ----------
        model:
            Default model endpoint mapped to all tiers (simple, analytical,
            complex) unless *model_mapping* is provided.
        model_mapping:
            Explicit tier → endpoint mapping.  Takes precedence over *model*.
            Values can be plain strings or ``ModelTierConfig`` instances for
            multi-endpoint failover.
        """
        mapping: dict[str, str | ModelTierConfig] = (
            dict(model_mapping)
            if model_mapping is not None
            else {"simple": model, "analytical": model, "complex": model}
        )

        host = os.getenv("DATABRICKS_HOST", "")
        token = os.getenv("DATABRICKS_TOKEN", "")

        # Path 1: direct token (no SDK required; static PAT doesn't need refresh)
        if host and token:
            base_url = f"{host.rstrip('/')}/serving-endpoints"
            client = AsyncOpenAI(api_key=token, base_url=base_url)
            return cls(
                openai_client=client,
                model_mapping=mapping,
            )

        # Path 2: SDK auto-detect (covers profiles, MSI, etc.)
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            sdk_host = (w.config.host or "").rstrip("/")
            if not sdk_host:
                raise RuntimeError("WorkspaceClient resolved but host is empty")

            base_url = f"{sdk_host}/serving-endpoints"

            def _fresh_client() -> AsyncOpenAI:
                headers = w.config.authenticate()
                fresh_token = headers.get("Authorization", "").removeprefix(
                    "Bearer "
                )
                return AsyncOpenAI(api_key=fresh_token, base_url=base_url)

            return cls(
                openai_client=_fresh_client(),
                model_mapping=mapping,
                client_provider=_fresh_client,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not authenticate with Databricks. Either set "
                "DATABRICKS_HOST + DATABRICKS_TOKEN env vars, or configure "
                "Databricks SDK auth (profile, Azure MSI, etc.).\n"
                f"SDK error: {exc}"
            ) from exc

    def derive(
        self,
        model_mapping: dict[str, str | ModelTierConfig],
    ) -> FrameworkLLMClient:
        """Create a new client sharing the same connection with updated model mappings.

        Entries in *model_mapping* are layered on top of the current mappings —
        new entries override same-named tiers; unmentioned tiers are preserved.

        The derived client gets **fresh health state** — endpoint health tracking
        is independent. This is intentional: each workflow run starts with a clean
        slate, preventing stale rate-limit windows from blocking new runs.

        The underlying ``AsyncOpenAI`` connection and ``client_provider`` are
        shared, so no new HTTP connections are created.
        """
        merged = {**self._models, **model_mapping}
        return FrameworkLLMClient(
            openai_client=self._client,
            model_mapping=merged,
            embedding_model=self._embedding_model,
            client_provider=self._client_provider,
        )

    # -- Model resolution ---------------------------------------------------

    def resolve_model(self, tier: str | ModelTier) -> str:
        """Resolve a model tier to a concrete model name.

        For ``str`` values the string itself is the model name.
        For ``ModelTierConfig`` values the best healthy endpoint is selected.
        """
        tier_str = tier.value if isinstance(tier, ModelTier) else tier
        cfg = self._models.get(tier_str)
        if cfg is None:
            raise ValueError(
                f"Unknown model tier: {tier_str}. "
                f"Available: {list(self._models.keys())}"
            )
        if isinstance(cfg, str):
            return cfg
        # ModelTierConfig -- delegate to endpoint selection.
        return self._select_endpoint(tier_str)

    def _get_health(self, endpoint: str) -> EndpointHealth:
        """Get or create EndpointHealth for an endpoint."""
        if endpoint not in self._endpoint_health:
            self._endpoint_health[endpoint] = EndpointHealth()
        return self._endpoint_health[endpoint]

    def _select_endpoint(self, tier: str) -> str:
        """Select best endpoint for *tier* based on health + rotation strategy."""
        cfg = self._models[tier]
        if isinstance(cfg, str):
            return cfg

        endpoints = cfg.endpoints
        if not endpoints:
            raise ValueError(f"No endpoints configured for tier '{tier}'")

        if cfg.rotation_strategy == "ROUND_ROBIN":
            idx = self._round_robin_index.get(tier, 0)
            # Try each endpoint starting from current index.
            for offset in range(len(endpoints)):
                candidate = endpoints[(idx + offset) % len(endpoints)]
                health = self._get_health(candidate)
                if health.can_handle_request(
                    _DEFAULT_ESTIMATED_TOKENS, cfg.tokens_per_minute
                ):
                    self._round_robin_index[tier] = (idx + offset + 1) % len(endpoints)
                    return candidate
            # All endpoints exhausted -- return first (let the caller handle failure).
            return endpoints[0]

        # PRIORITY: try endpoints in declared order.
        for ep in endpoints:
            health = self._get_health(ep)
            if health.can_handle_request(
                _DEFAULT_ESTIMATED_TOKENS, cfg.tokens_per_minute
            ):
                return ep
        # All unhealthy -- return first and hope for the best.
        return endpoints[0]

    def _find_fallback(self, tier: str, failed_endpoint: str) -> str | None:
        """Find a fallback endpoint after 429/failure on the primary."""
        cfg = self._models.get(tier)
        if cfg is None or isinstance(cfg, str):
            return None
        if not cfg.fallback_on_429:
            return None
        for ep in cfg.endpoints:
            if ep == failed_endpoint:
                continue
            health = self._get_health(ep)
            if health.can_handle_request(
                _DEFAULT_ESTIMATED_TOKENS, cfg.tokens_per_minute
            ):
                return ep
        return None

    # -- Embeddings ---------------------------------------------------------

    async def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        """Batch embed texts via OpenAI embeddings.create().

        Args:
            texts: List of strings to embed.
            model: Override embedding model. If *None*, uses configured
                   ``embedding_model``.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If no embedding model is configured and none provided.
        """
        effective_model = model or self._embedding_model
        if effective_model is None:
            raise ValueError(
                "No embedding model configured. Pass one explicitly or set "
                "embedding_model at construction time."
            )

        response = await self._get_client().embeddings.create(
            input=texts,
            model=effective_model,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        """Convenience for embedding a single text."""
        vectors = await self.embed([text])
        return vectors[0]

    # -- Retry helper -------------------------------------------------------

    async def _retry_with_backoff(
        self, func: Any, *, max_retries: int = 3, retry_rate_limit: bool = False,
    ) -> Any:
        """Retry *func* with exponential backoff and jitter on transient failures.

        *func* must be an async callable (zero-arg coroutine factory).

        When *retry_rate_limit* is ``False`` (default), 429 ``RateLimitError``
        is re-raised immediately so higher-level fallback logic can switch
        endpoints without added latency.  When ``True``, 429s are retried
        with backoff (respecting ``Retry-After`` header when present).
        """
        last_exc: BaseException | None = None
        for attempt in range(max_retries):
            try:
                return await func()
            except RateLimitError as exc:
                if not retry_rate_limit:
                    raise  # Let higher-level fallback handle it.
                last_exc = exc
                retry_after: float | None = None
                if hasattr(exc, "response") and exc.response is not None:
                    retry_header = exc.response.headers.get("retry-after")
                    if retry_header:
                        with contextlib.suppress(TypeError, ValueError):
                            retry_after = float(retry_header)
                if attempt < max_retries - 1:
                    backoff = min(retry_after or ((2 ** attempt) + random.random()), 60.0)
                    logger.warning(
                        "LLM_RATE_LIMIT_RETRY attempt=%d/%d backoff=%.2fs retry_after=%s",
                        attempt + 1, max_retries, backoff, retry_after,
                    )
                    await asyncio.sleep(backoff)
                else:
                    raise  # Exhausted retries — higher-level fallback can still try.
            except APITimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "LLM_TIMEOUT_RETRY attempt=%d/%d",
                    attempt + 1,
                    max_retries,
                )
            except APIStatusError as exc:
                if exc.status_code == 403 and self._client_provider is not None:
                    # Token may have been invalidated — refresh and retry ONCE
                    logger.warning(
                        "LLM_AUTH_REFRESH_RETRY status=%d attempt=%d",
                        exc.status_code, attempt + 1,
                    )
                    self._client = self._client_provider()
                    last_exc = exc
                    if attempt == 0:
                        continue
                    raise
                if exc.status_code == 403:
                    logger.error(
                        "LLM_TOKEN_EXPIRED_NO_PROVIDER status=403 "
                        "hint=configure WorkspaceClient or set DATABRICKS_TOKEN env var"
                    )
                if exc.status_code < 500:
                    raise  # Client errors (4xx) are never transient.
                # Server errors (5xx) fall through to retry.
                last_exc = exc
                if attempt < max_retries - 1:
                    backoff = (2**attempt) + random.random()
                    logger.warning(
                        "LLM_RETRY attempt=%d backoff=%.2fs error=%s",
                        attempt + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    backoff = (2**attempt) + random.random()
                    logger.warning(
                        "LLM_RETRY attempt=%d backoff=%.2fs error=%s",
                        attempt + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
        raise last_exc  # type: ignore[misc]

    # -- Standard completion helpers ----------------------------------------

    async def _standard_completion(
        self,
        kwargs: dict[str, Any],
        *,
        try_json_format: bool = False,
    ) -> LLMResponse:
        """Run a standard (non-structured) chat completion.

        When *try_json_format* is True (used after structured output
        validation failures), attempts ``response_format: json_object``
        first.  If the model returns 400 (unsupported), transparently
        retries without it — the harness ``_parse_output()`` handles
        raw text via the agent's ``output_format`` config.
        """
        if try_json_format:
            try:
                kwargs["response_format"] = {"type": "json_object"}
                return await self._do_standard_call(kwargs)
            except APIStatusError as exc:
                if exc.status_code == 400:
                    logger.info(
                        "JSON_FALLBACK_UNSUPPORTED model=%s",
                        kwargs.get("model", "unknown"),
                    )
                    kwargs.pop("response_format", None)
                    # Fall through to call without response_format
                else:
                    raise

        return await self._do_standard_call(kwargs)

    async def _do_standard_call(
        self, kwargs: dict[str, Any]
    ) -> LLMResponse:
        """Execute a single standard chat completion and wrap the response."""
        resp = await self._get_client().chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=_convert_tool_calls(choice.message.tool_calls),
            usage=_extract_usage(resp),
            model=resp.model,
            finish_reason=choice.finish_reason or "stop",
        )

    # -- Completions --------------------------------------------------------

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
        """Send messages to the LLM and return an LLMResponse.

        When a ``ModelTierConfig`` is configured for the given tier the client
        will:

        1. Select the best healthy endpoint via ``_select_endpoint()``.
        2. Track token usage against TPM limits.
        3. On 429 errors, automatically try fallback endpoints if
           ``fallback_on_429=True``.
        4. Use exponential backoff with jitter for transient failures.
        5. Mark endpoints healthy/unhealthy based on success/failure.
        """
        tier_str = tier.value if isinstance(tier, ModelTier) else tier
        model_name = self.resolve_model(tier_str)

        logger.info(
            "FWK_LLM_CALL tier=%s model=%s base_url=%s token_prefix=%s has_tools=%s structured=%s",
            tier_str, model_name,
            str(self._get_client().base_url)[:60],
            (self._get_client().api_key or "")[:8] + "***",
            bool(tools),
            structured_output is not None,
        )

        async def _do_call(model: str) -> LLMResponse:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
            if response_format is not None and structured_output is None:
                kwargs["response_format"] = response_format

            # -- Structured output (Pydantic model class) -------------------
            if structured_output is not None:
                try:
                    parsed_resp = await self._get_client().beta.chat.completions.parse(
                        response_format=structured_output,
                        **kwargs,
                    )
                    choice = parsed_resp.choices[0]
                    usage_dict = _extract_usage(parsed_resp)
                    return LLMResponse(
                        content=choice.message.content or "",
                        tool_calls=_convert_tool_calls(choice.message.tool_calls),
                        usage=usage_dict,
                        model=parsed_resp.model,
                        finish_reason=choice.finish_reason or "stop",
                        structured=choice.message.parsed,
                    )
                except Exception as exc:
                    if "ValidationError" in type(exc).__name__:
                        logger.warning(
                            "STRUCTURED_OUTPUT_VALIDATION_FAILED model=%s error=%s",
                            kwargs.get("model", "unknown"),
                            str(exc)[:300],
                        )
                        # Fall through — _standard_completion tries json_object
                        # then degrades gracefully if unsupported (400).
                        return await self._standard_completion(
                            kwargs, try_json_format=True
                        )
                    raise

            # -- Standard completion ----------------------------------------
            return await self._standard_completion(kwargs)

        # Execute with retry + fallback ------------------------------------
        cfg = self._models.get(tier_str)
        is_tier_config = isinstance(cfg, ModelTierConfig)

        # Retry 429 only when no per-request fallback is configured
        should_retry_rl = isinstance(cfg, ModelTierConfig) and not cfg.fallback_on_429
        try:
            result = await self._retry_with_backoff(
                lambda _m=model_name: _do_call(_m),
                retry_rate_limit=should_retry_rl,
            )
            if is_tier_config:
                health = self._get_health(model_name)
                health.mark_success()
                total = result.usage.get("total_tokens", 0)
                health.tokens_used_this_minute += total
            span = get_current_span()
            if span is not None:
                span.set_attributes({"llm.model": model_name, "llm.tier": tier_str})
            return result  # type: ignore[no-any-return]

        except RateLimitError:
            if is_tier_config:
                health = self._get_health(model_name)
                health.mark_failure(rate_limited=True)
                logger.warning(
                    "LLM_RATE_LIMITED endpoint=%s tier=%s",
                    model_name,
                    tier_str,
                )
                fallback = self._find_fallback(tier_str, model_name)
                if fallback is not None:
                    logger.info(
                        "LLM_FALLBACK from=%s to=%s tier=%s",
                        model_name,
                        fallback,
                        tier_str,
                    )
                    try:
                        # Fallback endpoint — always retry since this is the last resort
                        result = await self._retry_with_backoff(
                            lambda _m=fallback: _do_call(_m),
                            retry_rate_limit=True,
                        )
                        fb_health = self._get_health(fallback)
                        fb_health.mark_success()
                        total = result.usage.get("total_tokens", 0)
                        fb_health.tokens_used_this_minute += total
                        span = get_current_span()
                        if span is not None:
                            span.set_attributes({"llm.model": fallback, "llm.tier": tier_str})
                        return result  # type: ignore[no-any-return]
                    except RateLimitError:
                        fb_health = self._get_health(fallback)
                        fb_health.mark_failure(rate_limited=True)
            raise

        except Exception:
            if is_tier_config:
                health = self._get_health(model_name)
                health.mark_failure()
            raise

    # -- Streaming ----------------------------------------------------------

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

        Yields ``str`` chunks for content deltas and ``ToolCall`` objects for
        completed tool calls.
        """
        tier_str = tier.value if isinstance(tier, ModelTier) else tier
        model_name = self.resolve_model(tier_str)
        cfg = self._models.get(tier_str)
        is_tier_config = isinstance(cfg, ModelTierConfig)

        logger.info(
            "FWK_LLM_STREAM tier=%s model=%s base_url=%s token_prefix=%s has_tools=%s",
            tier_str, model_name,
            str(self._get_client().base_url)[:60],
            (self._get_client().api_key or "")[:8] + "***",
            bool(tools),
        )

        async def _open_stream(model: str) -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
            return await self._get_client().chat.completions.create(**kwargs)

        # Attempt to open the stream, with rate-limit fallback.
        try:
            response_stream = await self._retry_with_backoff(
                lambda _m=model_name: _open_stream(_m)
            )
        except RateLimitError:
            if is_tier_config:
                health = self._get_health(model_name)
                health.mark_failure(rate_limited=True)
                fallback = self._find_fallback(tier_str, model_name)
                if fallback is not None:
                    logger.info(
                        "LLM_STREAM_FALLBACK from=%s to=%s", model_name, fallback
                    )
                    response_stream = await self._retry_with_backoff(
                        lambda _m=fallback: _open_stream(_m)
                    )
                    model_name = fallback
                else:
                    raise
            else:
                raise

        # Consume the stream.
        # tool_calls_acc maps index -> {id, name, arguments_parts}
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        async for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Content tokens.
            if delta.content:
                yield delta.content

            # Tool-call deltas -- accumulate until finish_reason="tool_calls".
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments_parts": [],
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        acc["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        acc["arguments_parts"].append(
                            tc_delta.function.arguments
                        )

        # Yield accumulated tool calls after stream ends.
        for _idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[_idx]
            yield ToolCall(
                id=acc["id"],
                function_name=acc["name"],
                arguments="".join(acc["arguments_parts"]),
            )

        # Mark endpoint health.
        if is_tier_config:
            health = self._get_health(model_name)
            health.mark_success()


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract usage dict from an OpenAI response object."""
    if response.usage is None:
        return {}
    return {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def _convert_tool_calls(
    raw_tool_calls: Any,
) -> list[ToolCall]:
    """Convert OpenAI tool-call objects to framework ToolCall instances."""
    if not raw_tool_calls:
        return []
    return [
        ToolCall(
            id=tc.id,
            function_name=tc.function.name,
            arguments=tc.function.arguments,
        )
        for tc in raw_tool_calls
    ]
