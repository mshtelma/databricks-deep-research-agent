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
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from databricks_deep_research.errors import ContextWindowExceededError
from databricks_deep_research.events.types import ModelCallEvent, StreamEvent
from databricks_deep_research.llm.budget import estimate_message_tokens
from databricks_deep_research.llm.roles import OPENAI_CHAT_ROLES
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
    max_retries: int = 3  # Max retry attempts for rate limits / transient errors
    retry_base_backoff: float = 2.0  # Base seconds for exponential backoff
    # Per-endpoint context window (tokens) for this tier's endpoints.
    # endpoint identifier -> max_context_window. 0/absent = unknown.
    endpoint_context_windows: dict[str, int] = field(default_factory=dict)
    # Behavior when no available endpoint can fit the prompt even after
    # escalating to the largest known window: "truncate" (default — shrink the
    # prompt to the largest window and warn) or "fail" (raise
    # ContextWindowExceededError).
    on_context_overflow: Literal["truncate", "fail"] = "truncate"


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
            overflow = value.get("on_context_overflow", "truncate")
            if overflow not in ("truncate", "fail"):
                raise ValueError(
                    f"Invalid on_context_overflow '{overflow}' for tier "
                    f"'{tier}'. Valid: truncate, fail"
                )
            result[tier] = ModelTierConfig(
                endpoints=value["endpoints"],
                fallback_on_429=value.get("fallback_on_429", True),
                rotation_strategy=strategy,
                tokens_per_minute=value.get("tokens_per_minute", 0),
                max_retries=value.get("max_retries", 3),
                retry_base_backoff=float(value.get("retry_base_backoff", 2.0)),
                endpoint_context_windows=dict(
                    value.get("endpoint_context_windows", {})
                ),
                on_context_overflow=overflow,
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
# Output tokens to reserve when the caller does not specify max_tokens, used
# only for the context-window fit check (not sent to the API).
_DEFAULT_OUTPUT_RESERVE = 4096
# Headroom for tokenization variance + system framing on the fit check.
_CONTEXT_SAFETY_MARGIN = 1024


def _truncate_messages_to_tokens(
    messages: list[dict[str, Any]], max_input_tokens: int
) -> list[dict[str, Any]]:
    """Shrink *messages* to fit within *max_input_tokens* (best-effort).

    Last-resort path when no endpoint can hold the prompt. Strategy:

    1. Always keep system messages and the final message.
    2. Keep the longest possible suffix of the remaining conversation that
       fits the budget, dropping from the OLDEST non-system message inward.
    3. Never begin the kept suffix on an orphan ``tool`` message (it would lack
       the preceding assistant ``tool_calls`` and the API would reject it).
    4. If even the minimal kept set overflows, hard-truncate message contents.

    Returns a new list; the input is not mutated.
    """
    if max_input_tokens <= 0:
        max_input_tokens = 1

    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not rest:
        return list(messages)

    # Grow a suffix from the end until adding the next-oldest would overflow.
    kept_rest: list[dict[str, Any]] = []
    for msg in reversed(rest):
        candidate = [msg, *kept_rest]
        if (
            estimate_message_tokens([*system_msgs, *candidate]) <= max_input_tokens
            or not kept_rest
        ):
            kept_rest = candidate
        else:
            break

    # Don't start on an orphan tool message.
    while kept_rest and kept_rest[0].get("role") == "tool":
        kept_rest = kept_rest[1:]
    if not kept_rest:
        kept_rest = [rest[-1]]

    result = [*system_msgs, *kept_rest]

    # If still over budget, hard-truncate the largest string contents. Reserve
    # for per-message framing overhead + the truncation marker so the
    # post-truncation estimate actually lands under budget.
    if estimate_message_tokens(result) > max_input_tokens:
        marker = "\n...[truncated to fit context window]"
        reserve = len(result) * 16 + len(marker)
        budget_chars = max(1, max_input_tokens * 4 - reserve)
        running = 0
        clipped: list[dict[str, Any]] = []
        for msg in result:
            content = msg.get("content")
            if isinstance(content, str):
                remaining = max(0, budget_chars - running)
                if len(content) > remaining:
                    new_msg = dict(msg)
                    new_msg["content"] = content[:remaining] + marker
                    clipped.append(new_msg)
                    running = budget_chars
                    continue
                running += len(content)
            clipped.append(msg)
        result = clipped

    return result


class UnknownModelFamilyError(ValueError):
    """Raised when a node requests a model family absent from the configured
    ``model_families`` catalog. Fail-closed: an unconfigured family must surface
    rather than silently falling back to a tier (which would break the per-node
    family contract)."""

    def __init__(self, family: str, available: list[str]) -> None:
        super().__init__(
            f"Unknown model family {family!r}. Configured families: {available}."
        )
        self.family = family
        self.available = available


class FrameworkLLMClient:
    """Thin wrapper around AsyncOpenAI for framework use.

    Maps ModelTier to concrete model names and provides structured output
    support. This is NOT a Protocol -- the framework depends on openai
    directly.

    ``model_families`` is an OPTIONAL orthogonal axis: a node may set
    ``config.model_family`` (e.g. ``"claude"``) to pin its LLM family regardless
    of ``model_tier``. Family configs are stored alongside tiers (so endpoint
    selection / health / rotation / 429-fallback all work unchanged, keyed by the
    family name), but the 3-tier validation elsewhere is untouched — family is a
    separate, optional field. When unset, routing is tier-based exactly as before.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_mapping: dict[str, str | ModelTierConfig],
        *,
        embedding_model: str | None = None,
        client_provider: Callable[[], AsyncOpenAI] | None = None,
        endpoint_registry: dict[str, int] | None = None,
        model_families: dict[str, str | ModelTierConfig] | None = None,
    ) -> None:
        self._client = openai_client
        self._client_provider = client_provider
        # Copy so we never mutate the caller's mapping when merging families.
        self._models = dict(model_mapping)
        # Family configs share the resolution table (keyed by family name) so
        # _select_endpoint / _find_fallback / health / rotation work unchanged.
        # _family_keys is the validation set distinguishing families from tiers.
        self._family_keys: set[str] = set()
        for _fam_name, _fam_cfg in (model_families or {}).items():
            self._models[_fam_name] = _fam_cfg
            self._family_keys.add(_fam_name)
        self._embedding_model = embedding_model
        self._endpoint_health: dict[str, EndpointHealth] = {}
        self._round_robin_index: dict[str, int] = {}
        self._closed = False
        # endpoint identifier -> max_context_window (tokens). Used by
        # context-window-aware escalation to reach ANY known endpoint —
        # including ones referenced by no tier (e.g. a large-window model
        # reserved for overflow). Backfilled from each tier/family window map so a
        # registry is never required for tier endpoints to be window-aware.
        self._endpoint_registry: dict[str, int] = dict(endpoint_registry or {})
        for _cfg in self._models.values():
            if isinstance(_cfg, ModelTierConfig):
                for _ep, _win in _cfg.endpoint_context_windows.items():
                    self._endpoint_registry.setdefault(_ep, _win)

    @property
    def model_families(self) -> frozenset[str]:
        """Configured model-family names (empty when none are configured)."""
        return frozenset(self._family_keys)

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
        model_families: dict[str, str | ModelTierConfig] | None = None,
        profile: str | None = None,
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
        profile:
            Databricks CLI profile name from ``~/.databrickscfg``.
            When provided, passed directly to ``WorkspaceClient(profile=...)``.
            When *None* (default), the SDK resolves auth automatically
            (including the ``DATABRICKS_CONFIG_PROFILE`` env var).
            Ignored when ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` are set
            (Path 1 takes precedence).
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
                model_families=model_families,
            )

        # Path 2: SDK auto-detect (covers profiles, MSI, etc.)
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
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
                model_families=model_families,
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
        # Preserve the family catalog across derive (per-run derived clients must
        # still resolve model_family); the configs already ride in ``merged``.
        family_cfgs = {k: self._models[k] for k in self._family_keys}
        return FrameworkLLMClient(
            openai_client=self._client,
            model_mapping=merged,
            embedding_model=self._embedding_model,
            client_provider=self._client_provider,
            endpoint_registry=self._endpoint_registry,
            model_families=family_cfgs,
        )

    # -- Model resolution ---------------------------------------------------

    def _resolution_key(self, tier: str | ModelTier, family: str | None) -> str:
        """Return the ``self._models`` key to resolve from.

        A configured ``family`` overrides the tier (orthogonal axis). An
        unconfigured family fails closed (``UnknownModelFamilyError``) rather than
        silently degrading to the tier.
        """
        if family is not None:
            if family not in self._family_keys:
                raise UnknownModelFamilyError(family, sorted(self._family_keys))
            return family
        return tier.value if isinstance(tier, ModelTier) else tier

    def resolve_model(
        self, tier: str | ModelTier, family: str | None = None
    ) -> str:
        """Resolve a model tier (or family) to a concrete model name.

        For ``str`` values the string itself is the model name.
        For ``ModelTierConfig`` values the best healthy endpoint is selected.
        When ``family`` is set it overrides the tier for endpoint selection.
        """
        key = self._resolution_key(tier, family)
        cfg = self._models.get(key)
        if cfg is None:
            raise ValueError(
                f"Unknown model {'family' if family else 'tier'}: {key}. "
                f"Available: {list(self._models.keys())}"
            )
        if isinstance(cfg, str):
            return cfg
        # ModelTierConfig -- delegate to endpoint selection.
        return self._select_endpoint(key)

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

    def _find_fallback(
        self, tier: str, failed_endpoint: str, required_total: int = 0
    ) -> str | None:
        """Find a fallback endpoint after 429/failure on the primary.

        When *required_total* > 0, candidates whose context window is known to
        be smaller than the prompt are skipped — otherwise a 429-fallback could
        drop into a too-small sibling and re-trigger a 400 "prompt too long".
        """
        cfg = self._models.get(tier)
        if cfg is None or isinstance(cfg, str):
            return None
        if not cfg.fallback_on_429:
            return None
        for ep in cfg.endpoints:
            if ep == failed_endpoint:
                continue
            if required_total > 0 and not self._window_fits(ep, required_total):
                continue
            health = self._get_health(ep)
            if health.can_handle_request(
                _DEFAULT_ESTIMATED_TOKENS, cfg.tokens_per_minute
            ):
                return ep
        return None

    # -- Context-window-aware escalation ------------------------------------

    def _window_of(self, endpoint: str) -> int:
        """Return the known context window for *endpoint*, or 0 if unknown."""
        return self._endpoint_registry.get(endpoint, 0)

    def _window_fits(self, endpoint: str, required_total: int) -> bool:
        """Whether *endpoint*'s known window can hold *required_total* tokens.

        Endpoints with an unknown window (0) are treated as NOT a safe
        escalation target — we never knowingly route a large prompt into a
        model whose capacity we cannot verify.
        """
        window = self._window_of(endpoint)
        return window >= required_total if window > 0 else False

    def _select_context_fit_endpoint(
        self, tier: str, required_total: int
    ) -> tuple[str, str | None]:
        """Select an endpoint whose context window fits *required_total*.

        Returns ``(chosen_endpoint, escalated_from)`` where ``escalated_from``
        is ``None`` when the normally-selected endpoint already fits.

        Selection order:
          1. The normally-selected (health/TPM-aware) primary, if it fits.
          2. Other endpoints of the CURRENT tier that fit, in priority order.
          3. The smallest-window endpoint across ALL known endpoints that fits.

        When nothing fits, returns ``(primary, None)`` — the caller decides
        whether to truncate or fail.
        """
        primary = self._select_endpoint(tier)
        # Unknown primary window (0) → cannot prove it overflows; leave as-is.
        primary_window = self._window_of(primary)
        if primary_window <= 0 or primary_window >= required_total:
            return primary, None

        cfg = self._models.get(tier)
        if isinstance(cfg, ModelTierConfig):
            for ep in cfg.endpoints:
                if ep == primary:
                    continue
                if self._window_fits(ep, required_total) and self._get_health(
                    ep
                ).can_handle_request(_DEFAULT_ESTIMATED_TOKENS, cfg.tokens_per_minute):
                    return ep, primary

        # Global escalation: smallest fitting window across all endpoints.
        fitting = sorted(
            (
                ep
                for ep in self._endpoint_registry
                if self._window_fits(ep, required_total)
            ),
            key=self._window_of,
        )
        for ep in fitting:
            if self._get_health(ep).can_handle_request(_DEFAULT_ESTIMATED_TOKENS, 0):
                return ep, primary

        return primary, None

    def _largest_known_endpoint(self) -> tuple[str, int]:
        """Return the (endpoint, window) with the largest known window.

        Returns ``("", 0)`` when no endpoint windows are known.
        """
        best_ep = ""
        best_win = 0
        for ep, win in self._endpoint_registry.items():
            if win > best_win:
                best_ep, best_win = ep, win
        return best_ep, best_win

    def _resolve_for_context(
        self,
        tier_str: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        *,
        structured_output: type | None = None,
    ) -> tuple[str, list[dict[str, Any]], int]:
        """Pick an endpoint that fits the prompt, escalating/truncating as needed.

        Returns ``(model_name, messages, required_total)``. ``messages`` may be a
        truncated copy when no endpoint can fit and the overflow policy is
        ``truncate``. Raises :class:`ContextWindowExceededError` when no endpoint
        fits and the policy is ``fail``.
        """
        est_input = estimate_message_tokens(messages, tools)
        if structured_output is not None:
            try:
                schema = structured_output.model_json_schema()  # type: ignore[attr-defined]
                est_input += len(str(schema)) // 4
            except Exception:  # pragma: no cover - schema introspection best-effort
                pass
        output_reserve = max_tokens or _DEFAULT_OUTPUT_RESERVE
        required_total = est_input + output_reserve + _CONTEXT_SAFETY_MARGIN

        model_name, escalated_from = self._select_context_fit_endpoint(
            tier_str, required_total
        )
        if escalated_from is not None:
            logger.warning(
                "LLM_CONTEXT_ESCALATION tier=%s from=%s from_window=%d to=%s "
                "to_window=%d est_input=%d output_reserve=%d safety_margin=%d "
                "required_total=%d reason=context_overflow",
                tier_str, escalated_from, self._window_of(escalated_from),
                model_name, self._window_of(model_name), est_input,
                output_reserve, _CONTEXT_SAFETY_MARGIN, required_total,
            )
            span = get_current_span()
            if span is not None:
                span.set_attributes({
                    "llm.context.escalated_from": escalated_from,
                    "llm.context.escalated_to": model_name,
                    "llm.context.required_tokens": required_total,
                })
            return model_name, messages, required_total

        # No escalation chosen. Either the primary fits, or nothing fits.
        primary_window = self._window_of(model_name)
        if primary_window <= 0 or primary_window >= required_total:
            return model_name, messages, required_total

        # Nothing fits — apply the configured overflow policy.
        cfg = self._models.get(tier_str)
        policy = (
            cfg.on_context_overflow if isinstance(cfg, ModelTierConfig) else "truncate"
        )
        largest_ep, largest_win = self._largest_known_endpoint()
        if policy == "fail":
            logger.error(
                "LLM_CONTEXT_OVERFLOW_FATAL tier=%s required_total=%d "
                "largest_endpoint=%s largest_window=%d",
                tier_str, required_total, largest_ep, largest_win,
            )
            raise ContextWindowExceededError(
                required_total,
                largest_win,
                tried=sorted(self._endpoint_registry.items()),
            )

        # Last resort: route to the largest-window endpoint and truncate to fit.
        target_ep = largest_ep or model_name
        target_win = largest_win or primary_window
        n_before = len(messages)
        truncated = _truncate_messages_to_tokens(
            messages, max(1, target_win - output_reserve - _CONTEXT_SAFETY_MARGIN)
        )
        logger.warning(
            "LLM_CONTEXT_OVERFLOW_TRUNCATE tier=%s required_total=%d "
            "target_endpoint=%s target_window=%d messages_before=%d "
            "messages_after=%d est_after=%d",
            tier_str, required_total, target_ep, target_win, n_before,
            len(truncated), estimate_message_tokens(truncated, tools),
        )
        return target_ep, truncated, required_total

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
        self,
        func: Any,
        *,
        max_retries: int = 3,
        retry_rate_limit: bool = False,
        retry_base_backoff: float = 2.0,
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
                    backoff = min(
                        retry_after or (retry_base_backoff * (2 ** attempt) + random.random()),
                        60.0,
                    )
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
                # An expired/invalid OAuth bearer surfaces as either 401
                # (Unauthorized) or 403 (Forbidden) depending on the serving
                # edge — refresh on BOTH, not just 403, otherwise token expiry
                # blocks every call until the app is redeployed.
                if exc.status_code in (401, 403) and self._client_provider is not None:
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
                if exc.status_code in (401, 403):
                    logger.error(
                        "LLM_TOKEN_EXPIRED_NO_PROVIDER status=%d "
                        "hint=configure WorkspaceClient or set DATABRICKS_TOKEN env var",
                        exc.status_code,
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
        # Diagnostic: log the EXACT shape that reaches openai.create() — roles
        # only, plus the count + presence of tools/tool_choice/response_format.
        # The body content is intentionally NOT logged (PII / size). This lets
        # us debug "assistant message prefill" rejections from Databricks
        # Sonnet 4.6 without trawling logs for the wrong cause.
        try:
            _msgs = kwargs.get("messages") or []
            _roles = [str(m.get("role", "?")) for m in _msgs if isinstance(m, dict)]
            _last = _roles[-1] if _roles else "<empty>"
            # Observability backstop: a role outside the gateway's accepted set
            # would yield an opaque 400 "Invalid role". Surface it loudly with
            # context instead. We warn (not coerce) here so a real upstream bug
            # in non-history messages is caught rather than silently masked;
            # conversation history is already normalized at assembly time.
            _bad_roles = sorted({r for r in _roles if r not in OPENAI_CHAT_ROLES})
            if _bad_roles:
                logger.warning(
                    "LLM_API_CALL_INVALID_ROLE model=%s bad_roles=%s all_roles=%s",
                    kwargs.get("model", "?"),
                    _bad_roles,
                    _roles,
                )
            logger.info(
                "LLM_API_CALL model=%s n_msgs=%d last_role=%s roles=%s has_tools=%s tool_choice=%s response_format=%s",
                kwargs.get("model", "?"),
                len(_msgs),
                _last,
                _roles,
                bool(kwargs.get("tools")),
                kwargs.get("tool_choice", "<none>"),
                bool(kwargs.get("response_format")),
            )
        except Exception:  # pragma: no cover - defensive; logging must not break the call
            pass
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
        event_sink: Callable[[StreamEvent], None] | None = None,
        node_id: str = "",
        family: str | None = None,
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
        # A configured model family overrides the tier for endpoint resolution
        # (orthogonal axis); tier_str stays the human-facing capability label for
        # logs/spans. resolution_key drives _models lookup + 429-fallback so a
        # family-pinned call falls back WITHIN its family, never to a tier.
        resolution_key = self._resolution_key(tier, family)
        # Context-window-aware selection: escalate to a larger-window endpoint
        # (or truncate as a last resort) when the prompt would overflow the
        # normally-selected model. ``messages`` may be replaced with a
        # truncated copy.
        model_name, messages, required_total = self._resolve_for_context(
            resolution_key,
            messages,
            tools,
            max_tokens,
            structured_output=structured_output,
        )

        logger.info(
            "FWK_LLM_CALL tier=%s family=%s model=%s base_url=%s token_prefix=%s has_tools=%s structured=%s",
            tier_str, family or "-", model_name,
            str(self._get_client().base_url)[:60],
            (self._get_client().api_key or "")[:8] + "***",
            bool(tools),
            structured_output is not None,
        )

        if event_sink is not None:
            event_sink(ModelCallEvent(
                node_id=node_id,
                timestamp=datetime.now(tz=UTC).isoformat(),
                tier=tier_str,
                model=model_name,
            ))

        async def _do_call(model: str) -> LLMResponse:
            # Multi-model conversation-shape compat: GPT-family endpoints
            # reject conversations ending with an assistant turn. Claude
            # tolerates it. Apply per-call so the correct shape is used
            # even when a 429-fallback switches us from Claude → GPT.
            api_messages = _ensure_user_suffix_for_gpt(messages, model)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": api_messages,
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
        cfg = self._models.get(resolution_key)
        is_tier_config = isinstance(cfg, ModelTierConfig)

        # Retry 429 only when no per-request fallback is configured
        should_retry_rl = isinstance(cfg, ModelTierConfig) and not cfg.fallback_on_429
        try:
            result = await self._retry_with_backoff(
                lambda _m=model_name: _do_call(_m),
                retry_rate_limit=should_retry_rl,
                max_retries=cfg.max_retries if isinstance(cfg, ModelTierConfig) else 3,
                retry_base_backoff=cfg.retry_base_backoff if isinstance(cfg, ModelTierConfig) else 2.0,
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
                fallback = self._find_fallback(
                    resolution_key, model_name, required_total
                )
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
                            max_retries=cfg.max_retries if isinstance(cfg, ModelTierConfig) else 3,
                            retry_base_backoff=cfg.retry_base_backoff if isinstance(cfg, ModelTierConfig) else 2.0,
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
        # Context-window-aware selection (mirrors complete()). Raises/truncates
        # BEFORE opening the stream — a 400 mid-open is not cleanly recoverable.
        model_name, messages, required_total = self._resolve_for_context(
            tier_str, messages, tools, max_tokens
        )
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
            # Same GPT-compat treatment as complete(); see helper docstring.
            api_messages = _ensure_user_suffix_for_gpt(messages, model)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": api_messages,
                "stream": True,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
            return await self._get_client().chat.completions.create(**kwargs)

        # Retry 429 only when no per-request fallback is configured
        should_retry_rl = isinstance(cfg, ModelTierConfig) and not cfg.fallback_on_429

        # Attempt to open the stream, with rate-limit fallback.
        try:
            response_stream = await self._retry_with_backoff(
                lambda _m=model_name: _open_stream(_m),
                retry_rate_limit=should_retry_rl,
                max_retries=cfg.max_retries if isinstance(cfg, ModelTierConfig) else 3,
                retry_base_backoff=cfg.retry_base_backoff if isinstance(cfg, ModelTierConfig) else 2.0,
            )
        except RateLimitError:
            if is_tier_config:
                health = self._get_health(model_name)
                health.mark_failure(rate_limited=True)
                fallback = self._find_fallback(
                    tier_str, model_name, required_total
                )
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


def _is_gpt_endpoint(model: str) -> bool:
    """True when the resolved model name looks like a GPT-family endpoint.

    Heuristic: lowercase substring "gpt" matches Databricks identifiers like
    ``databricks-gpt-5-4``, ``databricks-gpt-5-mini``, ``databricks-gpt-5-nano``.
    Does NOT match Claude, Gemini, Llama. A hypothetical hybrid name
    containing "gpt" would false-positive, but the consequence is a harmless
    no-op `Continue.` user message — not a correctness issue.
    """
    return "gpt" in (model or "").lower()


def _ensure_user_suffix_for_gpt(
    messages: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Ensure messages end with a user-role turn when targeting a GPT model.

    GPT-family endpoints (e.g. ``databricks-gpt-5-4``) reject conversations
    that end with an ``assistant`` role:

        BadRequestError: This model does not support assistant message
        prefill. The conversation must end with a user message.

    Claude tolerates the assistant suffix (used for prefill prompting).
    To keep both providers callable from the same harness path, this helper
    is invoked at the LLM call site immediately before
    ``chat.completions.create``: when the resolved model is GPT-family AND
    the final message is from the assistant, append a no-op
    ``{"role": "user", "content": "Continue."}`` so the API accepts it.

    Returns the (possibly augmented) message list. Pure function, never
    mutates the input.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return messages
    if not _is_gpt_endpoint(model):
        return messages
    return [*messages, {"role": "user", "content": "Continue."}]


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
