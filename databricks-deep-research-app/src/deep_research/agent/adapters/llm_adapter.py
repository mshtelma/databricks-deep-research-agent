"""LLM adapter — wraps app's LLMClient → FrameworkLLMClient.

Extracts the ``AsyncOpenAI`` client and model tier mapping from the
app's ``LLMClient`` and ``ModelConfig`` so the framework can make LLM
calls without knowing about Databricks auth, tracing, or health tracking.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from databricks_deep_research import (
    FrameworkLLMClient,
    ModelTierConfig,
)

from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = logging.getLogger(__name__)


def create_framework_llm_client(
    app_llm: LLMClient,
    *,
    embedding_model: str | None = None,
    model_overrides: dict[str, str] | None = None,
) -> FrameworkLLMClient:
    """Create a ``FrameworkLLMClient`` from the app's ``LLMClient``.

    Extracts the underlying ``AsyncOpenAI`` client and builds a model
    tier mapping from the app's ``ModelConfig``.

    Args:
        app_llm: The application's LLM client (handles auth, health, tracing).
        embedding_model: Optional embedding model name for vector search.
        model_overrides: Optional per-tier model overrides from user config.

    Returns:
        A framework-level LLM client ready for workflow execution.
    """
    # Ensure the OpenAI client is initialized
    openai_client = app_llm._ensure_fresh_client()

    # Build model mapping from app config
    model_mapping = _build_model_mapping(app_llm, model_overrides)

    # Build the optional model-family catalog (family -> endpoints). Empty when
    # no families are configured; nodes only set model_family when the catalog is
    # non-empty (the Designer reads the same catalog), so resolution never raises.
    model_families = _build_family_mapping(app_llm)

    # Global endpoint → context-window registry built from ALL configured
    # endpoints (not just those referenced by a tier). This lets the framework
    # escalate an overflowing prompt to a large-window model — e.g. a 1M-token
    # endpoint reserved for overflow — even when no tier lists it.
    endpoint_registry: dict[str, int] = {}
    try:
        for endpoint in app_llm._config.endpoints.values():
            endpoint_registry[endpoint.endpoint_identifier] = endpoint.max_context_window
    except (AttributeError, KeyError, ValueError):
        logger.debug("LLM_ADAPTER_ENDPOINT_REGISTRY_SKIP")

    # Diagnostic: makes "which token is the framework using?" answerable
    # from the log stream. Compare token_prefix against the app's
    # LLM_CLIENT_INITIALIZED / LLM_CLIENT_FORCE_REFRESHED logs to confirm
    # the framework is on the same auth context as the main app.
    logger.info(
        "FWK_LLM_ADAPTER_BIND auth_mode=%s base_url=%s token_prefix=%s***",
        app_llm._auth.auth_mode,
        str(openai_client.base_url)[:80],
        (openai_client.api_key or "")[:8],
    )

    return FrameworkLLMClient(
        openai_client=openai_client,
        model_mapping=model_mapping,
        embedding_model=embedding_model,
        # Active refresh: invalidates DatabricksAuth + SDK cache and mints a
        # fresh token. Previously this was `_ensure_fresh_client` (passive),
        # which returned the same stale client when DatabricksAuth's
        # locally-computed 1h expiry hadn't elapsed — causing the framework's
        # 403-retry to loop with the same invalid bearer.
        client_provider=app_llm.force_refresh_client,
        endpoint_registry=endpoint_registry,
        model_families=model_families,
    )


def _build_model_mapping(
    app_llm: LLMClient,
    overrides: dict[str, str] | None = None,
) -> dict[str, str | ModelTierConfig]:
    """Build framework model tier → model name mapping.

    Maps framework tier names (simple, analytical, complex) to actual
    Databricks endpoint identifiers from the app's configuration.
    Multi-endpoint tiers produce ``ModelTierConfig`` objects so the
    framework can handle fallback and rotation.

    Args:
        app_llm: Application LLM client with model config.
        overrides: Optional per-tier model name overrides.

    Returns:
        Dict mapping tier name to model/endpoint identifier or
        ``ModelTierConfig`` for multi-endpoint tiers.
    """
    mapping: dict[str, str | ModelTierConfig] = {}
    config = app_llm._config

    # Map each app ModelTier to resolved endpoints
    for tier in ModelTier:
        tier_name = tier.value  # "simple", "analytical", "complex", etc.
        try:
            role = config.get_role(tier)
            if not role or not role.endpoints:
                continue
            # Resolve ALL endpoint identifiers, skipping failures. Track each
            # endpoint's context window so the framework can escalate to a
            # larger-window model when a prompt would overflow.
            resolved: list[str] = []
            windows: dict[str, int] = {}
            for ep_id in role.endpoints:
                try:
                    endpoint = config.get_endpoint(ep_id)
                    if endpoint:
                        resolved.append(endpoint.endpoint_identifier)
                        windows[endpoint.endpoint_identifier] = endpoint.max_context_window
                except (KeyError, AttributeError, ValueError):
                    logger.debug("ENDPOINT_RESOLVE_SKIP tier=%s ep=%s", tier_name, ep_id)
            if not resolved:
                continue
            # Always emit a ModelTierConfig (even single-endpoint tiers) so
            # context-window escalation logic is uniform across all tiers.
            mapping[tier_name] = ModelTierConfig(
                endpoints=resolved,
                fallback_on_429=role.fallback_on_429,
                rotation_strategy=cast(Literal["PRIORITY", "ROUND_ROBIN"], role.rotation_strategy.name),
                tokens_per_minute=0,
                endpoint_context_windows=windows,
            )
        except (KeyError, IndexError, AttributeError, ValueError):
            logger.debug("TIER_MAPPING_SKIP tier=%s", tier_name)

    # Apply overrides
    if overrides:
        for tier_name, model_name in overrides.items():
            mapping[tier_name] = model_name

    # Ensure at least the three core tiers have defaults
    if "simple" not in mapping and "analytical" in mapping:
        mapping["simple"] = mapping["analytical"]
    if "complex" not in mapping and "analytical" in mapping:
        logger.warning(
            "COMPLEX_TIER_FALLBACK_TO_ANALYTICAL — check that opus endpoint "
            "is configured in app.yaml and reachable"
        )
        mapping["complex"] = mapping["analytical"]
    if "analytical" not in mapping and mapping:
        # Fallback: use any available endpoint
        fallback = next(iter(mapping.values()))
        for tier_key in ("simple", "analytical", "complex"):
            mapping.setdefault(tier_key, fallback)

    if not mapping:
        raise ValueError(
            "LLM_ADAPTER_NO_TIERS: Could not map any model tiers. "
            "Check that app.yaml 'endpoints' section is configured and "
            "the ModelConfig is accessible. Available tiers: "
            f"{[t.value for t in ModelTier]}"
        )

    logger.info(
        "LLM_ADAPTER_MAPPING tiers=%s",
        {k: (str(v)[:50] if isinstance(v, ModelTierConfig) else v[:30])
         for k, v in mapping.items()},
    )
    return mapping


def _build_family_mapping(app_llm: LLMClient) -> dict[str, str | ModelTierConfig]:
    """Build the framework model-family catalog: family label -> ModelTierConfig.

    Mirrors :func:`_build_model_mapping` but keyed by family label from
    ``AppConfig.model_families`` (family -> endpoint ids), resolving each endpoint
    via the app's ModelConfig. Returns ``{}`` when no families are configured —
    family selection is then unavailable and the Designer assigns no
    ``model_family`` (so the framework client never sees an unknown family).
    """
    from deep_research.core.app_config import get_app_config

    families_cfg = get_app_config().model_families
    if not families_cfg:
        return {}
    config = app_llm._config
    mapping: dict[str, str | ModelTierConfig] = {}
    for family_name, endpoint_ids in families_cfg.items():
        resolved: list[str] = []
        windows: dict[str, int] = {}
        for ep_id in endpoint_ids:
            try:
                endpoint = config.get_endpoint(ep_id)
            except (KeyError, AttributeError, ValueError):
                logger.debug(
                    "FAMILY_ENDPOINT_RESOLVE_SKIP family=%s ep=%s", family_name, ep_id
                )
                continue
            if endpoint:
                resolved.append(endpoint.endpoint_identifier)
                windows[endpoint.endpoint_identifier] = endpoint.max_context_window
        if resolved:
            mapping[family_name] = ModelTierConfig(
                endpoints=resolved,
                fallback_on_429=True,
                rotation_strategy="PRIORITY",
                tokens_per_minute=0,
                endpoint_context_windows=windows,
            )
    if mapping:
        logger.info("LLM_ADAPTER_FAMILIES families=%s", sorted(mapping.keys()))
    return mapping


__all__ = ["create_framework_llm_client"]
