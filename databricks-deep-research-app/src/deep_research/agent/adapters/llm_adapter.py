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

    return FrameworkLLMClient(
        openai_client=openai_client,
        model_mapping=model_mapping,
        embedding_model=embedding_model,
        client_provider=app_llm._ensure_fresh_client,
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
            # Resolve ALL endpoint identifiers, skipping failures
            resolved: list[str] = []
            for ep_id in role.endpoints:
                try:
                    endpoint = config.get_endpoint(ep_id)
                    if endpoint:
                        resolved.append(endpoint.endpoint_identifier)
                except (KeyError, AttributeError, ValueError):
                    logger.debug("ENDPOINT_RESOLVE_SKIP tier=%s ep=%s", tier_name, ep_id)
            if not resolved:
                continue
            if len(resolved) == 1:
                mapping[tier_name] = resolved[0]
            else:
                mapping[tier_name] = ModelTierConfig(
                    endpoints=resolved,
                    fallback_on_429=role.fallback_on_429,
                    rotation_strategy=cast(Literal["PRIORITY", "ROUND_ROBIN"], role.rotation_strategy.name),
                    tokens_per_minute=0,
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


__all__ = ["create_framework_llm_client"]
