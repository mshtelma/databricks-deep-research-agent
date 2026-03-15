"""LLM adapter — wraps app's LLMClient → FrameworkLLMClient.

Extracts the ``AsyncOpenAI`` client and model tier mapping from the
app's ``LLMClient`` and ``ModelConfig`` so the framework can make LLM
calls without knowing about Databricks auth, tracing, or health tracking.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research import FrameworkLLMClient
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
) -> dict[str, str]:
    """Build framework model tier → model name mapping.

    Maps framework tier names (simple, analytical, complex) to actual
    Databricks endpoint identifiers from the app's configuration.

    Args:
        app_llm: Application LLM client with model config.
        overrides: Optional per-tier model name overrides.

    Returns:
        Dict mapping tier name to model/endpoint identifier.
    """
    mapping: dict[str, str] = {}
    config = app_llm._config

    # Map each app ModelTier to the primary endpoint
    for tier in ModelTier:
        tier_name = tier.value  # "simple", "analytical", "complex", etc.
        try:
            role = config.get_role(tier)
            if role and role.endpoints:
                primary = role.endpoints[0]
                endpoint = config.get_endpoint(primary)
                if endpoint:
                    mapping[tier_name] = endpoint.endpoint_identifier
        except (KeyError, IndexError, AttributeError):
            logger.debug("TIER_MAPPING_SKIP tier=%s", tier_name)

    # Apply overrides
    if overrides:
        for tier_name, model_name in overrides.items():
            mapping[tier_name] = model_name

    # Ensure at least the three core tiers have defaults
    if "simple" not in mapping and "analytical" in mapping:
        mapping["simple"] = mapping["analytical"]
    if "complex" not in mapping and "analytical" in mapping:
        mapping["complex"] = mapping["analytical"]
    if "analytical" not in mapping:
        # Fallback: use any available endpoint
        if mapping:
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
        {k: v[:30] for k, v in mapping.items()},
    )
    return mapping


__all__ = ["create_framework_llm_client"]
