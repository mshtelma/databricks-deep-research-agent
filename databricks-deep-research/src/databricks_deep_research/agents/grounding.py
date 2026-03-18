"""Shared helpers for synthesizer grounding-mode resolution and validation."""

from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig


def legacy_grounding_mode(config: AgentNodeConfig) -> str | None:
    """Infer grounding mode from legacy output_schema fields when present."""
    schema = config.output_schema or {}
    if schema.get("synthesis_mode") == "reclaim":
        return "reclaim"
    if schema.get("enable_citation_verification") is True:
        return "reclaim"
    if schema:
        return None
    return None


def resolve_grounding_mode(config: AgentNodeConfig) -> str:
    """Resolve the effective grounding mode for a synthesizer node."""
    if config.grounding_mode in {"none", "classical_lite", "reclaim"}:
        return str(config.grounding_mode)
    legacy_mode = legacy_grounding_mode(config)
    if legacy_mode in {"classical_lite", "reclaim"}:
        return legacy_mode
    return "none"


def validate_grounding_config(config: AgentNodeConfig) -> list[str]:
    """Return validation errors for conflicting synthesizer grounding settings."""
    if config.subtype != "synthesizer":
        return []

    explicit = config.grounding_mode
    legacy = legacy_grounding_mode(config)
    if explicit is None or legacy is None:
        return []
    if explicit == legacy:
        return []
    return [
        "Synthesizer config sets conflicting grounding controls: "
        f"grounding_mode={explicit!r} conflicts with legacy output_schema mode={legacy!r}"
    ]


def uses_legacy_grounding_alias(config: AgentNodeConfig) -> bool:
    """Whether a synthesizer relies on legacy output_schema grounding fields."""
    return config.subtype == "synthesizer" and config.grounding_mode is None and legacy_grounding_mode(config) is not None

