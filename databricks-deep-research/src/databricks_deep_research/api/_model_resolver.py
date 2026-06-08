"""Model spec resolver: convert Python ``model`` arguments to a tier name.

The Python API accepts ``model`` as one of:

- ``str``: either a :class:`ModelTier` literal (``"simple"``, ``"analytical"``,
  ``"complex"``) or an explicit endpoint name. Endpoint names are passed
  through unchanged via the workflow's ``models:`` section at compile time.
- :class:`ModelTier`: passed through directly.
- :class:`FrameworkLLMClient`: when an explicit client is provided, the API
  layer uses it to construct a runner; the workflow is compiled with the
  default tier name (``"analytical"``).

This module returns a tier name string suitable for ``AgentNodeConfig.model_tier``.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

DEFAULT_TIER: str = ModelTier.analytical.value


def resolve_tier_name(spec: Any | None) -> str:
    """Return a tier name for ``AgentNodeConfig.model_tier``.

    Args:
        spec: One of ``None``, a :class:`ModelTier`, a string (tier or
            endpoint name), or a :class:`FrameworkLLMClient`.

    Returns:
        Tier name string. Defaults to ``"analytical"`` when *spec* is ``None``
        or a :class:`FrameworkLLMClient` (the actual client is supplied at
        runtime via the runner).
    """
    if spec is None:
        return DEFAULT_TIER
    if isinstance(spec, ModelTier):
        return spec.value
    if isinstance(spec, FrameworkLLMClient):
        return DEFAULT_TIER
    if isinstance(spec, str):
        # If it is exactly a tier name, use it; otherwise, treat as an
        # endpoint name and fall back to analytical (the workflow's models:
        # section can override).
        if spec in {ModelTier.simple.value, ModelTier.analytical.value, ModelTier.complex.value}:
            return spec
        return DEFAULT_TIER
    return DEFAULT_TIER


__all__ = ["DEFAULT_TIER", "resolve_tier_name"]
