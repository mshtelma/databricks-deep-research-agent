"""YAML serialisation for agent WorkflowDefinition ASTs.

The only public symbol is :func:`serialize_to_yaml`, which wraps a raw
definition dict with a ``registry_version`` key and produces a deterministic,
human-readable YAML string suitable for export and round-trip loading.
"""

from __future__ import annotations

import yaml

from deep_research.agent_designer.registry import REGISTRY_VERSION


def serialize_to_yaml(
    definition: dict[str, object], registry_version: str = REGISTRY_VERSION
) -> str:
    """Convert a WorkflowDefinition AST dict to a deterministic YAML string.

    The ``registry_version`` key is prepended so that consumers can detect
    forward-incompatible schema changes without parsing the full AST.

    The default is pinned to :data:`~deep_research.agent_designer.registry.REGISTRY_VERSION`
    — the SAME constant :func:`~deep_research.agent_designer.yaml_import.parse_and_validate_yaml`
    checks on import.  This is the single source of truth that guarantees an
    exported document re-imports cleanly (the two must never drift, or every
    round-trip fails with ``registry_version_mismatch``).

    Args:
        definition: Raw AST dict (as stored in ``AgentV2.definition``).
            Only the AST fields are serialised — no credentials, tokens, or
            internal service state are included.
        registry_version: Semantic version string pinned at the top of the
            output document.  Defaults to the current ``REGISTRY_VERSION``.

    Returns:
        A UTF-8 safe, sort-key-stable YAML string with 2-space indentation.
    """
    wrapped: dict[str, object] = {"registry_version": registry_version, **definition}
    return yaml.safe_dump(
        wrapped,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
    )
