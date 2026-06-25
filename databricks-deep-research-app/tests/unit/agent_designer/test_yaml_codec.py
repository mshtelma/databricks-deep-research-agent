"""Pure-function unit tests for the agent-designer YAML codec.

These run in CI via ``make test-app`` (``pytest tests/unit``) — NO database, no
app import, no credentials.  They exist because the integration round-trip
tests are gated behind ``RUN_INTEGRATION_TESTS=1`` + a live DB and therefore do
NOT run in CI, which let the export/import ``registry_version`` mismatch
("1.0" vs "1.0.0") ship undetected.

Invariants asserted (topology-agnostic — every workflow shape, not one example):
  1. ``serialize_to_yaml`` defaults to the CURRENT ``REGISTRY_VERSION``.
  2. ``parse_and_validate_yaml(serialize_to_yaml(D)) == load(D).model_dump()``
     for sequence / parallel / nested / tool-bearing definitions.
  3. A freshly-exported document re-imports without raising (the regression).
  4. Serialisation is deterministic.
  5. No secret-like keys leak into exported YAML.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.registry import REGISTRY_VERSION
from deep_research.agent_designer.yaml_export import serialize_to_yaml
from deep_research.agent_designer.yaml_import import (
    YamlImportError,
    parse_and_validate_yaml,
)

# ---------------------------------------------------------------------------
# Topology fixtures — each is a minimal, structurally-valid WorkflowDefinition.
# Kept generic on purpose: no domain vocabulary, no benchmark coupling.
# ---------------------------------------------------------------------------


def _base(wf_id: str, root: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    defn: dict[str, Any] = {
        "id": wf_id,
        "name": f"Codec Test {wf_id}",
        "version": 1,
        "root": root,
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
    }
    defn.update(overrides)
    return defn


def _agent(node_id: str, subtype: str, label: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "agent",
        "label": label,
        "config": {"subtype": subtype},
        "children": [],
    }


_SEQUENCE = _base(
    "seq-wf",
    {
        "id": "root-seq",
        "type": "sequence",
        "label": "main",
        "config": {},
        "children": [_agent("a1", "researcher", "researcher")],
    },
)

_PARALLEL = _base(
    "par-wf",
    {
        "id": "root-par",
        "type": "parallel",
        "label": "fan-out",
        "config": {},
        "children": [
            _agent("a1", "researcher", "researcher-a"),
            _agent("a2", "researcher", "researcher-b"),
        ],
    },
)

_NESTED = _base(
    "nested-wf",
    {
        "id": "outer",
        "type": "sequence",
        "label": "outer",
        "config": {},
        "children": [
            {
                "id": "inner",
                "type": "sequence",
                "label": "inner",
                "config": {},
                "children": [_agent("a1", "synthesizer", "synth")],
            }
        ],
    },
)

# Tool-bearing: a declared tool referenced by an agent. Exercises the
# top-level ``tools:`` round-trip and feeds the no-secrets invariant.
_WITH_TOOLS = _base(
    "tools-wf",
    {
        "id": "root-seq",
        "type": "sequence",
        "label": "main",
        "config": {},
        "children": [
            {
                "id": "a1",
                "type": "agent",
                "label": "researcher",
                "config": {"subtype": "researcher", "tools": ["web"]},
                "children": [],
            }
        ],
    },
    tools=[
        {
            "name": "web",
            "kind": "web_search",
            "config": {"max_results": 5},
            "description": "Generic web search",
        }
    ],
)

_TOPOLOGIES = pytest.mark.parametrize(
    "definition",
    [_SEQUENCE, _PARALLEL, _NESTED, _WITH_TOOLS],
    ids=["sequence", "parallel", "nested", "with_tools"],
)


# ---------------------------------------------------------------------------
# 1. Version default — single source of truth
# ---------------------------------------------------------------------------


def test_default_registry_version_matches_constant() -> None:
    """serialize_to_yaml stamps the SAME constant the importer enforces."""
    text = serialize_to_yaml(_SEQUENCE)
    parsed = yaml.safe_load(text)
    assert parsed["registry_version"] == REGISTRY_VERSION


# ---------------------------------------------------------------------------
# 2. Round-trip fidelity across topologies
# ---------------------------------------------------------------------------


@_TOPOLOGIES
def test_round_trip_preserves_definition(definition: dict[str, Any]) -> None:
    """parse_and_validate_yaml(serialize_to_yaml(D)) == load(D).model_dump()."""
    expected = load_workflow_from_dict(definition).model_dump()
    round_tripped = parse_and_validate_yaml(serialize_to_yaml(definition).encode("utf-8"))
    assert round_tripped == expected


# ---------------------------------------------------------------------------
# 3. The regression: exported bytes re-import without raising
# ---------------------------------------------------------------------------


@_TOPOLOGIES
def test_exported_yaml_reimports_without_raising(definition: dict[str, Any]) -> None:
    """Before the version fix this raised YamlImportError(registry_version_mismatch)."""
    body = serialize_to_yaml(definition).encode("utf-8")
    try:
        parse_and_validate_yaml(body)
    except YamlImportError as exc:  # pragma: no cover - failure path
        pytest.fail(f"exported YAML failed to re-import: {exc.error_kind}: {exc.message}")


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_serialize_is_deterministic() -> None:
    assert serialize_to_yaml(_WITH_TOOLS) == serialize_to_yaml(_WITH_TOOLS)


# ---------------------------------------------------------------------------
# 5. No secret-like keys leak into exported YAML
# ---------------------------------------------------------------------------


@_TOPOLOGIES
def test_no_secret_keys_in_export(definition: dict[str, Any]) -> None:
    """The AST schema carries endpoint names/config, never credentials.

    Documents the export privacy invariant: serialisation is pass-through, so
    this guards against a future field (or fixture) smuggling a secret-shaped
    key into the exported document.
    """
    text = serialize_to_yaml(definition).lower()
    for needle in ("password", "secret", "api_key", "access_token", "client_secret"):
        assert needle not in text, f"unexpected secret-like key {needle!r} in export"


# ---------------------------------------------------------------------------
# 6. registry_version handling on import
# ---------------------------------------------------------------------------
# Contract (parse_and_validate_yaml):
#   • absent / null      → accept, treat as the current registry version, so raw
#                          framework YAML and legacy pre-envelope exports import.
#   • present & equal    → accept.
#   • present & different → reject with an actionable registry_version_mismatch.


def test_missing_registry_version_is_accepted() -> None:
    """Raw framework YAML (no ``registry_version`` envelope) imports cleanly."""
    assert "registry_version" not in _SEQUENCE  # fixture really omits the key
    body = yaml.safe_dump(_SEQUENCE, sort_keys=True).encode("utf-8")
    result = parse_and_validate_yaml(body)
    assert result["id"] == _SEQUENCE["id"]
    # The envelope key is stripped and never echoed back into the definition.
    assert "registry_version" not in result


def test_null_registry_version_is_accepted() -> None:
    """An explicit ``registry_version: null`` is treated the same as absent."""
    body = yaml.safe_dump(
        {"registry_version": None, **_SEQUENCE}, sort_keys=True
    ).encode("utf-8")
    result = parse_and_validate_yaml(body)
    assert result["id"] == _SEQUENCE["id"]


def test_matching_registry_version_is_accepted() -> None:
    """The current registry version is accepted (export round-trip)."""
    body = yaml.safe_dump(
        {"registry_version": REGISTRY_VERSION, **_SEQUENCE}, sort_keys=True
    ).encode("utf-8")
    result = parse_and_validate_yaml(body)
    assert result["id"] == _SEQUENCE["id"]


def test_mismatched_registry_version_is_rejected_with_guidance() -> None:
    """A present-but-different ``registry_version`` is rejected with guidance."""
    body = yaml.safe_dump(
        {"registry_version": "0.0.0", **_SEQUENCE}, sort_keys=True
    ).encode("utf-8")
    with pytest.raises(YamlImportError) as excinfo:
        parse_and_validate_yaml(body)
    exc = excinfo.value
    assert exc.error_kind == "registry_version_mismatch"
    # Both versions are named, plus the actionable raw-import hint.
    assert REGISTRY_VERSION in exc.message
    assert "0.0.0" in exc.message
    assert "remove the registry_version line" in exc.message
