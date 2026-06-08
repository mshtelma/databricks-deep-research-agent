"""Phase 1 — framework runtime-injected key registry.

These keys exist in workflow STATE / template context at runtime without any node
``output_key`` producing them; the dataflow checker seeds Pass A with them so it
does not false-positive on harness/runtime-supplied template variables.
"""
from __future__ import annotations

from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS


def test_registry_includes_known_runtime_keys() -> None:
    for k in (
        "all_observations",
        "plan_summary",
        "step_title",
        "source_quality",
        "reflector_feedback",
        "current_date",
        "tool_catalog",
        "claims",  # from state._RUNTIME_DERIVED_KEYS
        "verification_summary",
    ):
        assert k in RUNTIME_INJECTED_KEYS


def test_current_step_is_not_a_global_seed() -> None:
    # current_step is injected ONLY inside plan_and_execute (PAE-scoped); it must
    # NOT be a global runtime key, or a current_step read outside a PAE would be
    # masked instead of flagged dangling.
    assert "current_step" not in RUNTIME_INJECTED_KEYS


def test_query_is_seeded() -> None:
    assert "query" in RUNTIME_INJECTED_KEYS
