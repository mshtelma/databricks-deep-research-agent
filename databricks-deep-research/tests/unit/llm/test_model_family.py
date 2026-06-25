"""FW-1 — per-node model-family selection on FrameworkLLMClient.

A node may pin ``config.model_family`` (orthogonal to ``model_tier``) to route to
a configured family. Family configs are stored alongside tiers (so endpoint
selection / health / rotation / 429-fallback work unchanged), validated against
``_family_keys``, and preserved across ``derive``. ``complete()`` shares the same
``_resolution_key`` guard, so the unknown-family failure mode is covered by the
``resolve_model`` tests below.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    ModelTierConfig,
    UnknownModelFamilyError,
)


def _client(**kw: object) -> FrameworkLLMClient:
    return FrameworkLLMClient(
        openai_client=MagicMock(),
        model_mapping={
            "simple": "ep-simple",
            "analytical": "ep-analytical",
            "complex": ModelTierConfig(endpoints=["ep-complex"]),
        },
        **kw,  # type: ignore[arg-type]
    )


def test_no_families_by_default() -> None:
    c = _client()
    assert c.model_families == frozenset()
    assert c.resolve_model("analytical") == "ep-analytical"
    assert c.resolve_model("complex") == "ep-complex"


def test_family_overrides_tier_for_resolution() -> None:
    c = _client(
        model_families={
            "claude": ModelTierConfig(endpoints=["ep-claude"]),
            "llama": "ep-llama",
        }
    )
    assert c.model_families == frozenset({"claude", "llama"})
    # tier routing unaffected when no family is given
    assert c.resolve_model("analytical") == "ep-analytical"
    # family overrides the tier for endpoint selection
    assert c.resolve_model("analytical", family="claude") == "ep-claude"
    assert c.resolve_model("simple", family="llama") == "ep-llama"


def test_unknown_family_fails_closed() -> None:
    c = _client(model_families={"claude": "ep-claude"})
    with pytest.raises(UnknownModelFamilyError) as ei:
        c.resolve_model("analytical", family="gpt")
    assert ei.value.family == "gpt"
    assert "claude" in ei.value.available


def test_derive_preserves_families() -> None:
    c = _client(model_families={"claude": "ep-claude"})
    d = c.derive({"simple": "ep-new-simple"})
    assert d.model_families == frozenset({"claude"})
    assert d.resolve_model("analytical", family="claude") == "ep-claude"
    assert d.resolve_model("simple") == "ep-new-simple"  # tier override applied
    # derived client has independent (fresh) health state but same family catalog
    assert d.resolve_model("complex") == "ep-complex"


def test_family_endpoint_windows_backfilled() -> None:
    c = _client(
        model_families={
            "claude": ModelTierConfig(
                endpoints=["ep-claude"],
                endpoint_context_windows={"ep-claude": 200000},
            )
        }
    )
    # family endpoint windows are registered so context-window escalation works
    assert c._endpoint_registry.get("ep-claude") == 200000


def test_caller_model_mapping_not_mutated() -> None:
    mapping: dict[str, object] = {"analytical": "ep-analytical"}
    FrameworkLLMClient(
        openai_client=MagicMock(),
        model_mapping=mapping,  # type: ignore[arg-type]
        model_families={"claude": "ep-claude"},
    )
    # family merge must not leak into the caller's mapping dict
    assert "claude" not in mapping
