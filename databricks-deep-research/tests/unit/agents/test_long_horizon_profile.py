"""Tests for the ``long_horizon`` AgentNodeConfig profile preset (spec §2.4)."""

from __future__ import annotations

from databricks_deep_research.agents.config import (
    _LONG_HORIZON_MAX_TOOL_CALLS,
    AgentNodeConfig,
)


def test_long_horizon_profile_activates_knobs() -> None:
    """The preset turns on offload + a large tool budget; rescue is already on."""
    cfg = AgentNodeConfig(subtype="researcher", profile="long_horizon")
    assert cfg.tool_output_offload == "auto"
    assert cfg.max_tool_calls == _LONG_HORIZON_MAX_TOOL_CALLS
    assert cfg.evidence_rescue is True


def test_long_horizon_profile_respects_explicit_overrides() -> None:
    """An explicitly-set field always wins over the preset default."""
    cfg = AgentNodeConfig(
        subtype="researcher",
        profile="long_horizon",
        tool_output_offload="off",
        max_tool_calls=7,
    )
    assert cfg.tool_output_offload == "off"
    assert cfg.max_tool_calls == 7


def test_default_profile_is_a_no_op() -> None:
    """``profile='default'`` (the default) leaves every knob untouched."""
    cfg = AgentNodeConfig(subtype="researcher")
    assert cfg.profile == "default"
    assert cfg.tool_output_offload == "off"
    assert cfg.max_tool_calls is None
