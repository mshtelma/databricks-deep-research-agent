"""Tests for read_skill auto-attach (A1 — executor wiring helper)."""

from __future__ import annotations

from types import SimpleNamespace

from databricks_deep_research.agents.skill_attach import (
    maybe_attach_read_skill,
    maybe_attach_run_skill_script,
)
from databricks_deep_research.skills import FilesystemSkillStore
from databricks_deep_research.tools.builtins.read_skill import ReadSkillTool
from databricks_deep_research.tools.builtins.run_skill_script import RunSkillScriptTool
from databricks_deep_research.tools.protocol import ResearchTool


def _ctx_with_store() -> SimpleNamespace:
    return SimpleNamespace(extras={"_skill_store": FilesystemSkillStore()})


def _ctx_scripts_enabled() -> SimpleNamespace:
    return SimpleNamespace(
        extras={"_skill_store": FilesystemSkillStore(), "_skill_scripts_enabled": True}
    )


def test_attaches_when_skills_and_store() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_read_skill(tools, ["a"], _ctx_with_store()) is True
    assert any(isinstance(t, ReadSkillTool) for t in tools)


def test_noop_without_skills() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_read_skill(tools, [], _ctx_with_store()) is False
    assert tools == []


def test_noop_without_store() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_read_skill(tools, ["a"], SimpleNamespace(extras={})) is False
    assert maybe_attach_read_skill(tools, ["a"], None) is False
    assert tools == []


def test_noop_when_already_present() -> None:
    tools: list[ResearchTool] = [ReadSkillTool(skill_store=FilesystemSkillStore())]
    assert maybe_attach_read_skill(tools, ["a"], _ctx_with_store()) is False
    assert len(tools) == 1


# -- run_skill_script auto-attach gating (A2) --------------------------------


def test_script_attaches_when_both_switches_on() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_run_skill_script(tools, ["a"], True, _ctx_scripts_enabled()) is True
    assert any(isinstance(t, RunSkillScriptTool) for t in tools)


def test_script_noop_when_per_agent_off() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_run_skill_script(tools, ["a"], False, _ctx_scripts_enabled()) is False
    assert tools == []


def test_script_noop_when_global_off() -> None:
    # Per-agent flag on, but global kill-switch absent from extras.
    tools: list[ResearchTool] = []
    assert maybe_attach_run_skill_script(tools, ["a"], True, _ctx_with_store()) is False
    assert tools == []


def test_script_noop_without_skills() -> None:
    tools: list[ResearchTool] = []
    assert maybe_attach_run_skill_script(tools, [], True, _ctx_scripts_enabled()) is False
    assert tools == []


def test_script_noop_without_store() -> None:
    tools: list[ResearchTool] = []
    ctx = SimpleNamespace(extras={"_skill_scripts_enabled": True})
    assert maybe_attach_run_skill_script(tools, ["a"], True, ctx) is False
    assert tools == []


def test_script_noop_when_already_present() -> None:
    tools: list[ResearchTool] = [
        RunSkillScriptTool(skill_store=FilesystemSkillStore(), enabled=True)
    ]
    assert maybe_attach_run_skill_script(tools, ["a"], True, _ctx_scripts_enabled()) is False
    assert len(tools) == 1
