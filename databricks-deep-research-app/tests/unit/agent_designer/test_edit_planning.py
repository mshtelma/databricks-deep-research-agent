"""Unit tests for edit_planning: scope→allow-list, diff guard, topology helpers,
and the (fake-LLM) scope classifier."""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.edit_planning import (
    EditScope,
    apply_signature_delta,
    carry_over_prompts,
    classify_edit_scope,
    edit_diff_guard,
    stored_signature,
)

# --- EditScope.to_allow_list ------------------------------------------------


def test_allow_list_derives_fields_and_bounds() -> None:
    al = EditScope(
        route="surgical", levels=["tool", "node"], target_node_ids=["c0", "c1"]
    ).to_allow_list()
    assert al.node_ids == ["c0", "c1"]
    assert "tools" in al.allowed_fields
    assert al.max_added == 4 and al.max_removed == 4  # len(targets)+2


def test_allow_list_property_prompt_only_no_node_growth() -> None:
    al = EditScope(route="surgical", levels=["prompt"], target_node_ids=["s"]).to_allow_list()
    assert "system_prompt" in al.allowed_fields
    assert al.max_added == 0 and al.max_removed == 0


# --- edit_diff_guard --------------------------------------------------------


def _ast(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"root": {"id": "root", "type": "parallel", "config": {}, "children": children}}


def _agent(nid: str, prompt: str = "p") -> dict[str, Any]:
    return {"id": nid, "type": "agent", "label": nid, "config": {"system_prompt": prompt}}


def test_guard_ok_when_only_target_changed() -> None:
    before = _ast([_agent("c0"), _agent("c1")])
    after = _ast([_agent("c0", "edited"), _agent("c1")])
    al = EditScope(route="surgical", levels=["prompt"], target_node_ids=["c0"]).to_allow_list()
    report = edit_diff_guard(before, after, al)
    assert report.ok
    assert report.changed_node_ids == ["c0"]


def test_guard_flags_out_of_scope_node_change() -> None:
    before = _ast([_agent("c0"), _agent("c1")])
    after = _ast([_agent("c0", "edited"), _agent("c1", "ALSO edited")])
    al = EditScope(route="surgical", levels=["prompt"], target_node_ids=["c0"]).to_allow_list()
    report = edit_diff_guard(before, after, al)
    assert not report.ok
    assert any("c1" in v for v in report.violations)


def test_guard_flags_unauthorized_node_addition() -> None:
    before = _ast([_agent("c0")])
    after = _ast([_agent("c0"), _agent("c1")])
    # levels=[prompt] ⇒ max_added=0
    al = EditScope(route="surgical", levels=["prompt"], target_node_ids=["c0"]).to_allow_list()
    report = edit_diff_guard(before, after, al)
    assert not report.ok
    assert report.added_node_ids == ["c1"]


def test_guard_allows_authorized_node_addition() -> None:
    before = _ast([_agent("c0")])
    after = _ast([_agent("c0"), _agent("c1")])
    al = EditScope(route="surgical", levels=["node"], target_node_ids=["c0"]).to_allow_list()
    report = edit_diff_guard(before, after, al)
    assert report.ok and report.added_node_ids == ["c1"]


def test_guard_advisory_mode_when_no_targets() -> None:
    before = _ast([_agent("c0"), _agent("c1")])
    after = _ast([_agent("c0", "x"), _agent("c1", "y")])
    al = EditScope(route="surgical", levels=["node"], target_node_ids=[]).to_allow_list()
    report = edit_diff_guard(before, after, al)
    # no node_ids ⇒ no per-node scope violations
    assert report.ok
    assert set(report.changed_node_ids) == {"c0", "c1"}


# --- topology helpers -------------------------------------------------------


def test_stored_signature_present_and_legacy() -> None:
    assert stored_signature({"designer_signature": {"a": 1}}) == {"a": 1}
    assert stored_signature({}) is None
    assert stored_signature({"designer_signature": {}}) is None
    assert stored_signature("nope") is None


def test_apply_signature_delta_shallow_merge() -> None:
    sig = {"coordination_pattern": "best_of_n", "coordination_candidate_count": 6}
    out = apply_signature_delta(sig, {"coordination_pattern": None, "step_dependencies_present": True})
    assert out["coordination_pattern"] is None
    assert out["step_dependencies_present"] is True
    assert out["coordination_candidate_count"] == 6  # untouched
    assert sig["coordination_pattern"] == "best_of_n"  # input not mutated


def test_carry_over_prompts_matches_by_subtype_ordinal() -> None:
    old = {
        "root": {
            "id": "r",
            "type": "sequence",
            "children": [
                {"id": "old-syn", "type": "agent", "config": {"subtype": "synthesizer", "system_prompt": "SYN"}},
            ],
        }
    }
    new = {
        "root": {
            "id": "r2",
            "type": "sequence",
            "children": [
                {"id": "new-syn", "type": "agent", "config": {"subtype": "synthesizer"}},
                {"id": "new-coord", "type": "agent", "config": {"subtype": "coordinator"}},
            ],
        }
    }
    patches, regenerated = carry_over_prompts(old, new)
    assert patches.get("new-syn", {}).get("system_prompt") == "SYN"
    # coordinator had no old counterpart → regenerated
    assert any("coordinator" in r for r in regenerated)


# --- classify_edit_scope (fake LLM) -----------------------------------------


class _Resp:
    def __init__(self, structured: Any) -> None:
        self.structured = structured
        self.content = None


class _FakeLLM:
    def __init__(self, structured: Any) -> None:
        self._structured = structured

    async def complete(self, **_kwargs: Any) -> _Resp:
        return _Resp(self._structured)


class _RaisingLLM:
    async def complete(self, **_kwargs: Any) -> _Resp:
        raise RuntimeError("boom")


async def test_classify_returns_structured_scope() -> None:
    scope = EditScope(route="surgical", levels=["tool"], target_node_ids=["c0"], tool_names=["compute"])
    out = await classify_edit_scope(llm=_FakeLLM(scope), intent="add compute", ast_summary={"nodes": ["c0"]})
    assert out.route == "surgical" and out.tool_names == ["compute"]


async def test_classify_fallback_on_none_llm() -> None:
    out = await classify_edit_scope(llm=None, intent="x", ast_summary={})
    assert out.route == "surgical"


async def test_classify_fallback_on_llm_error() -> None:
    out = await classify_edit_scope(llm=_RaisingLLM(), intent="x", ast_summary={})
    assert out.route == "surgical"  # fail-safe: never silently rebuild
