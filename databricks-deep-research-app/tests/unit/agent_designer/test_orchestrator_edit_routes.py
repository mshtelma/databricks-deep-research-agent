"""Behavioral tests for the orchestrator's edit lane: route helpers, the
topology handler, and the single-net-event finalize."""

from __future__ import annotations

import copy
from typing import Any

from deep_research.agent_designer import mutations
from deep_research.agent_designer.blueprint import build_blueprint
from deep_research.agent_designer.edit_planning import EditScope
from deep_research.agent_designer.orchestrator import (
    DesignerChatOrchestrator,
    _compact_ast_summary,
    _is_meaningful_ast,
)


def _best_of_n_ast(count: int = 6) -> dict[str, Any]:
    sig = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "comparative_analysis",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 1,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["evidence"],
        "coordination_pattern": "best_of_n",
        "coordination_candidate_count": count,
    }
    return build_blueprint(sig, "compare options", [])


def _orch() -> DesignerChatOrchestrator:
    # _run_topology_edit / _finalize_edit use neither llm nor discovery.
    return DesignerChatOrchestrator(llm=None, discovery=None)  # type: ignore[arg-type]


async def _collect(agen: Any) -> list[Any]:
    return [e async for e in agen]


def _types(events: list[Any]) -> list[str]:
    return [type(e).__name__ for e in events]


# --- route helpers ----------------------------------------------------------


def test_is_meaningful_ast() -> None:
    assert _is_meaningful_ast(_best_of_n_ast())
    assert not _is_meaningful_ast({})
    assert not _is_meaningful_ast({"root": {"type": "sequence"}})  # no children
    assert _is_meaningful_ast({"root": {"type": "agent", "id": "a"}})  # lone agent


def test_compact_ast_summary_is_prompt_safe() -> None:
    rows = _compact_ast_summary(_best_of_n_ast())
    assert rows and all("id" in r and "type" in r for r in rows)
    # no prompt text leaks into the summary
    blob = str(rows)
    assert "system_prompt" not in blob and "user_prompt_template" not in blob


# --- topology handler -------------------------------------------------------


async def test_topology_edit_emits_single_mutation_with_updated_signature() -> None:
    current = _best_of_n_ast(6)
    scope = EditScope(route="topology", delta={"coordination_candidate_count": 4},
                      change_summary="Reduce to 4 candidates")
    events = await _collect(
        _orch()._run_topology_edit(
            current_ast=current,
            user_intent="make it best of 4",
            edit_scope=scope,
            normalized_assets=[],
            resolved_tool_contract=None,
        )
    )
    names = _types(events)
    assert "MutationProposedEvent" in names
    assert names[-1] == "DoneEvent"
    mut = next(e for e in events if type(e).__name__ == "MutationProposedEvent")
    # the rebuilt AST carries the UPDATED persisted signature
    assert mut.new_ast["designer_signature"]["coordination_candidate_count"] == 4


async def test_topology_edit_legacy_ast_is_never_silent() -> None:
    legacy = _best_of_n_ast(6)
    legacy.pop("designer_signature", None)  # simulate a pre-persistence workflow
    events = await _collect(
        _orch()._run_topology_edit(
            current_ast=legacy,
            user_intent="switch topology",
            edit_scope=EditScope(route="topology", delta={}),
            normalized_assets=[],
            resolved_tool_contract=None,
        )
    )
    names = _types(events)
    assert "MutationProposedEvent" not in names  # no lossy guess
    assert "MessageEvent" in names and names[-1] == "DoneEvent"


# --- finalize ---------------------------------------------------------------


async def test_finalize_edit_emits_one_mutation_for_a_real_change() -> None:
    current = _best_of_n_ast(6)
    final = mutations.declare_tool(copy.deepcopy(current), "compute", "compute", {})
    scope = EditScope(route="surgical", levels=["tool"], change_summary="Add compute tool")
    events = await _collect(
        _orch()._finalize_edit(
            current_ast=current, final_ast=final, edit_scope=scope
        )
    )
    names = _types(events)
    assert names.count("MutationProposedEvent") == 1  # exactly one net event
    assert "MessageEvent" in names  # human summary


async def test_finalize_edit_no_change_is_never_silent_without_mutation() -> None:
    current = _best_of_n_ast(6)
    events = await _collect(
        _orch()._finalize_edit(
            current_ast=current,
            final_ast=copy.deepcopy(current),  # unchanged
            edit_scope=EditScope(route="surgical", change_summary="noop"),
        )
    )
    names = _types(events)
    assert "MutationProposedEvent" not in names
    assert names == ["MessageEvent"]
