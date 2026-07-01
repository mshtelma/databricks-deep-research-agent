"""Unit tests for the deterministic architect-synopsis derivation.

``_build_architect_synopsis`` turns a final workflow AST into the readable
"What I built" payload. It must be pure, deterministic, and topology-agnostic
(no hardcoded topology names or domain vocabulary) — these tests pin that.
"""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.orchestrator import _build_architect_synopsis
from deep_research.agent_designer.sse_events import ArchitectSynopsisEvent


def _parallel_lanes_ast() -> dict[str, Any]:
    return {
        "id": "wf1",
        "root": {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "coordinator",
                    "type": "agent",
                    "label": "Coordinator",
                    "config": {"subtype": "coordinator"},
                    "children": [],
                },
                {
                    "id": "lanes",
                    "type": "parallel",
                    "children": [
                        {
                            "id": "l1",
                            "type": "agent",
                            "label": "Market Sizing Researcher",
                            "config": {"subtype": "researcher", "tools": ["web_search"]},
                            "children": [],
                        },
                        {
                            "id": "l2",
                            "type": "agent",
                            "label": "Competitor Researcher",
                            "config": {
                                "subtype": "researcher",
                                "tools": ["web_search", "vector_search"],
                            },
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "synth",
                    "type": "agent",
                    "label": "Report Synthesizer",
                    "config": {
                        "subtype": "synthesizer",
                        "synthesis_metadata": {
                            "designer_required_outputs": "Executive summary\nSource list"
                        },
                    },
                    "children": [],
                },
            ],
        },
        "tools": [
            {"name": "web_search", "kind": "web_search", "config": {}},
            {"name": "vector_search", "kind": "vector_index", "config": {}},
        ],
        "pools": [],
        "placeholder_pending_nodes": ["l2"],
    }


def test_synopsis_splits_lanes_from_pipeline_and_lists_tools() -> None:
    event = _build_architect_synopsis(_parallel_lanes_ast(), change_kind="created")
    assert isinstance(event, ArchitectSynopsisEvent)
    assert event.change_kind == "created"
    assert event.headline.startswith("Built")
    assert event.topology and event.topology != "unknown"

    lane_labels = {lane["label"] for lane in event.lanes}
    assert "Market Sizing Researcher" in lane_labels
    assert "Competitor Researcher" in lane_labels
    # Coordinator + synthesizer are pipeline stages, not evidence lanes.
    assert "Coordinator" not in lane_labels
    assert "Report Synthesizer" not in lane_labels
    assert "Coordinator" in event.pipeline
    assert "Report Synthesizer" in event.pipeline

    competitor = next(lane for lane in event.lanes if lane["label"] == "Competitor Researcher")
    assert competitor["tools"] == ["web_search", "vector_search"]
    assert "web_search" in event.tools
    assert "vector_search" in event.tools


def test_synopsis_surfaces_placeholder_warning_and_outputs() -> None:
    event = _build_architect_synopsis(_parallel_lanes_ast(), change_kind="created")
    assert any("default prompt" in w for w in event.warnings)
    assert "Executive summary" in event.outputs
    assert "Source list" in event.outputs


def test_edited_change_kind_uses_updated_headline() -> None:
    event = _build_architect_synopsis(_parallel_lanes_ast(), change_kind="edited")
    assert event.change_kind == "edited"
    assert event.headline.startswith("Updated")


def test_synopsis_is_topology_agnostic_single_agent() -> None:
    # A single research agent (no coordinator/synth stages) must still produce a
    # sensible synopsis — the helper never assumes a particular topology shape.
    ast: dict[str, Any] = {
        "id": "wf2",
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Lookup Agent",
            "config": {"subtype": "researcher", "tools": ["web_search"]},
            "children": [],
        },
        "tools": [{"name": "web_search", "kind": "web_search", "config": {}}],
    }
    event = _build_architect_synopsis(ast, change_kind="created")
    assert isinstance(event, ArchitectSynopsisEvent)
    assert any(lane["label"] == "Lookup Agent" for lane in event.lanes)


def test_synopsis_does_not_raise_on_empty_ast() -> None:
    # Best-effort: a degenerate AST yields a sparse synopsis, never an exception
    # (the orchestrator emits the synopsis only when an AST exists, but the
    # derivation itself must stay safe).
    event = _build_architect_synopsis({}, change_kind="created")
    assert event.topology == "unknown"
    assert event.lanes == []
    assert event.pipeline == []


def test_unlabeled_agent_falls_back_to_humanized_subtype() -> None:
    ast: dict[str, Any] = {
        "id": "wf3",
        "root": {
            "id": "root",
            "type": "agent",
            "label": "",
            "config": {"subtype": "researcher", "tools": []},
            "children": [],
        },
        "tools": [],
    }
    event = _build_architect_synopsis(ast, change_kind="created")
    assert any(lane["label"] == "Researcher" for lane in event.lanes)
