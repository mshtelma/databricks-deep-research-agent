"""Phase 4.1 tests — workflow migrator detects saved agents missing the
per-researcher user_prompt_template, regardless of topology."""
from __future__ import annotations

from deep_research.agent_designer.workflow_migrator import (
    MigratableResearcher,
    scan_workflow_for_migration,
)


_VALID_TEMPLATE = (
    "## Investigation Brief\n\n"
    "You are investigating: **{query}**\n\n"
    "### Sub-questions you MUST address\n"
    "1. Q1?\n2. Q2?\n3. Q3?\n4. Q4?\n5. Q5?\n\n"
    "### Required output structure\n- A\n- B\n- C\n\n"
    "### Search strategy\n- query strategy.\n- primary sources.\n\n"
    "### Definition of done\nData unavailable when not found."
)

_GENERIC_DEFAULT = (
    "Execute the following research step:\n\n"
    "## Step Details\nTitle: {step_title}\nDescription: {step_description}"
)


def _agent(
    *,
    node_id: str,
    subtype: str,
    user_prompt_template: str | None = None,
) -> dict:
    config: dict = {"subtype": subtype}
    if user_prompt_template is not None:
        config["user_prompt_template"] = user_prompt_template
    return {
        "id": node_id,
        "type": "agent",
        "label": node_id,
        "config": config,
        "children": [],
    }


def _wrap(*children: dict) -> dict:
    return {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": list(children),
        }
    }


# ---------------------------------------------------------------------------
# Single-agent topology
# ---------------------------------------------------------------------------


def test_single_agent_with_missing_template_flagged() -> None:
    workflow = _wrap(_agent(node_id="answer", subtype="researcher"))
    report = scan_workflow_for_migration(workflow)
    assert report.needs_regeneration
    assert len(report.researchers) == 1
    assert report.researchers[0].node_id == "answer"
    assert report.researchers[0].reason == "missing"


def test_single_agent_with_generic_default_flagged() -> None:
    workflow = _wrap(
        _agent(node_id="answer", subtype="researcher", user_prompt_template=_GENERIC_DEFAULT)
    )
    report = scan_workflow_for_migration(workflow)
    assert report.needs_regeneration
    assert report.researchers[0].reason == "generic_default"


def test_single_agent_with_valid_template_passes() -> None:
    workflow = _wrap(
        _agent(node_id="answer", subtype="researcher", user_prompt_template=_VALID_TEMPLATE)
    )
    report = scan_workflow_for_migration(workflow)
    assert not report.needs_regeneration
    assert report.researchers == []


# ---------------------------------------------------------------------------
# Parallel-lanes topology — every lane is checked
# ---------------------------------------------------------------------------


def test_parallel_lanes_each_missing_template_flagged() -> None:
    parallel = {
        "id": "parallel-lanes",
        "type": "parallel",
        "label": "Parallel",
        "config": {},
        "children": [
            _agent(node_id="lane_1-researcher", subtype="researcher"),
            _agent(node_id="lane_2-researcher", subtype="researcher", user_prompt_template=_GENERIC_DEFAULT),
            _agent(node_id="lane_3-researcher", subtype="researcher", user_prompt_template=_VALID_TEMPLATE),
        ],
    }
    workflow = _wrap(parallel)
    report = scan_workflow_for_migration(workflow)
    assert report.needs_regeneration
    ids = {r.node_id for r in report.researchers}
    assert ids == {"lane_1-researcher", "lane_2-researcher"}


# ---------------------------------------------------------------------------
# Plan_and_execute topology — body researcher gets checked too
# ---------------------------------------------------------------------------


def test_plan_and_execute_body_researcher_checked() -> None:
    body_researcher = _agent(node_id="step-researcher", subtype="researcher")
    workflow = {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": [
                {
                    "id": "p_and_e",
                    "type": "plan_and_execute",
                    "label": "P&E",
                    "config": {
                        "planner": {
                            "subtype": "planner",
                            "system_prompt": "Planner.",
                        },
                        "body": body_researcher,
                    },
                    "children": [],
                }
            ],
        }
    }
    report = scan_workflow_for_migration(workflow)
    assert report.needs_regeneration
    assert any(r.node_id == "step-researcher" for r in report.researchers)


# ---------------------------------------------------------------------------
# Non-researcher subtypes ignored
# ---------------------------------------------------------------------------


def test_non_researcher_agents_ignored() -> None:
    workflow = _wrap(
        _agent(node_id="coordinator", subtype="coordinator"),
        _agent(node_id="synthesizer", subtype="synthesizer"),
        _agent(node_id="reflector", subtype="reflector"),
    )
    report = scan_workflow_for_migration(workflow)
    assert not report.needs_regeneration
    assert report.researchers == []


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_includes_node_path_for_each_violation() -> None:
    workflow = _wrap(
        _agent(node_id="answer", subtype="researcher"),
    )
    report = scan_workflow_for_migration(workflow)
    [item] = report.researchers
    assert isinstance(item, MigratableResearcher)
    assert item.path.startswith("root")
