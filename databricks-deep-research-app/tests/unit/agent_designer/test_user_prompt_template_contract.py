"""Tests for the per-researcher user_prompt_template wiring and structural
contract — the prevention-first fix for lane researchers emitting planning
text into findings.

Spans Phase 1 (schema + wiring) and Phase 2 (validator). Topology-agnostic
by construction — the validator inspects any researcher node regardless of
where it sits in the graph.
"""
from __future__ import annotations

import pytest

from deep_research.agent_designer.designer_architect import (
    LaneSpec,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.semantic_validation import (
    detect_unspecialized_agents,
)
from deep_research.agent_designer.workflow_builder import (
    _lane_extra_config,
    _lane_specs,
    _with_lane_user_prompt_contract,
    build_web_research_workflow,
)

# ---------------------------------------------------------------------------
# Phase 1: LaneSpec carries user_prompt_template, wired through to extra_config
# ---------------------------------------------------------------------------


def test_lane_spec_accepts_user_prompt_template() -> None:
    spec = LaneSpec(
        description="Risk analysis lane",
        system_prompt="You are the risk analysis researcher.",
        user_prompt_template="## Investigation Brief\n\nYou are investigating: **{query}**",
    )
    assert spec.user_prompt_template.startswith("## Investigation Brief")


def test_lane_spec_normalizes_none_user_prompt_template() -> None:
    spec = LaneSpec(description="Lane", system_prompt="", user_prompt_template=None)  # type: ignore[arg-type]
    assert spec.user_prompt_template == ""


def test_workflow_design_brief_preserves_user_prompt_template() -> None:
    brief = WorkflowDesignBrief(
        research_lanes=[
            {
                "description": "Lane A",
                "system_prompt": "You are the A researcher.",
                "user_prompt_template": "## Brief A\n\n{query}",
            },
        ],
    )
    assert len(brief.research_lanes) == 1
    assert brief.research_lanes[0].user_prompt_template == "## Brief A\n\n{query}"


def test_lane_specs_propagate_user_prompt_template() -> None:
    brief = WorkflowDesignBrief(
        research_lanes=[
            LaneSpec(
                description="Lane B",
                system_prompt="",
                user_prompt_template="## Brief B\n\n{query}",
            ),
        ],
    )
    specs = _lane_specs(brief)
    # Newlines preserved end-to-end (LaneSpec → compile → _lane_specs).
    assert "## Brief B" in specs[0]["user_prompt_template"]
    assert "{query}" in specs[0]["user_prompt_template"]
    assert "\n" in specs[0]["user_prompt_template"]
    assert "Designer-authored lane brief" in specs[0]["user_prompt_template"]


def test_lane_user_prompt_contract_wraps_designer_content() -> None:
    template = "Focus on concrete evidence for the requested workstream."
    rendered = _with_lane_user_prompt_contract(
        description="Supplier risk and operational resilience",
        designer_template=template,
    )

    assert rendered.startswith("## Investigation Brief")
    assert "You are investigating: **{query}**" in rendered
    assert "### Sub-questions you MUST address" in rendered
    assert rendered.count("?") >= 5
    assert "### Required output structure" in rendered
    assert "### Search strategy" in rendered
    assert "Data unavailable" in rendered
    assert template in rendered


def test_lane_extra_config_includes_user_prompt_template_when_present() -> None:
    spec = {
        "id": "lane_1",
        "description": "x",
        "system_prompt": "",
        "user_prompt_template": "## Brief\n\n{query}",
    }
    extra = _lane_extra_config(system_prompt="SYS", spec=spec)
    assert extra["system_prompt"] == "SYS"
    assert extra["user_prompt_template"] == "## Brief\n\n{query}"


def test_lane_extra_config_omits_user_prompt_template_when_empty() -> None:
    spec = {"id": "lane_1", "description": "x", "system_prompt": "", "user_prompt_template": ""}
    extra = _lane_extra_config(system_prompt="SYS", spec=spec)
    assert "user_prompt_template" not in extra


def test_lane_extra_config_handles_none_spec() -> None:
    # single_agent path when no lanes were specified.
    extra = _lane_extra_config(system_prompt="SYS", spec=None)
    assert extra == {"system_prompt": "SYS"}


def test_built_workflow_carries_lane_user_prompt_template_to_agent_config() -> None:
    """End-to-end Phase 1 wiring: a brief with user_prompt_template on a
    LaneSpec produces a lane researcher agent node whose config carries it."""
    template = (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        "### Sub-questions you MUST address (in this order)\n"
        "1. Q1?\n2. Q2?\n3. Q3?\n4. Q4?\n5. Q5?\n\n"
        "### Required output structure\n"
        "- **A**: guidance.\n- **B**: guidance.\n- **C**: guidance.\n\n"
        "### Search strategy\n- One query per sub-question.\n- Primary sources.\n\n"
        "### Definition of done\nData unavailable when not found."
    )
    brief = WorkflowDesignBrief(
        topology="parallel_lanes",
        research_lanes=[
            LaneSpec(
                description="Lane 1",
                system_prompt="Lane 1 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
            LaneSpec(
                description="Lane 2",
                system_prompt="Lane 2 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
        ],
    )
    workflow = build_web_research_workflow("test", "Test", brief)

    # Walk the workflow tree, find lane researcher agent nodes.
    researcher_templates: list[str] = []
    researcher_inputs: list[list[str]] = []
    node_types: list[str] = []

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        node_types.append(str(node.get("type") or ""))
        if node.get("type") == "agent":
            cfg = node.get("config") or {}
            if cfg.get("subtype") == "researcher":
                researcher_templates.append(cfg.get("user_prompt_template", ""))
                researcher_inputs.append(cfg.get("input_keys", []))
        for child in node.get("children", []) or []:
            walk(child)

    walk(workflow.get("root"))
    assert "plan_and_execute" not in node_types
    assert "conditional" not in node_types
    assert researcher_templates, "Expected at least one researcher node"
    assert researcher_inputs, "Expected researcher input keys"
    for rendered in researcher_templates:
        assert "You are investigating" in rendered
        assert "Sub-questions" in rendered
    for input_keys in researcher_inputs:
        assert "current_step" not in input_keys
        assert "research_plan" not in input_keys


def test_static_parallel_workflow_runs_coverage_review_before_final_report() -> None:
    template = (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        "### Sub-questions you MUST address\n"
        "1. Q1?\n2. Q2?\n3. Q3?\n4. Q4?\n5. Q5?\n\n"
        "### Required output structure\n"
        "- **A**: guidance.\n- **B**: guidance.\n- **C**: guidance.\n\n"
        "### Definition of done\nData unavailable when not found."
    )
    brief = WorkflowDesignBrief(
        topology="parallel_lanes",
        required_outputs=["A", "B", "C"],
        quality_gates=["Every material claim is source-backed"],
        research_lanes=[
            LaneSpec(
                description="Lane 1",
                system_prompt="Lane 1 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
            LaneSpec(
                description="Lane 2",
                system_prompt="Lane 2 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
        ],
    )

    workflow = build_web_research_workflow("test", "Test", brief)
    children = workflow["root"]["children"]

    assert [child["id"] for child in children] == [
        "coordinator",
        "parallel-lanes",
        "synthesizer",
        "coverage-reflector",
        "final-report-synthesizer",
    ]
    assert workflow["output_keys"] == ["report"]

    draft = children[2]["config"]
    coverage = children[3]["config"]
    finalizer = children[4]["config"]

    assert draft["subtype"] == "synthesizer"
    assert draft["output_key"] == "draft_report"
    assert coverage["subtype"] == "reflector"
    assert coverage["output_key"] == "coverage_review"
    assert "draft_report" in coverage["input_keys"]
    assert "Required coverage obligations: A; B; C; Lane coverage - Lane 1" in coverage[
        "system_prompt"
    ]
    assert "exhaustive for coverage" in coverage["system_prompt"]
    assert finalizer["subtype"] == "synthesizer"
    assert finalizer["output_key"] == "report"
    assert {"draft_report", "coverage_review"}.issubset(finalizer["input_keys"])
    assert "Do not include reviewer JSON" in finalizer["system_prompt"]
    assert "Required coverage obligations: A; B; C; Lane coverage - Lane 1" in finalizer[
        "system_prompt"
    ]
    assert "treat it as a hint, not a limit" in finalizer["system_prompt"]
    assert "current_step" not in finalizer["input_keys"]
    assert "research_plan" not in finalizer["input_keys"]


@pytest.mark.parametrize("topology", ["parallel_lanes", "plan_and_execute"])
def test_designer_synthesizer_prompt_does_not_impose_generic_section_cap(
    topology: str,
) -> None:
    template = (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        "### Sub-questions you MUST address\n"
        "1. Q1?\n2. Q2?\n3. Q3?\n4. Q4?\n5. Q5?\n\n"
        "### Required output structure\n"
        "- **A**: guidance.\n- **B**: guidance.\n- **C**: guidance.\n\n"
        "### Definition of done\nData unavailable when not found."
    )
    brief = WorkflowDesignBrief(
        topology=topology,
        required_outputs=["A", "B", "C", "D", "E"],
        research_lanes=[
            LaneSpec(
                description="Lane 1",
                system_prompt="Lane 1 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
            LaneSpec(
                description="Lane 2",
                system_prompt="Lane 2 charter that is at least two hundred chars long. " * 5,
                user_prompt_template=template,
            ),
        ],
    )

    workflow = build_web_research_workflow("test", "Test", brief)

    synthesizer_prompts: list[str] = []

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "agent":
            cfg = node.get("config") or {}
            if cfg.get("subtype") == "synthesizer":
                synthesizer_prompts.append(cfg.get("system_prompt", ""))
        for child in node.get("children", []) or []:
            walk(child)

    walk(workflow["root"])

    assert synthesizer_prompts
    for prompt in synthesizer_prompts:
        assert "2-3 main sections" not in prompt
        assert "2-3 max" not in prompt
        assert "do not impose a fixed section count" in prompt
        assert "workflow-specific required outputs" in prompt


def test_built_workflow_preserves_missing_lane_prompts_for_gate_feedback() -> None:
    brief = WorkflowDesignBrief(
        topology="parallel_lanes",
        research_lanes=[
            LaneSpec(description="Policy implementation evidence"),
            LaneSpec(description="Operational rollout risks"),
        ],
    )
    workflow = build_web_research_workflow(
        "Compare rollout readiness for a public program",
        "Program Readiness",
        brief,
    )

    researcher_configs: list[dict] = []

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "agent":
            cfg = node.get("config") or {}
            if cfg.get("subtype") == "researcher":
                researcher_configs.append(cfg)
        for child in node.get("children", []) or []:
            walk(child)

    walk(workflow["root"])
    assert researcher_configs
    for cfg in researcher_configs:
        system_prompt = cfg.get("system_prompt", "")
        user_prompt = cfg.get("user_prompt_template", "")
        assert "## Lane Specialization" not in system_prompt
        assert "You are investigating: **{query}**" not in user_prompt
        assert "Sub-questions" not in user_prompt
        assert "Data unavailable" not in user_prompt
        assert "Execute the following research step" in user_prompt

    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "RESEARCHER_USER_PROMPT default")


def test_supplied_parallel_brief_without_lanes_raises_instead_of_generic_fallback() -> None:
    brief = WorkflowDesignBrief(
        topology="parallel_lanes",
        workflow_name="Incomplete",
        research_lanes=[],
    )

    with pytest.raises(ValueError, match="parallel_lanes requires"):
        build_web_research_workflow(
            "Build a research workflow without authored lanes",
            "Incomplete",
            brief,
        )


# ---------------------------------------------------------------------------
# Phase 2: Structural validator detects contract violations on any topology
# ---------------------------------------------------------------------------


_VALID_TEMPLATE = (
    "## Investigation Brief\n\n"
    "You are investigating: **{query}**\n\n"
    "### Sub-questions you MUST address (in this order)\n"
    "1. What is the first concrete question?\n"
    "2. What is the second concrete question?\n"
    "3. What is the third concrete question?\n"
    "4. What is the fourth concrete question?\n"
    "5. What is the fifth concrete question?\n\n"
    "### Required output structure\n"
    "- **Section A**: evidence guidance for first slice of work.\n"
    "- **Section B**: evidence guidance for second slice.\n"
    "- **Section C**: evidence guidance for third slice.\n\n"
    "### Search strategy\n"
    "- One focused query per sub-question.\n"
    "- Prefer primary sources for this domain.\n\n"
    "### Definition of done\n"
    "Each sub-question has a concrete answer with citation, OR is marked "
    "\"Data unavailable\" — DO NOT improvise."
)

_GENERIC_DEFAULT_TEMPLATE = (
    "Execute the following research step:\n\n"
    "## Step Details\nTitle: {step_title}\nDescription: {step_description}\n"
)

_LONG_SYSTEM_PROMPT = (
    "You are a research agent in a multi-agent workflow. Use the available "
    "tools to gather evidence for the lane workstream described below. "
    "Return JSON matching the output contract at the end of this prompt.\n\n"
    "Investigate the user's actual topic. Cite primary sources. Flag dated "
    "data. Do not produce a generic overview."
)


def _researcher_node(
    *,
    node_id: str = "lane_1-researcher",
    user_prompt_template: str | None = "",
    system_prompt: str | None = None,
    tools: list[str] | None = None,
) -> dict:
    """Build a researcher-shaped agent node for validator tests."""
    config: dict = {"subtype": "researcher"}
    config["system_prompt"] = system_prompt if system_prompt is not None else _LONG_SYSTEM_PROMPT
    if user_prompt_template is not None:
        config["user_prompt_template"] = user_prompt_template
    if tools is not None:
        config["tools"] = tools
    return {
        "id": node_id,
        "type": "agent",
        "label": node_id,
        "config": config,
        "children": [],
    }


def _wrap(node: dict, *, retrieval_tool: bool = True) -> dict:
    tools = (
        [{"name": "web_search", "kind": "web_search"}]
        if retrieval_tool
        else []
    )
    return {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": [node],
        },
        "tools": tools,
    }


def _has_error_about(errors: list, fragment: str, path_suffix: str = "") -> bool:
    for err in errors:
        if fragment in err.message and (
            not path_suffix or (err.path or "").endswith(path_suffix)
        ):
            return True
    return False


def test_validator_flags_missing_user_prompt_template() -> None:
    workflow = _wrap(_researcher_node(user_prompt_template=""))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(
        errors, "no user_prompt_template", "config.user_prompt_template"
    )


def test_validator_flags_generic_default_template() -> None:
    workflow = _wrap(_researcher_node(user_prompt_template=_GENERIC_DEFAULT_TEMPLATE))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "RESEARCHER_USER_PROMPT default")


def test_validator_flags_too_short_template() -> None:
    workflow = _wrap(_researcher_node(user_prompt_template="too short."))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "short user_prompt_template")


def test_validator_flags_missing_sub_questions() -> None:
    bad = _VALID_TEMPLATE.replace(
        "### Sub-questions you MUST address (in this order)\n"
        "1. What is the first concrete question?\n"
        "2. What is the second concrete question?\n"
        "3. What is the third concrete question?\n"
        "4. What is the fourth concrete question?\n"
        "5. What is the fifth concrete question?\n",
        "### Sub-questions you MUST address\n"
        "1. Only one?\n2. Only two?\n",
    )
    workflow = _wrap(_researcher_node(user_prompt_template=bad))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "sub-questions")


def test_validator_flags_missing_output_structure() -> None:
    bad = _VALID_TEMPLATE.replace(
        "### Required output structure\n"
        "- **Section A**: evidence guidance for first slice of work.\n"
        "- **Section B**: evidence guidance for second slice.\n"
        "- **Section C**: evidence guidance for third slice.\n",
        "(deliverable block intentionally absent here)\n",
    )
    workflow = _wrap(_researcher_node(user_prompt_template=bad))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "Required output structure")


def test_validator_flags_missing_search_strategy() -> None:
    bad = _VALID_TEMPLATE.replace(
        "### Search strategy\n"
        "- One focused query per sub-question.\n"
        "- Prefer primary sources for this domain.\n",
        "(query approach block intentionally absent here)\n",
    )
    workflow = _wrap(_researcher_node(user_prompt_template=bad))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "Search strategy")


def test_validator_flags_missing_unknowns_handling() -> None:
    bad = _VALID_TEMPLATE.replace(
        "### Definition of done\n"
        "Each sub-question has a concrete answer with citation, OR is marked "
        "\"Data unavailable\" — DO NOT improvise.",
        "### Wrap up\nYou're done when you feel ready.",
    )
    workflow = _wrap(_researcher_node(user_prompt_template=bad))
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "unknowns-handling clause")


def test_validator_passes_on_compliant_template() -> None:
    workflow = _wrap(
        _researcher_node(
            user_prompt_template=_VALID_TEMPLATE,
            tools=["web_search"],
        )
    )
    errors = detect_unspecialized_agents(workflow)
    # No user_prompt_template-related errors should fire.
    assert not _has_error_about(errors, "user_prompt_template")
    assert not _has_error_about(errors, "sub-questions")


# ---------------------------------------------------------------------------
# Phase 2: contract is topology-agnostic — same validator catches violations
# regardless of where the researcher node sits in the graph.
# ---------------------------------------------------------------------------


def test_validator_catches_violation_in_plan_and_execute_body() -> None:
    """Researcher sitting inside a plan_and_execute body should be checked
    the same way as one in a parallel branch or a sequence leg."""
    researcher = _researcher_node(user_prompt_template="")
    plan_and_execute = {
        "id": "p_and_e",
        "type": "plan_and_execute",
        "label": "P&E",
        "config": {
            "planner": {
                "subtype": "planner",
                "system_prompt": "Planner prompt that is long enough to pass the system prompt check. " * 5,
            },
            "body": researcher,
        },
        "children": [],
    }
    workflow = {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": [plan_and_execute],
        },
        "tools": [{"name": "web_search", "kind": "web_search"}],
    }
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "no user_prompt_template")


def test_validator_catches_violation_in_parallel_child() -> None:
    bad_researcher = _researcher_node(user_prompt_template="")
    parallel = {
        "id": "parallel-lanes",
        "type": "parallel",
        "label": "Parallel",
        "config": {},
        "children": [bad_researcher],
    }
    workflow = {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": [parallel],
        },
        "tools": [{"name": "web_search", "kind": "web_search"}],
    }
    errors = detect_unspecialized_agents(workflow)
    assert _has_error_about(errors, "no user_prompt_template")


# ---------------------------------------------------------------------------
# Phase 2: contract is domain-agnostic — same validator decisions across
# deliberately diverse synthetic queries. Each test constructs a
# minimal-counterexample to prove the structural check, not the content.
# ---------------------------------------------------------------------------


def _render_compliant_for_query(query_phrase: str) -> str:
    """Substitute a domain phrase into the contract skeleton."""
    return (
        f"## Investigation Brief\n\n"
        f"You are investigating: **{{query}}**\n\n"
        f"### Sub-questions you MUST address (in this order)\n"
        f"1. What is the structure of {query_phrase}?\n"
        f"2. What are the key drivers of {query_phrase}?\n"
        f"3. How does {query_phrase} compare to alternatives?\n"
        f"4. What risks affect {query_phrase}?\n"
        f"5. What recent changes apply to {query_phrase}?\n\n"
        f"### Required output structure\n"
        f"- **Overview**: high-level summary of {query_phrase}.\n"
        f"- **Evidence**: concrete data points.\n"
        f"- **Risks**: open questions and uncertainties.\n\n"
        f"### Search strategy\n"
        f"- One focused query per sub-question.\n"
        f"- Prefer primary sources.\n\n"
        f"### Definition of done\n"
        f"Each sub-question has a concrete answer with citation, OR is "
        f"marked \"Data unavailable\" — DO NOT improvise."
    )


def test_validator_passes_for_diverse_domains() -> None:
    for query_phrase in [
        "Germany's healthcare system",
        "stage 2 melanoma treatment options",
        "California non-compete enforceability",
        "Hyundai Ioniq 5 winter range",
        "Spain family-safe cities",
        "CDN cache invalidation strategy",
    ]:
        template = _render_compliant_for_query(query_phrase)
        workflow = _wrap(
            _researcher_node(
                user_prompt_template=template,
                tools=["web_search"],
            )
        )
        errors = detect_unspecialized_agents(workflow)
        assert not _has_error_about(errors, "user_prompt_template"), (
            f"Compliant template for '{query_phrase}' should not flag "
            f"user_prompt_template errors. Got: {[e.message for e in errors]}"
        )
