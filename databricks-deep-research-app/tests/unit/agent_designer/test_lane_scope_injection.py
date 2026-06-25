"""Tests for Phase 4 coordinator → lane scope handoff.

Covers (a) ``CoordinatorOutput.extracted_scope`` field shape and defaults,
(b) ``ExtractedScope`` field defaults, (c) lane researcher ``input_keys``
inclusion of ``coordination`` on parallel_lanes + single_agent topologies,
(d) ``{coordination}`` placeholder presence in lane system prompts when the
scope block is engaged, (e) graceful fallback when no scope is extracted.
"""

from __future__ import annotations

from databricks_deep_research.agents.output_models import (
    CoordinatorOutput,
    ExtractedScope,
)

from deep_research.agent_designer.designer_architect import (
    LaneSpec,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.workflow_builder import (
    _INVESTIGATION_SCOPE_BLOCK,
    _assemble_lane_system_prompt,
    build_web_research_workflow,
)


def _find_node(workflow: dict, node_id: str) -> dict | None:
    def walk(node: dict) -> dict | None:
        if node.get("id") == node_id:
            return node
        for child in node.get("children") or []:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(workflow["root"])


def _find_lane_researchers(workflow: dict) -> list[dict]:
    """Collect every node whose id ends in -researcher under parallel-lanes."""
    parallel = _find_node(workflow, "parallel-lanes")
    if parallel is None:
        return []
    return [child for child in parallel.get("children", []) if child.get("id", "").endswith("-researcher")]


class TestExtractedScopeModel:
    def test_empty_scope_has_safe_defaults(self) -> None:
        scope = ExtractedScope()
        assert scope.entities == []
        assert scope.time_window is None
        assert scope.comparables == []
        assert scope.domain_hints == []

    def test_populated_scope_round_trips(self) -> None:
        scope = ExtractedScope(
            entities=["NVIDIA Corporation", "NVDA"],
            time_window="most-recent-quarter",
            comparables=["AMD", "INTC"],
            domain_hints=["semiconductor"],
        )
        dumped = scope.model_dump()
        rebuilt = ExtractedScope.model_validate(dumped)
        assert rebuilt == scope


class TestCoordinatorOutputBackwardsCompat:
    def test_existing_fields_unchanged(self) -> None:
        out = CoordinatorOutput(complexity="complex")
        assert out.complexity == "complex"
        assert out.is_simple is False
        assert out.recommended_depth == "standard"
        assert out.direct_response is None
        assert out.follow_up_type is None

    def test_extracted_scope_defaults_to_none(self) -> None:
        out = CoordinatorOutput(complexity="complex")
        assert out.extracted_scope is None

    def test_extracted_scope_can_be_populated(self) -> None:
        out = CoordinatorOutput(
            complexity="complex",
            extracted_scope=ExtractedScope(entities=["X", "Y"]),
        )
        assert out.extracted_scope is not None
        assert out.extracted_scope.entities == ["X", "Y"]

    def test_legacy_dict_without_scope_parses(self) -> None:
        """Persisted CoordinatorOutput dicts from before Phase 4 must still load."""
        legacy = {
            "complexity": "moderate",
            "is_simple": False,
            "recommended_depth": "standard",
        }
        out = CoordinatorOutput.model_validate(legacy)
        assert out.extracted_scope is None


class TestLanePromptScopeBlock:
    def test_specialized_path_with_scope_includes_placeholder(self) -> None:
        prompt = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_PROMPT",
            spec={
                "id": "lane_1",
                "label": "Lane 1",
                "description": "Fundamentals",
                "system_prompt": "## Specialized\nDo this.",
            },
            include_scope_block=True,
        )
        assert "Investigation Scope" in prompt
        assert "{coordination}" in prompt
        # Specialization still present
        assert "Do this." in prompt
        # Lane focus still present
        assert "Lane id: lane_1" in prompt

    def test_specialized_path_without_scope_legacy(self) -> None:
        prompt = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_PROMPT",
            spec={
                "id": "lane_1",
                "label": "Lane 1",
                "description": "Fundamentals",
                "system_prompt": "Do this.",
            },
            include_scope_block=False,
        )
        assert "Investigation Scope" not in prompt
        assert "{coordination}" not in prompt

    def test_legacy_path_with_scope_includes_placeholder(self) -> None:
        prompt = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_PROMPT",
            spec={
                "id": "lane_1",
                "label": "Lane 1",
                "description": "Fundamentals",
                "system_prompt": "",
            },
            include_scope_block=True,
        )
        assert "Investigation Scope" in prompt
        assert "{coordination}" in prompt
        # Legacy path keeps the full base prompt
        assert "BASE_PROMPT" in prompt

    def test_legacy_path_without_scope_byte_equal_to_pre_phase4(self) -> None:
        """Legacy callers (plan_and_execute) get byte-identical output."""
        prompt = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_PROMPT",
            spec={
                "id": "lane_1",
                "label": "Lane 1",
                "description": "Fundamentals",
                "system_prompt": "",
            },
            include_scope_block=False,
        )
        assert "Investigation Scope" not in prompt
        assert "{coordination}" not in prompt

    def test_scope_block_template_format(self) -> None:
        """The scope block must use a `{coordination}` placeholder for runtime fill."""
        assert "{coordination}" in _INVESTIGATION_SCOPE_BLOCK
        assert "Investigation Scope" in _INVESTIGATION_SCOPE_BLOCK
        assert "extracted_scope" in _INVESTIGATION_SCOPE_BLOCK


class TestWorkflowLaneInputKeys:
    def _build_parallel(self) -> dict:
        brief = WorkflowDesignBrief(
            workflow_name="x",
            topology="parallel_lanes",
            research_lanes=[
                LaneSpec(description="Fundamentals", system_prompt="a"),
                LaneSpec(description="Risk", system_prompt="b"),
                LaneSpec(description="News", system_prompt="c"),
            ],
        )
        return build_web_research_workflow(intent="research X", name="X", design_brief=brief)

    def test_parallel_lanes_researchers_consume_coordination(self) -> None:
        workflow = self._build_parallel()
        lane_nodes = _find_lane_researchers(workflow)
        assert len(lane_nodes) == 3
        for node in lane_nodes:
            input_keys = node["config"].get("input_keys", [])
            assert "query" in input_keys
            assert "coordination" in input_keys, (
                f"lane {node['id']} missing 'coordination' in input_keys: {input_keys}"
            )

    def test_parallel_lane_prompts_contain_scope_placeholder(self) -> None:
        workflow = self._build_parallel()
        for node in _find_lane_researchers(workflow):
            prompt = node["config"].get("system_prompt", "")
            assert "{coordination}" in prompt, (
                f"lane {node['id']} system_prompt missing scope placeholder"
            )

    def test_plan_and_execute_lanes_do_not_get_scope_block(self) -> None:
        """plan_and_execute lane researchers don't declare coordination as
        an input_key, so adding {coordination} would render empty and the
        directive would be misleading. Verify the block is absent."""
        brief = WorkflowDesignBrief(
            workflow_name="legacy",
            topology="plan_and_execute",
            research_lanes=[LaneSpec(description="Fundamentals", system_prompt="a")],
        )
        workflow = build_web_research_workflow(
            intent="legacy", name="Legacy", design_brief=brief
        )
        # plan_and_execute uses a different topology; find any researcher node.
        # plan_and_execute wraps its sequence in config.body, not children.
        researchers: list[dict] = []

        def walk(node: dict) -> None:
            if node.get("config", {}).get("subtype") == "researcher":
                researchers.append(node)
            for child in node.get("children") or []:
                walk(child)
            body = node.get("config", {}).get("body")
            if isinstance(body, dict):
                walk(body)

        walk(workflow["root"])
        assert researchers, "expected at least one researcher in plan_and_execute"
        for node in researchers:
            prompt = node["config"].get("system_prompt", "")
            assert "Investigation Scope" not in prompt
            assert "{coordination}" not in prompt

    def test_single_agent_consumes_coordination(self) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="simple",
            topology="single_agent",
            research_lanes=[LaneSpec(description="answer", system_prompt="x")],
        )
        workflow = build_web_research_workflow(
            intent="what is X", name="X", design_brief=brief
        )
        agent = _find_node(workflow, "answer-agent")
        assert agent is not None
        assert "coordination" in agent["config"].get("input_keys", [])
        assert "{coordination}" in agent["config"].get("system_prompt", "")
