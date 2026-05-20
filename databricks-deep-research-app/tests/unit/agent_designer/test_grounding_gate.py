"""Tests for Phase 3 synthesizer grounding gate.

Covers (a) the default ``grounding_mode="reclaim"`` on the parallel_lanes
synthesizer, (b) Designer LLM override via WorkflowDesignBrief.grounding_mode,
(c) the lane-coverage directive injected into the synthesizer system prompt,
(d) the strengthened anti-confabulation clauses in the reclaim prompt.
"""

from __future__ import annotations

from databricks_deep_research.agents.builtins.synthesizer import (
    _build_reclaim_system_prompt,
)

from deep_research.agent_designer.designer_architect import (
    LaneSpec,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.workflow_builder import (
    _synthesizer_lane_coverage_directive,
    build_web_research_workflow,
)


def _find_node(workflow: dict, node_id: str) -> dict | None:
    """Depth-first lookup of a node by id."""

    def walk(node: dict) -> dict | None:
        if node.get("id") == node_id:
            return node
        for child in node.get("children") or []:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(workflow["root"])


class TestReclaimPromptClauses:
    """Reclaim system prompt must contain explicit anti-confabulation rules."""

    def test_no_url_invention_clause(self) -> None:
        prompt = _build_reclaim_system_prompt()
        assert "NEVER cite a URL" in prompt

    def test_no_numeric_invention_clause(self) -> None:
        prompt = _build_reclaim_system_prompt()
        assert "NEVER emit numerical claims" in prompt

    def test_no_unsupported_judgment_fabrication_clause(self) -> None:
        prompt = _build_reclaim_system_prompt()
        assert "NEVER fabricate unsupported" in prompt
        assert "recommendations" in prompt
        assert "analyst" not in prompt

    def test_surface_contradictions_clause(self) -> None:
        prompt = _build_reclaim_system_prompt()
        assert "surface the contradiction" in prompt

    def test_hedging_or_omit_clause(self) -> None:
        prompt = _build_reclaim_system_prompt()
        assert "hedging" in prompt.lower()
        # The prompt must mention that omission is preferable to fabrication.
        assert "omit" in prompt.lower()


class TestLaneCoverageDirective:
    """Synthesizer prompt must enumerate lanes + forbid invention for empty ones."""

    def test_empty_lane_list_produces_empty_directive(self) -> None:
        directive = _synthesizer_lane_coverage_directive([])
        assert directive == ""

    def test_lanes_enumerated_in_directive(self) -> None:
        specs = [
            {"id": "lane_1", "description": "Fundamentals — financials"},
            {"id": "lane_2", "description": "Risk — regulatory exposure"},
        ]
        directive = _synthesizer_lane_coverage_directive(specs)
        assert "lane_1" in directive
        assert "lane_2" in directive
        assert "Fundamentals" in directive
        assert "Risk" in directive

    def test_directive_forbids_invention(self) -> None:
        specs = [{"id": "lane_1", "description": "x"}]
        directive = _synthesizer_lane_coverage_directive(specs)
        assert "Data unavailable for this lane" in directive
        assert "NEVER invent" in directive
        assert "NEVER cite a URL" in directive
        assert "NEVER emit numerical claims" in directive

    def test_directive_handles_missing_description(self) -> None:
        specs = [{"id": "lane_1"}]
        directive = _synthesizer_lane_coverage_directive(specs)
        assert "lane_1" in directive
        # Falls back to placeholder rather than raising KeyError
        assert "(no description)" in directive


class TestWorkflowGroundingModeWiring:
    """``build_web_research_workflow`` must wire the brief's grounding_mode
    onto the synthesizer node and default to ``reclaim``."""

    def _build(self, brief: WorkflowDesignBrief | None = None) -> dict:
        return build_web_research_workflow(
            intent="Research the EV market across multiple angles",
            name="EV Research",
            design_brief=brief,
        )

    def test_default_is_reclaim(self) -> None:
        workflow = self._build()
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None
        assert synth["config"].get("grounding_mode") == "reclaim"
        assert synth["config"].get("output_schema", {}).get("claim_disposition") == {
            "abstained": "remove"
        }

    def test_output_schema_carries_designer_report_contract(self) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="x",
            domain="Product analysis",
            user_goal="Compare launch readiness",
            required_outputs=["Executive summary", "Launch risks"],
            quality_gates=["No unsupported recommendations"],
            constraints=["Use recent source text"],
            topology="parallel_lanes",
            research_lanes=[
                LaneSpec(description="Adoption", system_prompt="prompt"),
            ],
        )
        workflow = self._build(brief)
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None

        report_contract = synth["config"]["output_schema"]["report_contract"]
        assert report_contract["domain"] == "Product analysis"
        assert "Executive summary" in report_contract["required_outputs"]
        assert "Launch risks" in report_contract["required_outputs"]
        assert "No unsupported recommendations" in report_contract["quality_gates"]
        assert "Use recent source text" in report_contract["constraints"]

    def test_brief_can_override_to_classical_lite(self) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="High-assurance",
            grounding_mode="classical_lite",
            topology="parallel_lanes",
            research_lanes=[LaneSpec(description="Fundamentals", system_prompt="x")],
        )
        workflow = self._build(brief)
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None
        assert synth["config"].get("grounding_mode") == "classical_lite"

    def test_brief_can_override_to_none(self) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="Brainstorm",
            grounding_mode="none",
            topology="parallel_lanes",
            research_lanes=[LaneSpec(description="Ideas", system_prompt="x")],
        )
        workflow = self._build(brief)
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None
        assert synth["config"].get("grounding_mode") == "none"
        assert synth["config"].get("output_schema") == {}

    def test_unknown_grounding_mode_falls_back_to_reclaim(self) -> None:
        """Coerce validator catches bad LLM emissions and uses safe default."""
        brief = WorkflowDesignBrief.model_validate({
            "workflow_name": "x",
            "grounding_mode": "totally-bogus-value",
            "topology": "parallel_lanes",
        })
        assert brief.grounding_mode == "reclaim"

    def test_synthesizer_prompt_includes_lane_coverage_directive(self) -> None:
        """Lane-coverage block must be in the synthesizer's system_prompt."""
        brief = WorkflowDesignBrief(
            workflow_name="x",
            topology="parallel_lanes",
            research_lanes=[
                LaneSpec(description="Lane One Description", system_prompt="prompt"),
                LaneSpec(description="Lane Two Description", system_prompt="prompt"),
            ],
        )
        workflow = self._build(brief)
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None
        system_prompt = synth["config"].get("system_prompt", "")
        assert "Lane Reporting Status" in system_prompt
        assert "Lane One Description" in system_prompt
        assert "Lane Two Description" in system_prompt
        assert "Data unavailable for this lane" in system_prompt


class TestPlanAndExecuteGrounding:
    """plan_and_execute remains supported and gets the same grounding floor."""

    def test_plan_and_execute_defaults_grounding_mode(self) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="legacy",
            topology="plan_and_execute",
            research_lanes=[LaneSpec(description="Fundamentals", system_prompt="x")],
        )
        workflow = build_web_research_workflow(
            intent="Test plan_and_execute path",
            name="Legacy",
            design_brief=brief,
        )
        synth = _find_node(workflow, "synthesizer")
        assert synth is not None
        assert synth["config"].get("grounding_mode") == "reclaim"
        assert synth["config"].get("output_schema", {}).get("claim_disposition") == {
            "abstained": "remove"
        }
        assert "Plan-And-Execute Evidence Contract" in synth["config"].get(
            "system_prompt", ""
        )
