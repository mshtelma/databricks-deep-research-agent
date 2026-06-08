"""Unit tests for the LaneSpec backwards-compatible brief coercion and the
``system_prompt`` plumbing into the generated lane-researcher prompts.

Covers PRD stories US-01 (schema + validator) and US-02 (builder consumption).
"""
from __future__ import annotations

from deep_research.agent_designer.designer_architect import (
    DomainProfile,
    LaneSpec,
    WorkflowDesignBrief,
    compile_workflow_design_brief,
    designer_system_prompt,
    format_workflow_design_brief,
)
from deep_research.agent_designer.tools import CHAT_TOOLS
from deep_research.agent_designer.workflow_builder import (
    _assemble_lane_system_prompt,
    _lane_specs,
    build_web_research_workflow,
)


class TestLaneSpecCoercion:
    """US-01: WorkflowDesignBrief / DomainProfile coerce research_lanes."""

    def test_legacy_list_of_strings_coerces(self) -> None:
        brief = WorkflowDesignBrief(
            research_lanes=["Analyze fundamentals", "Review competitors"],
        )
        assert all(isinstance(lane, LaneSpec) for lane in brief.research_lanes)
        assert [lane.description for lane in brief.research_lanes] == [
            "Analyze fundamentals",
            "Review competitors",
        ]
        assert all(lane.system_prompt == "" for lane in brief.research_lanes)

    def test_structured_dicts_with_system_prompt_parse(self) -> None:
        brief = WorkflowDesignBrief(
            research_lanes=[
                {
                    "description": "Analyze fundamentals",
                    "system_prompt": (
                        "Investigate revenue CAGR, operating margin, FCF."
                    ),
                },
                {"description": "Review competitors"},
            ],
        )
        assert brief.research_lanes[0].description == "Analyze fundamentals"
        assert "FCF" in brief.research_lanes[0].system_prompt
        assert brief.research_lanes[1].description == "Review competitors"
        assert brief.research_lanes[1].system_prompt == ""

    def test_mixed_list_of_strings_and_dicts(self) -> None:
        brief = WorkflowDesignBrief(
            research_lanes=[
                "Analyze fundamentals",
                {
                    "description": "Review competitors",
                    "system_prompt": "Compare 3 named peers on growth.",
                },
            ],
        )
        assert brief.research_lanes[0].system_prompt == ""
        assert "Compare 3 named peers" in brief.research_lanes[1].system_prompt

    def test_empties_filtered(self) -> None:
        brief = WorkflowDesignBrief(
            research_lanes=[
                "",
                None,
                "Real lane",
                {"description": "", "system_prompt": "orphan"},
            ],
        )
        assert len(brief.research_lanes) == 1
        assert brief.research_lanes[0].description == "Real lane"

    def test_domain_profile_coerces_too(self) -> None:
        profile = DomainProfile(
            label="Demo",
            research_lanes=[
                "Lane A",
                {"description": "Lane B", "system_prompt": "B prompt"},
            ],
        )
        assert all(isinstance(lane, LaneSpec) for lane in profile.research_lanes)
        assert profile.research_lanes[1].system_prompt == "B prompt"


class TestMergeAndCompile:
    """US-01: _merge_lane_lists preserves first-occurrence system_prompt."""

    def test_supplied_brief_specialization_wins_over_profile_default(self) -> None:
        supplied = WorkflowDesignBrief(
            research_lanes=[
                {
                    "description": "Identify the company, ticker, exchange, business model, and peer set.",
                    "system_prompt": "LLM-supplied specialization wins.",
                },
            ],
        )
        compiled = compile_workflow_design_brief(
            "investment research on NVDA",
            supplied,
        )
        first = compiled.research_lanes[0]
        assert first.description.startswith("Identify the company")
        assert first.system_prompt == "LLM-supplied specialization wins."

    def test_supplied_brief_does_not_merge_semantic_profile_defaults(self) -> None:
        supplied = WorkflowDesignBrief(
            workflow_name="Custom workflow",
            domain="Custom Domain",
            research_lanes=[],
            required_outputs=[],
            quality_gates=[],
            constraints=[],
        )
        compiled = compile_workflow_design_brief(
            "build a workflow for a specific customer request",
            supplied,
        )

        assert compiled.research_lanes == []
        assert compiled.required_outputs == []
        assert compiled.quality_gates == []
        assert compiled.constraints == []

    def test_format_workflow_design_brief_shows_descriptions_only(self) -> None:
        brief = WorkflowDesignBrief(
            domain="Test Domain",
            research_lanes=[
                {
                    "description": "Analyze fundamentals",
                    "system_prompt": "SECRET specialization content not for brief summary",
                },
            ],
        )
        formatted = format_workflow_design_brief(brief)
        assert "Analyze fundamentals" in formatted
        assert "SECRET specialization" not in formatted, (
            "system_prompt content must NOT leak into the brief-summary block "
            "rendered into agent prompts; it is consumed by the builder per-lane."
        )

    def test_no_lane_spec_repr_in_rendered_designer_system_prompt(self) -> None:
        prompt = designer_system_prompt()
        assert "LaneSpec(" not in prompt
        assert "description='" not in prompt or "research_lanes" in prompt

    def test_designer_contract_requires_llm_authored_lane_prompts(self) -> None:
        prompt = designer_system_prompt()
        assert "topology: choose parallel_lanes, plan_and_execute, or single_agent" in prompt
        assert "specialized system_prompt, user_prompt_template" in prompt
        assert "the LLM authors these use-case prompts" in prompt
        assert "STATIC ``parallel_lanes``" in prompt
        assert "Legacy compatibility profile summaries" in prompt

        propose_tool = next(
            tool for tool in CHAT_TOOLS if tool["function"]["name"] == "propose_workflow"
        )
        description = propose_tool["function"]["description"]
        assert "topology" in description
        assert "static parallel_lanes" in description
        assert "LLM-authored system_prompt and user_prompt_template" in description

    def test_designer_contract_does_not_force_public_web_tools(self) -> None:
        prompt = designer_system_prompt()
        normalized = " ".join(prompt.split())
        assert "web_search + web_crawl" not in prompt
        assert "answerable by web search" not in prompt
        assert "at least one evidence tool" in normalized
        assert "appropriate to their evidence path" in normalized


class TestBuilderPlumbing:
    """US-02: lane researcher prompt assembly injects '## Lane Specialization'
    block when the LaneSpec carries a non-empty system_prompt."""

    def test_lane_specs_preserves_designer_prompt_fields_without_fallback(self) -> None:
        brief = WorkflowDesignBrief(
            research_lanes=[
                {
                    "description": "Analyze fundamentals",
                    "system_prompt": "Specialized addendum text.",
                },
                "Plain lane",
            ],
        )
        specs = _lane_specs(brief)
        assert len(specs) == 2
        assert specs[0]["description"] == "Analyze fundamentals"
        assert specs[0]["system_prompt"] == "Specialized addendum text."
        assert specs[1]["description"] == "Plain lane"
        assert specs[1]["system_prompt"] == ""
        assert specs[1]["user_prompt_template"] == ""

    def test_assemble_with_specialization_replaces_default_method(self) -> None:
        # Specialized path: the lane researcher's system_prompt no longer
        # contains the generic ``RESEARCHER_DEFAULT_METHOD`` opening; instead
        # it starts with the minimal preamble and is dominated by the
        # LLM-supplied specialization content. The output contract is
        # appended at the end so observation parsing still works.
        rendered = _assemble_lane_system_prompt(
            base_researcher_prompt="SHOULD_NOT_APPEAR",
            spec={
                "id": "lane_1",
                "label": "Lane 1: foo",
                "description": "Analyze fundamentals",
                "system_prompt": (
                    "Investigate revenue CAGR, margins, FCF. "
                    "Cite 10-K/10-Q. Flag restated figures."
                ),
            },
        )
        # Minimal preamble present.
        assert rendered.startswith(
            "You are a research agent in a multi-agent workflow."
        )
        # Generic method opening is gone — the specialization REPLACES it.
        assert "You are the Researcher agent for a deep research system" not in rendered
        # The provided base_researcher_prompt is ignored on the specialized path.
        assert "SHOULD_NOT_APPEAR" not in rendered
        # Specialization content present.
        assert "Investigate revenue CAGR" in rendered
        # Lane focus footer + output contract present.
        assert "## Required Lane Focus" in rendered
        assert "Observation Format" in rendered
        # Ordering: specialization before lane focus before output contract.
        assert rendered.index("Investigate revenue CAGR") < rendered.index(
            "## Required Lane Focus"
        )
        assert rendered.index("## Required Lane Focus") < rendered.index(
            "Observation Format"
        )

    def test_assemble_without_specialization_byte_equal_to_legacy(self) -> None:
        rendered = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_RESEARCHER",
            spec={
                "id": "lane_2",
                "label": "Lane 2: bar",
                "description": "Plain lane",
                "system_prompt": "",
            },
        )
        # Must contain the standard footer.
        assert "## Required Lane Focus" in rendered
        assert "Lane id: lane_2" in rendered
        assert "Lane workstream: Plain lane" in rendered
        # The legacy assembly produced exactly this shape; assert it explicitly.
        expected_legacy = (
            "BASE_RESEARCHER\n"
            "\n"
            "## Required Lane Focus\n"
            "Lane id: lane_2\n"
            "Lane workstream: Plain lane\n"
            "\n"
            "When current_step selects this lane, keep the search strategy, "
            "evidence extraction, and findings focused on this workstream."
        )
        assert rendered == expected_legacy

    def test_static_parallel_lane_prompt_does_not_reference_current_step(self) -> None:
        rendered = _assemble_lane_system_prompt(
            base_researcher_prompt="BASE_RESEARCHER",
            spec={
                "id": "lane_2",
                "label": "Lane 2: bar",
                "description": "Plain lane",
                "system_prompt": "",
            },
            include_scope_block=True,
        )

        assert "Run this static lane once" in rendered
        assert "current_step selects this lane" not in rendered

    def test_build_web_research_workflow_routes_specialization_to_lane_agent(
        self,
    ) -> None:
        brief = WorkflowDesignBrief(
            workflow_name="Test",
            domain="Test",
            research_lanes=[
                {
                    "description": "Analyze financial fundamentals",
                    "system_prompt": (
                        "Investigate 5-year revenue CAGR, operating margin "
                        "trend, and FCF conversion. Cite latest 10-K/10-Q. "
                        "Flag restated figures. Do not produce a generic "
                        "overview."
                    ),
                    "user_prompt_template": (
                        "## Investigation Brief\n\n"
                        "You are investigating: **{query}**\n\n"
                        "### Sub-questions you MUST address (in this order)\n"
                        "1. What is the latest revenue growth rate for the target company?\n"
                        "2. What is the current operating margin trend for the target company?\n"
                        "3. What is the free cash flow conversion for the target company?\n"
                        "4. What balance-sheet metrics affect the target company thesis?\n"
                        "5. What accounting or restatement risks affect the target company?\n\n"
                        "### Required output structure\n"
                        "- **Financial trajectory**: Cite source-backed revenue and margin facts.\n"
                        "- **Cash generation**: Cite free cash flow and liquidity facts.\n"
                        "- **Open issues**: Mark missing or conflicting data.\n\n"
                        "### Search strategy\n"
                        "- Search filings and investor relations pages first.\n"
                        "- Refine with exact metric names before using secondary sources.\n\n"
                        "### Definition of done\n"
                        "Mark missing evidence as \"Data unavailable\" -- DO NOT improvise."
                    ),
                },
            ],
        )
        wf = build_web_research_workflow(
            "investment research on NVDA", "TestAgent", brief
        )

        # Walk the workflow tree to find lane_1's researcher node
        def find_lane_1_researcher(node: dict | None) -> dict | None:
            if not isinstance(node, dict):
                return None
            if node.get("id") == "lane_1-researcher":
                return node
            for child in node.get("children", []) or []:
                hit = find_lane_1_researcher(child)
                if hit is not None:
                    return hit
            cfg = node.get("config", {}) or {}
            if isinstance(cfg, dict):
                body = cfg.get("body")
                if isinstance(body, dict):
                    hit = find_lane_1_researcher(body)
                    if hit is not None:
                        return hit
            return None

        agent = find_lane_1_researcher(wf.get("root", wf))
        assert agent is not None, "lane_1-researcher not found in generated workflow"
        sp = agent.get("config", {}).get("system_prompt", "") or ""
        # Specialized path: minimal preamble + task content + output contract.
        # The generic RESEARCHER_DEFAULT_METHOD opening is REPLACED, not appended to.
        assert sp.startswith(
            "You are a research agent in a multi-agent workflow."
        )
        assert "You are the Researcher agent for a deep research system" not in sp
        assert "Investigate 5-year revenue CAGR" in sp
        assert "Observation Format" in sp  # output contract preserved
