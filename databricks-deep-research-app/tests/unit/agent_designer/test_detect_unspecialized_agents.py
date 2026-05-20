"""Unit tests for ``detect_unspecialized_agents`` — the per-agent property
validate-time defect emitter (PRD story US-04).

Covers four check categories:
    1. system_prompt missing / too short
    2. lane researcher still using the generic researcher scaffold
    3. lane researcher with no tools bound when retrieval tools are declared
    4. synthesizer on default model_tier with multiple lane producers
"""
from __future__ import annotations

from deep_research.agent_designer.semantic_validation import (
    detect_topology_mismatch,
    detect_unspecialized_agents,
    semantic_validation_errors,
)


def _agent_node(
    *,
    node_id: str,
    label: str | None = None,
    subtype: str | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    tools: list[str] | None = None,
    model_tier: str | None = None,
) -> dict:
    config: dict = {}
    if subtype is not None:
        config["subtype"] = subtype
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    if user_prompt_template is not None:
        config["user_prompt_template"] = user_prompt_template
    if tools is not None:
        config["tools"] = tools
    if model_tier is not None:
        config["model_tier"] = model_tier
    return {
        "id": node_id,
        "type": "agent",
        "label": label or node_id,
        "config": config,
        "children": [],
    }


def _compliant_user_prompt_template() -> str:
    """A user_prompt_template that satisfies the Phase 2 structural contract.

    Used by tests that assert ``defects == []`` for researcher nodes: without
    a compliant template, the validator's Check 5 would emit "no
    user_prompt_template" / "missing structural block" defects unrelated to
    what the test is verifying.
    """
    return (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        "### Sub-questions you MUST address (in this order)\n"
        "1. What is the first concrete question?\n"
        "2. What is the second concrete question?\n"
        "3. What is the third concrete question?\n"
        "4. What is the fourth concrete question?\n"
        "5. What is the fifth concrete question?\n\n"
        "### Required output structure\n"
        "- **Section A**: evidence guidance.\n"
        "- **Section B**: evidence guidance.\n"
        "- **Section C**: evidence guidance.\n\n"
        "### Search strategy\n"
        "- One focused query per sub-question.\n"
        "- Prefer primary sources.\n\n"
        "### Definition of done\n"
        "Mark unanswered sub-questions \"Data unavailable\" — DO NOT improvise."
    )


def _wrap_root(*children: dict, tools: list[dict] | None = None) -> dict:
    return {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": list(children),
        },
        "tools": tools or [],
    }


def _long_specialized_prompt() -> str:
    # Matches the structure produced by ``_assemble_lane_system_prompt`` in
    # the specialized path: minimal preamble + task-specific content + lane
    # focus footer + output contract. The detector looks for the preamble
    # AND the absence of the legacy "You are the Researcher agent..." opener.
    return (
        "You are a research agent in a multi-agent workflow. Use the available "
        "tools to gather evidence for the lane workstream described below. "
        "Return JSON matching the output contract at the end of this prompt.\n\n"
        "Investigate revenue CAGR, operating margin trend, FCF conversion. "
        "Cite latest 10-K/10-Q filings. Flag restated figures and unusual "
        "working-capital swings. Do not produce a generic overview.\n\n"
        "## Required Lane Focus\n"
        "Lane id: lane_1\n"
        "Lane workstream: Analyze fundamentals\n\n"
        "When current_step selects this lane, keep the search strategy, "
        "evidence extraction, and findings focused on this workstream.\n\n"
        "## Observation Format (CRITICAL - ALWAYS REQUIRED)\n"
        "You MUST always provide an observation. IMPORTANT: The \"observation\" "
        "field in your JSON response is REQUIRED."
    )


class TestSystemPromptQuality:
    def test_pass_long_specialized_prompt_has_no_defect(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                user_prompt_template=_compliant_user_prompt_template(),
                tools=["web_search", "web_crawl"],
            ),
            tools=[
                {"name": "web_search", "kind": "web_search", "config": {}},
                {"name": "web_crawl", "kind": "web_crawl", "config": {}},
            ],
        )
        defects = detect_unspecialized_agents(wf)
        assert defects == []

    def test_pass_designer_authored_prompt_without_builder_preamble(self) -> None:
        prompt = (
            "You are a senior domain researcher specializing in comparative "
            "analysis for this lane. Investigate concrete evidence, cite "
            "primary sources, reconcile conflicting data, flag uncertainty, "
            "and avoid unsupported claims. Focus on the assigned workstream "
            "instead of producing a generic overview. " + "x" * 260
        )
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=prompt,
                user_prompt_template=_compliant_user_prompt_template(),
                tools=["web_search", "web_crawl"],
            ),
            tools=[
                {"name": "web_search", "kind": "web_search", "config": {}},
                {"name": "web_crawl", "kind": "web_crawl", "config": {}},
            ],
        )

        defects = detect_unspecialized_agents(wf)

        assert defects == []

    def test_fail_empty_system_prompt(self) -> None:
        wf = _wrap_root(
            _agent_node(node_id="researcher_1", system_prompt=""),
        )
        defects = detect_unspecialized_agents(wf)
        empty = [d for d in defects if "empty system_prompt" in d.message]
        assert len(empty) == 1
        assert empty[0].path == "root.children[0].config.system_prompt"
        assert "update_block" in empty[0].message

    def test_fail_short_system_prompt_below_threshold(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="researcher_1",
                system_prompt="Research the topic carefully.",  # < 200 chars
            ),
        )
        defects = detect_unspecialized_agents(wf)
        short = [d for d in defects if "short system_prompt" in d.message]
        assert len(short) == 1
        assert "update_block" in short[0].message


class TestLaneSpecializationBlock:
    def test_pass_lane_researcher_with_specialization_block(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            tools=[
                {"name": "web_search", "kind": "web_search", "config": {}},
            ],
        )
        defects = detect_unspecialized_agents(wf)
        block_defects = [d for d in defects if "Lane Specialization" in d.message]
        assert block_defects == []

    def test_fail_lane_researcher_with_default_method_still_present(self) -> None:
        # Long enough to pass the length floor; explicitly carries the default
        # researcher opening line and lacks the specialized preamble — i.e.,
        # the lane never had update_block patch + propose_workflow did not
        # populate research_lanes[].system_prompt.
        prompt = (
            "You are the Researcher agent for a deep research system. Your "
            "role is to execute individual research steps.\n\n"
            "## Required Lane Focus\nLane id: lane_1\n"
            "Lane workstream: Analyze fundamentals\n\nWhen current_step "
            "selects this lane, keep the search strategy, evidence extraction, "
            "and findings focused on this workstream. " + "x" * 150
        )
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=prompt,
                tools=["web_search"],
            ),
            tools=[{"name": "web_search", "kind": "web_search", "config": {}}],
        )
        defects = detect_unspecialized_agents(wf)
        block_defects = [
            d for d in defects if "generic researcher prompt" in d.message
        ]
        assert len(block_defects) == 1
        assert "propose_workflow" in block_defects[0].message
        assert "update_block" in block_defects[0].message


class TestToolBindings:
    def test_fail_lane_researcher_with_no_tools_when_workflow_has_retrieval_tools(
        self,
    ) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=[],
            ),
            tools=[
                {"name": "web_search", "kind": "web_search", "config": {}},
                {"name": "web_crawl", "kind": "web_crawl", "config": {}},
            ],
        )
        defects = detect_unspecialized_agents(wf)
        tool_defects = [d for d in defects if "no tools bound" in d.message]
        assert len(tool_defects) == 1
        assert "bind_tool_to_block" in tool_defects[0].message
        assert "web_search" in tool_defects[0].message

    def test_pass_when_workflow_has_no_retrieval_tools_declared(self) -> None:
        # If no retrieval tools are declared at top level, do not flag agents
        # for missing tool bindings — there is nothing to bind.
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=[],
            ),
            tools=[],
        )
        defects = detect_unspecialized_agents(wf)
        tool_defects = [d for d in defects if "no tools bound" in d.message]
        assert tool_defects == []


class TestSynthesizerModelTier:
    def test_fail_synthesizer_on_default_tier_with_multiple_producers(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            _agent_node(
                node_id="lane_2-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            _agent_node(
                node_id="synthesizer",
                subtype="synthesizer",
                system_prompt="x" * 250,
                model_tier="analytical",
            ),
            tools=[{"name": "web_search", "kind": "web_search", "config": {}}],
        )
        defects = detect_unspecialized_agents(wf)
        tier_defects = [d for d in defects if "model_tier" in (d.path or "")]
        assert len(tier_defects) == 1
        assert "complex" in tier_defects[0].message
        assert "set_model_tier" in tier_defects[0].message

    def test_pass_synthesizer_on_complex_tier(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            _agent_node(
                node_id="lane_2-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            _agent_node(
                node_id="synthesizer",
                subtype="synthesizer",
                system_prompt="x" * 250,
                model_tier="complex",
            ),
            tools=[{"name": "web_search", "kind": "web_search", "config": {}}],
        )
        defects = detect_unspecialized_agents(wf)
        tier_defects = [d for d in defects if "model_tier" in (d.path or "")]
        assert tier_defects == []

    def test_pass_synthesizer_with_single_producer(self) -> None:
        # Only one lane researcher — synthesizer on analytical tier is fine.
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["web_search"],
            ),
            _agent_node(
                node_id="synthesizer",
                subtype="synthesizer",
                system_prompt="x" * 250,
                model_tier="analytical",
            ),
            tools=[{"name": "web_search", "kind": "web_search", "config": {}}],
        )
        defects = detect_unspecialized_agents(wf)
        tier_defects = [d for d in defects if "model_tier" in (d.path or "")]
        assert tier_defects == []


class TestSeparationFromCrudValidator:
    """``semantic_validation_errors`` is the CRUD save-path validator and must
    stay limited to STRUCTURAL invariants. The new per-agent quality defects
    are advice for the Designer chat LLM, not CRUD-blocking errors.

    Confirm the two are intentionally NOT merged: a minimal valid AST passes
    ``semantic_validation_errors`` even though it would trigger several
    ``detect_unspecialized_agents`` defects (no system_prompt, etc.).
    """

    def test_minimal_ast_passes_semantic_validation_errors(self) -> None:
        # Identical shape to the CRUD test's _minimal_valid_definition.
        wf = {
            "tools": [],
            "root": {
                "id": "root",
                "type": "agent",
                "label": "root",
                "config": {"subtype": "researcher"},
                "children": [],
            },
        }
        structural = semantic_validation_errors(wf)
        assert structural == [], (
            "semantic_validation_errors must stay structural-only; quality "
            "defects go through detect_unspecialized_agents instead."
        )
        # The same workflow DOES trigger the quality check.
        quality = detect_unspecialized_agents(wf)
        assert any("empty system_prompt" in d.message for d in quality)

    def test_undeclared_tool_still_flagged_by_semantic_validation_errors(self) -> None:
        wf = _wrap_root(
            _agent_node(
                node_id="lane_1-researcher",
                subtype="researcher",
                system_prompt=_long_specialized_prompt(),
                tools=["undeclared_tool"],
            ),
        )
        structural = semantic_validation_errors(wf)
        assert any(
            "undeclared tool 'undeclared_tool'" in d.message for d in structural
        )


class TestDetectTopologyMismatch:
    """Phase 4.5 — flag plan_and_execute workflows whose lanes are independent
    and would run better as parallel_lanes."""

    def _pae_with_lanes(self, lane_count: int) -> dict:
        """Build a minimal plan_and_execute workflow shape with N lane
        researchers under a conditional router (+1 cross-lane fallback)."""
        lane_children = [
            {
                "id": f"lane_{i}-researcher",
                "type": "agent",
                "label": f"Lane {i}",
                "config": {"subtype": "researcher", "system_prompt": "x" * 200},
                "children": [],
            }
            for i in range(1, lane_count + 1)
        ]
        fallback = {
            "id": "cross-lane-researcher",
            "type": "agent",
            "label": "Cross-lane",
            "config": {"subtype": "researcher", "system_prompt": "x" * 200},
            "children": [],
        }
        lane_router = {
            "id": "lane-router",
            "type": "conditional",
            "label": "Lane Router",
            "config": {"conditions": [], "default_branch": lane_count},
            "children": [*lane_children, fallback],
        }
        body = {
            "id": "body",
            "type": "sequence",
            "label": "body",
            "config": {},
            "children": [lane_router],
        }
        pae = {
            "id": "plan-and-execute",
            "type": "plan_and_execute",
            "label": "P&E",
            "config": {"body": body},
            "children": [],
        }
        return {
            "root": {
                "id": "main",
                "type": "sequence",
                "label": "main",
                "config": {},
                "children": [pae],
            },
            "tools": [],
        }

    def test_flags_plan_and_execute_with_three_or_more_lanes(self) -> None:
        wf = self._pae_with_lanes(3)
        defects = detect_topology_mismatch(wf)
        assert len(defects) == 1
        assert "parallel_lanes" in defects[0].message
        assert "3 lanes" in defects[0].message

    def test_flags_plan_and_execute_with_many_lanes(self) -> None:
        wf = self._pae_with_lanes(10)
        defects = detect_topology_mismatch(wf)
        assert len(defects) == 1
        assert "10 lanes" in defects[0].message

    def test_does_not_flag_plan_and_execute_with_two_lanes(self) -> None:
        # Threshold is ≥3 lanes — two-lane workflows might genuinely want
        # plan_and_execute's reflection loop.
        wf = self._pae_with_lanes(2)
        defects = detect_topology_mismatch(wf)
        assert defects == []

    def test_does_not_flag_parallel_lanes_topology(self) -> None:
        # parallel_lanes workflows have no plan_and_execute node — they
        # cannot trigger this finding.
        wf = {
            "root": {
                "id": "main",
                "type": "sequence",
                "label": "main",
                "config": {},
                "children": [
                    {
                        "id": "coordinator",
                        "type": "agent",
                        "label": "Coordinator",
                        "config": {"subtype": "coordinator", "system_prompt": "x" * 200},
                        "children": [],
                    },
                    {
                        "id": "parallel-lanes",
                        "type": "parallel",
                        "label": "Parallel",
                        "config": {},
                        "children": [
                            {
                                "id": "lane_1-researcher",
                                "type": "agent",
                                "label": "L1",
                                "config": {"subtype": "researcher", "system_prompt": "x" * 200},
                                "children": [],
                            }
                        ],
                    },
                ],
            },
            "tools": [],
        }
        assert detect_topology_mismatch(wf) == []
