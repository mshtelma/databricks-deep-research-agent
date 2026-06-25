"""Unit tests for the config translator (config_translator.py).

Tests the translation of OrchestrationConfig -> WorkflowDefinition,
using parameterized tests for research depth mapping, source scope
filtering, system instructions injection, and workflow structure.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from databricks_deep_research.workflow.definition import NodeType, WorkflowDefinition, WorkflowNode

from deep_research.agent.adapters.config_translator import (
    _build_pools,
    _depth_to_step_limits,
    _resolve_search_tools,
    translate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(**overrides: Any) -> MagicMock:
    """Build a MagicMock OrchestrationConfig with sensible defaults.

    Uses spec=None so getattr(config, field, default) works correctly:
    for fields NOT in overrides, getattr returns the provided default.
    """
    config = MagicMock()

    # Disable auto-creation of attributes so getattr(..., default) works
    # by explicitly deleting attributes that the translator looks up via
    # getattr(config, name, default).
    # The translator uses: enable_background_investigation, research_depth,
    # source_scope, system_instructions, output_format, output_schema,
    # structured_system_prompt, structured_user_prompt, workflow_mode,
    # manual_steps, max_plan_iterations
    defaults: dict[str, Any] = {
        "enable_background_investigation": True,
        "research_depth": "auto",
        "source_scope": None,
        "system_instructions": None,
        "output_format": "markdown",
        "output_schema": None,
        "structured_system_prompt": None,
        "structured_user_prompt": None,
        "workflow_mode": "planner",
        "manual_steps": None,
        "max_plan_iterations": 3,
        "synthesis_mode": "simple",
        "verify_sources": True,
        "enable_post_verification": False,
        "tone": None,
        "output_language": None,
    }
    defaults.update(overrides)

    for key, value in defaults.items():
        setattr(config, key, value)

    return config


def _find_child(node: WorkflowNode, child_id: str) -> WorkflowNode | None:
    """Find a direct child by id."""
    for c in node.children:
        if c.id == child_id:
            return c
    return None


# ---------------------------------------------------------------------------
# Tests — Default workflow structure
# ---------------------------------------------------------------------------


class TestDefaultWorkflowStructure:
    """Tests for the default config -> 4-child workflow tree."""

    def test_produces_workflow_definition(self) -> None:
        config = _mock_config()
        wf = translate(config)

        assert isinstance(wf, WorkflowDefinition)
        assert wf.id == "deep_research"
        assert wf.name == "Deep Research"
        assert "query" in wf.required_inputs
        assert "report" in wf.output_keys

    def test_root_is_sequence_with_four_children(self) -> None:
        config = _mock_config()
        wf = translate(config)

        root = wf.root
        assert root.type == NodeType.sequence
        assert root.id == "main"
        assert len(root.children) == 4

    def test_child_order(self) -> None:
        config = _mock_config()
        wf = translate(config)

        ids = [c.id for c in wf.root.children]
        assert ids == ["coordinator", "background", "research_cycle", "synthesizer"]

    def test_coordinator_node(self) -> None:
        config = _mock_config()
        wf = translate(config)

        coord = wf.root.children[0]
        assert coord.id == "coordinator"
        assert coord.type == NodeType.agent
        assert coord.config["subtype"] == "coordinator"
        assert coord.config["model_tier"] == "simple"

    def test_background_node(self) -> None:
        config = _mock_config()
        wf = translate(config)

        bg = _find_child(wf.root, "background")
        assert bg is not None
        assert bg.type == NodeType.agent
        assert bg.config["subtype"] == "background"
        assert bg.config["pool_writes"] == [{"pool": "discovery_sources", "extract": "sources"}]

    def test_research_cycle_node(self) -> None:
        config = _mock_config()
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.type == NodeType.plan_and_execute

    def test_synthesizer_node(self) -> None:
        config = _mock_config()
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.type == NodeType.agent
        assert synth.config["subtype"] == "synthesizer"
        assert synth.config["model_tier"] == "complex"


# ---------------------------------------------------------------------------
# Tests — Background disabled
# ---------------------------------------------------------------------------


class TestBackgroundDisabled:
    """Tests for enable_background_investigation=False."""

    def test_three_children_without_background(self) -> None:
        config = _mock_config(enable_background_investigation=False)
        wf = translate(config)

        assert len(wf.root.children) == 3
        ids = [c.id for c in wf.root.children]
        assert "background" not in ids

    def test_order_preserved(self) -> None:
        config = _mock_config(enable_background_investigation=False)
        wf = translate(config)

        ids = [c.id for c in wf.root.children]
        assert ids == ["coordinator", "research_cycle", "synthesizer"]


# ---------------------------------------------------------------------------
# Tests — Research depth mapping (parameterized)
# ---------------------------------------------------------------------------


class TestDepthMapping:
    """Tests for _depth_to_step_limits and its integration in translate."""

    @pytest.mark.parametrize(
        "depth, expected_min, expected_max",
        [
            ("light", 1, 3),
            ("medium", 2, 5),
            ("extended", 3, 7),
            ("auto", 2, 7),
        ],
        ids=["light", "medium", "extended", "auto"],
    )
    def test_depth_to_step_limits(
        self, depth: str, expected_min: int, expected_max: int
    ) -> None:
        min_steps, max_steps = _depth_to_step_limits(depth)
        assert min_steps == expected_min
        assert max_steps == expected_max

    def test_unknown_depth_defaults_to_auto(self) -> None:
        min_steps, max_steps = _depth_to_step_limits("unknown_depth")
        assert (min_steps, max_steps) == (2, 7)

    @pytest.mark.parametrize(
        "depth, expected_min, expected_max",
        [
            ("light", 1, 3),
            ("medium", 2, 5),
            ("extended", 3, 7),
            ("auto", 2, 7),
        ],
        ids=["light-wf", "medium-wf", "extended-wf", "auto-wf"],
    )
    def test_depth_wired_into_research_cycle(
        self, depth: str, expected_min: int, expected_max: int
    ) -> None:
        """Verify depth is wired through to the plan_and_execute node config."""
        config = _mock_config(research_depth=depth)
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.config["min_iterations"] == expected_min
        assert rc.config["max_iterations"] == expected_max


# ---------------------------------------------------------------------------
# Tests — Source scope filtering (parameterized)
# ---------------------------------------------------------------------------


class TestSourceScopeFiltering:
    """Tests for _resolve_search_tools and source scope in translate."""

    @pytest.mark.parametrize(
        "scope, expected_tool_count",
        [
            (None, 2),          # Default: web allowed
            ("web_only", 2),    # Explicit web
            ("all", 2),         # All: web + (enterprise handled elsewhere)
            ("none", 0),        # No tools
            ("enterprise", 0),  # Enterprise only: no web tools
        ],
        ids=["default-none", "web_only", "all", "none", "enterprise"],
    )
    def test_resolve_search_tools(self, scope: str | None, expected_tool_count: int) -> None:
        config = _mock_config(source_scope=scope)
        tools = _resolve_search_tools(config)
        assert len(tools) == expected_tool_count

    @pytest.mark.parametrize(
        "scope",
        [None, "web_only", "all"],
        ids=["default", "web_only", "all"],
    )
    def test_web_tools_include_search_and_crawl(self, scope: str | None) -> None:
        config = _mock_config(source_scope=scope)
        tools = _resolve_search_tools(config)
        names = {t if isinstance(t, str) else t["name"] for t in tools}
        assert "web_search" in names
        assert "web_crawl" in names

    def test_scope_none_gives_no_tools(self) -> None:
        """source_scope='none' disables all tools."""
        config = _mock_config(source_scope="none")
        tools = _resolve_search_tools(config)
        assert tools == []

    def test_tools_propagated_to_background_node(self) -> None:
        config = _mock_config(source_scope="web_only")
        wf = translate(config)

        bg = _find_child(wf.root, "background")
        assert bg is not None
        assert len(bg.config["tools"]) == 2

    def test_tools_propagated_to_researcher(self) -> None:
        config = _mock_config(source_scope="web_only")
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        body = rc.config["body"]
        assert len(body["config"]["tools"]) == 2

    # --- Enterprise tools via available_tools ---

    def test_enterprise_only_with_available_tools(self) -> None:
        """source_scope='enterprise_only' + available_tools includes enterprise, not web."""
        config = _mock_config(source_scope="enterprise_only")
        tools = _resolve_search_tools(
            config,
            available_tools=["web_search", "web_crawl", "query_genie_sales", "search_docs"],
        )
        names = {t if isinstance(t, str) else t["name"] for t in tools}
        assert "web_search" not in names
        assert "web_crawl" not in names
        assert "query_genie_sales" in names
        assert "search_docs" in names

    def test_all_scope_with_available_tools(self) -> None:
        """source_scope='all' + available_tools includes both web and enterprise."""
        config = _mock_config(source_scope="all")
        tools = _resolve_search_tools(
            config,
            available_tools=["web_search", "web_crawl", "query_genie_hr"],
        )
        names = {t if isinstance(t, str) else t["name"] for t in tools}
        assert "web_search" in names
        assert "web_crawl" in names
        assert "query_genie_hr" in names

    def test_web_only_scope_ignores_enterprise_tools(self) -> None:
        """source_scope='web_only' + available_tools does NOT include enterprise tools."""
        config = _mock_config(source_scope="web_only")
        tools = _resolve_search_tools(
            config,
            available_tools=["web_search", "web_crawl", "query_genie_sales"],
        )
        names = {t if isinstance(t, str) else t["name"] for t in tools}
        assert "web_search" in names
        assert "web_crawl" in names
        assert "query_genie_sales" not in names

    def test_no_available_tools_no_enterprise(self) -> None:
        """available_tools=None -> no enterprise tools regardless of scope."""
        config = _mock_config(source_scope="all")
        tools = _resolve_search_tools(config, available_tools=None)
        assert len(tools) == 2  # Only web tools

    def test_enterprise_tools_propagated_to_researcher_node(self) -> None:
        """Enterprise tools from available_tools end up in researcher node config."""
        config = _mock_config(source_scope="all")
        wf = translate(
            config,
            available_tools=["web_search", "web_crawl", "search_product_docs"],
        )

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        body = rc.config["body"]
        tool_names = {t if isinstance(t, str) else t["name"] for t in body["config"]["tools"]}
        assert "search_product_docs" in tool_names
        assert "web_search" in tool_names

    def test_enterprise_tools_propagated_to_background_node(self) -> None:
        """Enterprise tools appear in background node config."""
        config = _mock_config(source_scope="all")
        wf = translate(
            config,
            available_tools=["web_search", "web_crawl", "ask_expert"],
        )

        bg = _find_child(wf.root, "background")
        assert bg is not None
        tool_names = {t if isinstance(t, str) else t["name"] for t in bg.config["tools"]}
        assert "ask_expert" in tool_names


# ---------------------------------------------------------------------------
# Tests — System instructions injection
# ---------------------------------------------------------------------------


class TestSystemInstructions:
    """Tests for system_instructions injection into planner/researcher."""

    def test_injected_into_planner(self) -> None:
        config = _mock_config(system_instructions="Focus on Python libraries only.")
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        planner = rc.config["planner"]
        assert planner["system_prompt"] == "Focus on Python libraries only."

    def test_injected_into_researcher(self) -> None:
        config = _mock_config(system_instructions="Focus on Python libraries only.")
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        body = rc.config["body"]
        assert body["config"]["system_prompt"] == "Focus on Python libraries only."

    def test_injected_into_synthesizer_when_no_structured_prompt(self) -> None:
        config = _mock_config(system_instructions="Be concise.")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["system_prompt"] == "Be concise."

    def test_not_injected_into_synthesizer_when_structured_prompt_exists(self) -> None:
        config = _mock_config(
            system_instructions="Be concise.",
            structured_system_prompt="Custom structured prompt.",
        )
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        # structured_system_prompt takes precedence
        assert synth.config["system_prompt"] == "Custom structured prompt."

    def test_no_system_prompt_when_instructions_absent(self) -> None:
        config = _mock_config(system_instructions=None)
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        planner = rc.config["planner"]
        assert "system_prompt" not in planner


# ---------------------------------------------------------------------------
# Tests — Per-run tone + output-language threading onto the synthesizer node
# ---------------------------------------------------------------------------


class TestToneAndLanguageThreading:
    """tone/output_language reach the synthesizer node config (AgentNodeConfig)."""

    def test_tone_resolved_to_enum_on_synth_config(self) -> None:
        from databricks_deep_research.agents.config import Tone

        config = _mock_config(tone="objective")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["tone"] is Tone.OBJECTIVE

    def test_output_language_stamped_on_synth_config(self) -> None:
        config = _mock_config(output_language="Spanish")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["output_language"] == "Spanish"

    def test_unknown_tone_is_dropped_not_raised(self) -> None:
        config = _mock_config(tone="not-a-real-tone")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        # Unrecognized tone degrades to absent rather than raising.
        assert "tone" not in synth.config

    def test_absent_tone_language_keys_not_present(self) -> None:
        """Default path parity: no tone/language => keys absent from config."""
        config = _mock_config()  # tone=None, output_language=None
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert "tone" not in synth.config
        assert "output_language" not in synth.config

    def test_synth_config_parses_into_agent_node_config(self) -> None:
        """The stamped dict must validate into a framework AgentNodeConfig with
        the tone coerced to a Tone member and language preserved."""
        from databricks_deep_research.agents.config import AgentNodeConfig, Tone

        config = _mock_config(tone="formal", output_language="Japanese")
        wf = translate(config)
        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None

        node_config = AgentNodeConfig(**synth.config)
        assert node_config.tone is Tone.FORMAL
        assert node_config.output_language == "Japanese"


# ---------------------------------------------------------------------------
# Tests — Output format JSON with schema
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Tests for output_format and output_schema on synthesizer."""

    def test_default_format_is_markdown(self) -> None:
        config = _mock_config()
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        # No "output_format" key when markdown (the default)
        assert "output_format" not in synth.config

    def test_json_format_set(self) -> None:
        config = _mock_config(output_format="json")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["output_format"] == "json"

    def test_json_with_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        config = _mock_config(output_format="json", output_schema=schema)
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["output_format"] == "json"
        assert synth.config["output_model"] == schema

    def test_json_without_schema(self) -> None:
        config = _mock_config(output_format="json", output_schema=None)
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["output_format"] == "json"
        assert "output_model" not in synth.config

    def test_structured_user_prompt(self) -> None:
        config = _mock_config(structured_user_prompt="Answer in bullet points.")
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        assert synth.config["user_prompt_template"] == "Answer in bullet points."


# ---------------------------------------------------------------------------
# Tests — Manual workflow mode
# ---------------------------------------------------------------------------


class TestManualWorkflowMode:
    """Tests for manual workflow mode with preset steps."""

    def test_manual_mode_disables_evaluator(self) -> None:
        manual_steps = [{"title": "Step 1"}, {"title": "Step 2"}]
        config = _mock_config(workflow_mode="manual", manual_steps=manual_steps)
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.config["evaluator"] is None

    def test_planner_mode_keeps_evaluator(self) -> None:
        config = _mock_config(workflow_mode="planner")
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.config["evaluator"] is not None
        assert rc.config["evaluator"]["subtype"] == "reflector"

    def test_manual_mode_without_steps_keeps_evaluator(self) -> None:
        """Manual mode without manual_steps is effectively planner mode."""
        config = _mock_config(workflow_mode="manual", manual_steps=None)
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.config["evaluator"] is not None


# ---------------------------------------------------------------------------
# Tests — Pools
# ---------------------------------------------------------------------------


class TestPools:
    """Tests for pool configuration."""

    def test_default_pools(self) -> None:
        config = _mock_config()
        pools = _build_pools(config)

        assert len(pools) == 3
        names = {p["name"] for p in pools}
        assert names == {"sources", "observations", "discovery_sources"}

    def test_pools_in_workflow(self) -> None:
        config = _mock_config()
        wf = translate(config)

        assert len(wf.pools) == 3
        names = {p["name"] for p in wf.pools}
        assert names == {"sources", "observations", "discovery_sources"}

    def test_sources_pool_has_dedup_key(self) -> None:
        config = _mock_config()
        pools = _build_pools(config)

        sources_pool = next(p for p in pools if p["name"] == "sources")
        assert sources_pool["dedup_key"] == "url"

    def test_observations_pool_has_dedup_key(self) -> None:
        config = _mock_config()
        pools = _build_pools(config)

        obs_pool = next(p for p in pools if p["name"] == "observations")
        assert obs_pool["dedup_key"] == "content_hash"

    def test_pool_dicts_valid_for_pool_config(self) -> None:
        """Regression: pool dicts must pass PoolConfig(extra='forbid') validation."""
        from databricks_deep_research.pools.pool_state import PoolConfig

        config = _mock_config()
        pools = _build_pools(config)
        for pool_dict in pools:
            cfg = PoolConfig(**pool_dict)  # Must not raise
            assert cfg.name in ("sources", "observations", "discovery_sources")
            assert cfg.max_items > 0


# ---------------------------------------------------------------------------
# Tests — Research cycle internals
# ---------------------------------------------------------------------------


class TestResearchCycleConfig:
    """Tests for plan_and_execute config details."""

    def test_planner_config(self) -> None:
        config = _mock_config()
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        planner = rc.config["planner"]
        assert planner["subtype"] == "planner"
        assert planner["model_tier"] == "analytical"
        assert "query" in planner["input_keys"]
        assert "background" in planner["input_keys"]

    def test_body_researcher_config(self) -> None:
        config = _mock_config()
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        body = rc.config["body"]
        assert body["id"] == "researcher"
        assert body["type"] == "agent"
        body_config = body["config"]
        assert body_config["subtype"] == "researcher"
        assert body_config["max_tool_calls"] == 15

    def test_evaluator_is_reflector(self) -> None:
        config = _mock_config()
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        evaluator = rc.config["evaluator"]
        assert evaluator["subtype"] == "reflector"
        assert evaluator["model_tier"] == "analytical"

    def test_max_replan_cycles_from_config(self) -> None:
        config = _mock_config(max_plan_iterations=5)
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        assert rc.config["max_replan_cycles"] == 5

    def test_pool_writes_on_researcher(self) -> None:
        config = _mock_config()
        wf = translate(config)

        rc = _find_child(wf.root, "research_cycle")
        assert rc is not None
        body_config = rc.config["body"]["config"]
        pool_names = {pw["pool"] for pw in body_config["pool_writes"]}
        assert pool_names == {"observations", "sources"}

    def test_synthesizer_pool_inject(self) -> None:
        config = _mock_config()
        wf = translate(config)

        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        pool_names = {pi["pool"] for pi in synth.config["pool_inject"]}
        assert pool_names == {"observations", "sources"}


# ---------------------------------------------------------------------------
# Tests — Synthesis mode (reclaim / citation verification)
# ---------------------------------------------------------------------------


class TestSynthesisMode:
    """Grounding mode + citation config on the synthesizer.

    Contract (cheap citations always; ``verify_sources`` = NLI overlay only):
      * citations are ALWAYS generated (a grounded mode), never disabled;
      * ``verify_sources`` OR ``synthesis_mode='reclaim'`` → ``grounding_mode=
        'reclaim'`` (cite + NLI verify); otherwise → ``'classical_lite'``
        (cite-only; NLI / correction / numeric skipped);
      * ``output_schema['synthesis_mode']`` is a valid ``SynthesisMode``.
    """

    def test_default_verify_sources_enables_reclaim(self) -> None:
        """Default config has verify_sources=True → reclaim (full verify)."""
        synth = _find_child(translate(_mock_config()).root, "synthesizer")
        assert synth is not None
        assert synth.config["grounding_mode"] == "reclaim"
        assert synth.config["output_schema"]["enable_isolated_verification"] is True
        assert synth.config["output_schema"]["enable_citation_verification"] is True

    def test_explicit_reclaim_mode(self) -> None:
        """Explicit synthesis_mode='reclaim' → reclaim even when verify off."""
        config = _mock_config(synthesis_mode="reclaim", verify_sources=False)
        synth = _find_child(translate(config).root, "synthesizer")
        assert synth is not None
        assert synth.config["grounding_mode"] == "reclaim"
        assert synth.config["output_schema"]["enable_isolated_verification"] is True

    def test_verify_sources_true_implies_reclaim(self) -> None:
        """verify_sources=True → reclaim even when synthesis_mode='simple'."""
        config = _mock_config(synthesis_mode="simple", verify_sources=True)
        synth = _find_child(translate(config).root, "synthesizer")
        assert synth is not None
        assert synth.config["grounding_mode"] == "reclaim"

    def test_simple_mode_with_no_verification_is_cheap_grounding(self) -> None:
        """synthesis_mode='simple' + verify off → classical_lite, STILL cites."""
        config = _mock_config(synthesis_mode="simple", verify_sources=False)
        synth = _find_child(translate(config).root, "synthesizer")
        assert synth is not None
        assert synth.config["grounding_mode"] == "classical_lite"
        assert synth.config["output_schema"]["enable_isolated_verification"] is False
        # Citation generation is never disabled — only the NLI overlay is.
        assert synth.config["output_schema"]["enable_citation_verification"] is True

    def test_dead_post_verification_key_never_written(self) -> None:
        """``enable_post_verification`` was unread by the framework and is no
        longer written to the node config, in either grounding mode."""
        for verify in (True, False):
            config = _mock_config(verify_sources=verify, enable_post_verification=True)
            synth = _find_child(translate(config).root, "synthesizer")
            assert synth is not None
            assert "enable_post_verification" not in synth.config["output_schema"]

    @pytest.mark.parametrize(
        "synthesis_mode, verify_sources, expected_grounding",
        [
            ("simple", False, "classical_lite"),
            ("simple", True, "reclaim"),
            ("reclaim", False, "reclaim"),
            ("reclaim", True, "reclaim"),
        ],
        ids=["simple-noverify", "simple-verify", "reclaim-noverify", "reclaim-verify"],
    )
    def test_grounding_mode_truth_table(
        self, synthesis_mode: str, verify_sources: bool, expected_grounding: str
    ) -> None:
        """Parameterized truth table for grounding-mode resolution."""
        config = _mock_config(
            synthesis_mode=synthesis_mode, verify_sources=verify_sources
        )
        synth = _find_child(translate(config).root, "synthesizer")
        assert synth is not None
        assert synth.config["grounding_mode"] == expected_grounding

    def test_synth_config_valid_for_agent_node_config(self) -> None:
        """Regression: synthesizer config must pass AgentNodeConfig(extra='forbid')."""
        from databricks_deep_research.agents.config import AgentNodeConfig

        config = _mock_config()
        wf = translate(config)
        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        AgentNodeConfig(**synth.config)  # Must not raise ValidationError


# ---------------------------------------------------------------------------
# Tests — Query mode routing
# ---------------------------------------------------------------------------


class TestQueryModeRouting:
    """Tests that translate() routes correctly based on query_mode."""

    def test_default_produces_deep_research(self) -> None:
        """No query_mode → deep_research pipeline (id='deep_research', 4 children)."""
        config = _mock_config()
        wf = translate(config)

        assert wf.id == "deep_research"
        assert len(wf.root.children) == 4

    def test_simple_mode_produces_coordinator_only(self) -> None:
        """query_mode='simple' → workflow with id='simple', single coordinator child."""
        config = _mock_config(query_mode="simple")
        wf = translate(config)

        assert wf.id == "simple"
        assert len(wf.root.children) == 1
        assert wf.root.children[0].id == "coordinator"

    def test_web_search_mode_produces_lightweight(self) -> None:
        """query_mode='web_search' → workflow with id='web_search', 3 children
        (coordinator, researcher, synthesizer), no background, no plan_and_execute."""
        config = _mock_config(query_mode="web_search")
        wf = translate(config)

        assert wf.id == "web_search"
        assert len(wf.root.children) == 3
        ids = [c.id for c in wf.root.children]
        assert ids == ["coordinator", "researcher", "synthesizer"]
        # No background node
        assert _find_child(wf.root, "background") is None
        # No plan_and_execute node
        for child in wf.root.children:
            assert child.type != NodeType.plan_and_execute

    def test_web_search_researcher_has_max_tool_calls_5(self) -> None:
        """query_mode='web_search' → researcher node has max_tool_calls=5."""
        config = _mock_config(query_mode="web_search")
        wf = translate(config)

        researcher = _find_child(wf.root, "researcher")
        assert researcher is not None
        assert researcher.config["max_tool_calls"] == 5

    def test_simple_mode_has_no_pools(self) -> None:
        """query_mode='simple' → pools list is empty."""
        config = _mock_config(query_mode="simple")
        wf = translate(config)

        assert wf.pools == []

    def test_web_search_mode_has_pools(self) -> None:
        """query_mode='web_search' → pools list includes discovery sources."""
        config = _mock_config(query_mode="web_search")
        wf = translate(config)

        assert len(wf.pools) == 3

    def test_web_search_with_system_instructions(self) -> None:
        """query_mode='web_search' with system_instructions → researcher has system_prompt set."""
        config = _mock_config(
            query_mode="web_search",
            system_instructions="Only use Python sources.",
        )
        wf = translate(config)

        researcher = _find_child(wf.root, "researcher")
        assert researcher is not None
        assert researcher.config["system_prompt"] == "Only use Python sources."


# ---------------------------------------------------------------------------
# Tests — Enterprise-only edge cases (empty/None available_tools)
# ---------------------------------------------------------------------------


class TestResolveSearchToolsEnterpriseEdgeCases:
    """Tests for _resolve_search_tools with enterprise_only scope edge cases.

    Covers the fix for the falsy empty-list check (available_tools=[]
    was previously treated the same as available_tools=None).
    """

    def test_enterprise_only_with_empty_tools_list(self) -> None:
        """enterprise_only + available_tools=[] → no tools, but no crash."""
        config = _mock_config(source_scope="enterprise_only")
        tools = _resolve_search_tools(config, available_tools=[])
        assert tools == []

    def test_enterprise_only_with_none_tools(self) -> None:
        """enterprise_only + available_tools=None → no enterprise tools added."""
        config = _mock_config(source_scope="enterprise_only")
        tools = _resolve_search_tools(config, available_tools=None)
        assert tools == []

    def test_enterprise_only_with_tools(self) -> None:
        """enterprise_only + available_tools with enterprise names → tools added, no web."""
        config = _mock_config(source_scope="enterprise_only")
        tools = _resolve_search_tools(
            config,
            available_tools=["web_search", "web_crawl", "genie_crm"],
        )
        assert len(tools) == 1
        assert tools[0] == "genie_crm"

    def test_all_scope_includes_both(self) -> None:
        """all scope includes web + enterprise."""
        config = _mock_config(source_scope="all")
        tools = _resolve_search_tools(config, available_tools=["genie_crm"])
        names = [t if isinstance(t, str) else t["name"] for t in tools]
        assert "web_search" in names
        assert "web_crawl" in names
        assert "genie_crm" in names

    def test_enterprise_only_empty_list_vs_none_differ(self) -> None:
        """available_tools=[] and available_tools=None should both give no tools
        for enterprise_only, but [] enters the enterprise branch (no-op loop)
        while None skips it entirely."""
        config = _mock_config(source_scope="enterprise_only")
        tools_empty = _resolve_search_tools(config, available_tools=[])
        tools_none = _resolve_search_tools(config, available_tools=None)
        assert tools_empty == []
        assert tools_none == []

    def test_enterprise_only_web_only_tools_filtered(self) -> None:
        """enterprise_only + available_tools with only web names → no tools."""
        config = _mock_config(source_scope="enterprise_only")
        tools = _resolve_search_tools(
            config,
            available_tools=["web_search", "web_crawl"],
        )
        assert tools == []


# ---------------------------------------------------------------------------
# Tests — grounding mode: cheap citations always, verify_sources = NLI only
# ---------------------------------------------------------------------------


class TestSynthesizerGroundingMode:
    """verify_sources now selects the grounding mode (cite-vs-verify), not
    whether citations exist. Off → classical_lite (cheap, cite-only); on →
    reclaim (cite + NLI verify)."""

    def _synth(self, **overrides: Any) -> WorkflowNode:
        wf = translate(_mock_config(**overrides))
        synth = _find_child(wf.root, "synthesizer")
        assert synth is not None
        return synth

    def test_verify_off_uses_classical_lite_grounding_only(self) -> None:
        synth = self._synth(verify_sources=False, synthesis_mode="simple")
        assert synth.config["grounding_mode"] == "classical_lite"
        schema = synth.config["output_schema"]
        assert schema["enable_isolated_verification"] is False
        assert schema["enable_citation_correction"] is False
        assert schema["enable_numeric_qa_verification"] is False

    def test_verify_on_uses_reclaim_full_verification(self) -> None:
        synth = self._synth(verify_sources=True, synthesis_mode="simple")
        assert synth.config["grounding_mode"] == "reclaim"
        assert synth.config["output_schema"]["enable_isolated_verification"] is True

    def test_explicit_reclaim_mode_forces_full_verify_even_when_verify_off(
        self,
    ) -> None:
        synth = self._synth(verify_sources=False, synthesis_mode="reclaim")
        assert synth.config["grounding_mode"] == "reclaim"
        assert synth.config["output_schema"]["enable_isolated_verification"] is True

    def test_both_modes_keep_citation_generation_on(self) -> None:
        """Either way, citations are generated/linked (the floor) — only the
        NLI overlay differs. ``synthesis_mode`` stays a valid SynthesisMode."""
        for verify in (True, False):
            schema = self._synth(verify_sources=verify).config["output_schema"]
            assert schema["enable_citation_verification"] is True
            assert schema["synthesis_mode"] == "interleaved"
