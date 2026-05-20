"""Layer 2 auto-repair tests (designer-hardening plan).

Each test exercises one normalizer in isolation, then a combined end-to-end
test reproduces the exact failure shape captured at
``tests/_runs/20260518T055122-investment_research/designer/workflow.json``
to prove the normalizer would have produced a workable workflow.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.agent_designer.ast_normalizer import (
    NormalizationFix,
    normalize_ast,
)
from deep_research.agent_designer.validation_helpers import _quality_advice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ast(
    *,
    subtype: str = "researcher",
    model_tier: str = "analytical",
    tools: list[str] | None = None,
    pool_writes: Any | None = None,
    declared_tools: list[dict[str, Any]] | None = None,
    declared_pools: list[dict[str, Any]] | None = None,
    max_tool_calls: int | None = None,
) -> dict[str, Any]:
    """Build a minimal AST with one agent node + optional config."""
    config: dict[str, Any] = {
        "subtype": subtype,
        "model_tier": model_tier,
        "tools": list(tools or []),
    }
    if pool_writes is not None:
        config["pool_writes"] = pool_writes
    if max_tool_calls is not None:
        config["max_tool_calls"] = max_tool_calls
    return {
        "root": {
            "type": "agent",
            "label": "Test agent",
            "config": config,
            "children": [],
        },
        "tools": list(declared_tools or []),
        "pools": list(declared_pools or []),
    }


def _fixes_by_kind(fixes: list[NormalizationFix]) -> dict[str, list[NormalizationFix]]:
    out: dict[str, list[NormalizationFix]] = {}
    for f in fixes:
        out.setdefault(f.kind, []).append(f)
    return out


# ---------------------------------------------------------------------------
# subtype_rewrite
# ---------------------------------------------------------------------------


class TestSubtypeRewrite:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("lane_researcher", "researcher"),
            ("research_lane", "researcher"),
            ("investigator", "researcher"),
            ("analyst", "researcher"),
            ("specialist", "researcher"),
            ("summarizer", "synthesizer"),
            ("reporter", "synthesizer"),
            ("writer", "synthesizer"),
            ("evaluator", "reflector"),
            ("reviewer", "reflector"),
            ("judge", "reflector"),
            ("decomposer", "planner"),
            ("router", "coordinator"),
        ],
    )
    def test_alias_table_maps_to_known(self, raw: str, expected: str) -> None:
        ast = _make_ast(subtype=raw)
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["subtype"] == expected
        kinds = _fixes_by_kind(fixes)
        assert "subtype_rewrite" in kinds
        assert kinds["subtype_rewrite"][0].before == raw
        assert kinds["subtype_rewrite"][0].after == expected

    def test_unknown_subtype_falls_back_to_researcher(self) -> None:
        ast = _make_ast(subtype="quantum_consultant")
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["subtype"] == "researcher"
        kinds = _fixes_by_kind(fixes)
        assert kinds["subtype_rewrite"][0].after == "researcher"

    def test_known_subtype_passes_through(self) -> None:
        ast = _make_ast(subtype="researcher")
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        # No subtype rewrite emitted for an already-valid subtype.
        assert "subtype_rewrite" not in kinds
        assert new["root"]["config"]["subtype"] == "researcher"


# ---------------------------------------------------------------------------
# tier_rewrite
# ---------------------------------------------------------------------------


class TestTierRewrite:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("standard", "analytical"),
            ("default", "analytical"),
            ("balanced", "analytical"),
            ("reasoning", "complex"),
            ("deep", "complex"),
            ("lite", "simple"),
            ("light", "simple"),
        ],
    )
    def test_alias_table(self, raw: str, expected: str) -> None:
        ast = _make_ast(model_tier=raw)
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["model_tier"] == expected
        kinds = _fixes_by_kind(fixes)
        assert kinds["tier_rewrite"][0].before == raw
        assert kinds["tier_rewrite"][0].after == expected

    def test_unknown_tier_uses_subtype_fallback(self) -> None:
        ast = _make_ast(subtype="synthesizer", model_tier="ultra_max")
        new, fixes = normalize_ast(ast)
        # synthesizer fallback → complex
        assert new["root"]["config"]["model_tier"] == "complex"
        kinds = _fixes_by_kind(fixes)
        assert kinds["tier_rewrite"][0].after == "complex"

    def test_known_tier_passes_through(self) -> None:
        ast = _make_ast(model_tier="complex")
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "tier_rewrite" not in kinds


# ---------------------------------------------------------------------------
# tool_kind_rewrite
# ---------------------------------------------------------------------------


class TestToolKindRewrite:
    def test_search_alias_to_web_search(self) -> None:
        ast = _make_ast(declared_tools=[{"kind": "search", "name": "s", "config": {}}])
        new, fixes = normalize_ast(ast)
        assert new["tools"][0]["kind"] == "web_search"
        kinds = _fixes_by_kind(fixes)
        assert kinds["tool_kind_rewrite"][0].before == "search"

    def test_canonical_kind_passes_through(self) -> None:
        ast = _make_ast(
            declared_tools=[{"kind": "web_search", "name": "s", "config": {}}]
        )
        _, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "tool_kind_rewrite" not in kinds


# ---------------------------------------------------------------------------
# auto_bind_retrieval
# ---------------------------------------------------------------------------


class TestAutoBindRetrieval:
    def test_empty_tools_gets_merged_research_pair(self) -> None:
        """Auto-bind uses the MERGED pair (web_research + web_crawl) so the
        researcher gets real source bodies on the first call. web_crawl
        stays for selective candidate follow-up."""
        ast = _make_ast(subtype="researcher", tools=[])
        new, fixes = normalize_ast(ast)
        bound = new["root"]["config"]["tools"]
        assert "web_research" in bound and "web_crawl" in bound
        # Tools should also be declared at top level.
        declared_names = {t["name"] for t in new["tools"]}
        assert {"web_research", "web_crawl"} <= declared_names
        kinds = _fixes_by_kind(fixes)
        assert "auto_bind_retrieval" in kinds

    def test_existing_retrieval_tool_left_alone(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["vector_search"],
            declared_tools=[
                {"kind": "vector_search", "name": "vector_search", "config": {}}
            ],
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["tools"] == ["vector_search"]
        kinds = _fixes_by_kind(fixes)
        assert "auto_bind_retrieval" not in kinds

    def test_non_researcher_subtype_skipped(self) -> None:
        ast = _make_ast(subtype="synthesizer", tools=[])
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["tools"] == []
        kinds = _fixes_by_kind(fixes)
        assert "auto_bind_retrieval" not in kinds


# ---------------------------------------------------------------------------
# pool_spec_rewrite
# ---------------------------------------------------------------------------


class TestPoolSpecRewrite:
    def test_pool_write_fields_shape_gets_schema_extracts(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            pool_writes=[
                {"pool": "sources", "fields": ["url", "title"]},
                {
                    "pool": "observations",
                    "fields": ["content_hash", "content", "source_url"],
                },
            ],
        )
        ast["root"]["config"]["output_key"] = "lane_output"

        new, fixes = normalize_ast(ast)

        assert new["root"]["config"]["pool_writes"] == [
            {"pool": "sources", "extract": "sources"},
            {"pool": "observations", "extract": "lane_output"},
        ]
        kinds = _fixes_by_kind(fixes)
        assert "pool_spec_rewrite" in kinds

    def test_pool_write_string_gets_schema_extract(self) -> None:
        ast = _make_ast(subtype="researcher", pool_writes="sources")

        new, fixes = normalize_ast(ast)

        assert new["root"]["config"]["pool_writes"] == [
            {"pool": "sources", "extract": "sources"}
        ]
        kinds = _fixes_by_kind(fixes)
        assert "pool_spec_rewrite" in kinds

    def test_pool_inject_string_and_name_alias_get_schema_pool(self) -> None:
        ast = _make_ast(subtype="synthesizer")
        ast["root"]["config"]["pool_inject"] = [
            "sources",
            {"name": "observations", "format": "json", "fields": ["content"]},
        ]

        new, fixes = normalize_ast(ast)

        assert new["root"]["config"]["pool_inject"] == [
            {"pool": "sources"},
            {"format": "json", "pool": "observations"},
        ]
        kinds = _fixes_by_kind(fixes)
        assert "pool_spec_rewrite" in kinds


# ---------------------------------------------------------------------------
# auto_declare_pool
# ---------------------------------------------------------------------------


class TestAutoDeclarePool:
    def test_referenced_pool_auto_declared(self) -> None:
        ast = _make_ast(subtype="researcher", pool_writes="sources")
        new, fixes = normalize_ast(ast)
        names = {p["name"] for p in new["pools"]}
        assert "sources" in names
        kinds = _fixes_by_kind(fixes)
        assert "auto_declare_pool" in kinds
        assert kinds["auto_declare_pool"][0].after["name"] == "sources"

    def test_declared_pool_passes_through(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            pool_writes="sources",
            declared_pools=[
                {"name": "sources", "dedup_key": "url", "max_items": 50}
            ],
        )
        new, fixes = normalize_ast(ast)
        # Pool kept its original config; no auto-declare fix.
        sources_pool = next(p for p in new["pools"] if p["name"] == "sources")
        assert sources_pool["max_items"] == 50
        kinds = _fixes_by_kind(fixes)
        assert "auto_declare_pool" not in kinds


# ---------------------------------------------------------------------------
# set_minimum_max_tool_calls
# ---------------------------------------------------------------------------


class TestMaxToolCallsFloor:
    def test_none_floor_for_researcher_is_6(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search"],
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}}
            ],
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["max_tool_calls"] == 6
        kinds = _fixes_by_kind(fixes)
        assert "set_minimum_max_tool_calls" in kinds

    def test_zero_floor_replaced(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search"],
            max_tool_calls=0,
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}}
            ],
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["max_tool_calls"] == 6
        kinds = _fixes_by_kind(fixes)
        assert kinds["set_minimum_max_tool_calls"][0].before == 0

    def test_existing_positive_value_preserved(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search"],
            max_tool_calls=12,
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}}
            ],
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["max_tool_calls"] == 12
        kinds = _fixes_by_kind(fixes)
        assert "set_minimum_max_tool_calls" not in kinds

    def test_no_tools_no_budget_fix(self) -> None:
        """Agents without tools don't need a max_tool_calls floor."""
        ast = _make_ast(subtype="synthesizer", tools=[])
        _, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "set_minimum_max_tool_calls" not in kinds


# ---------------------------------------------------------------------------
# brace_escape — Jinja-escape literal { } in prompts (JSON-schema examples)
# ---------------------------------------------------------------------------


class TestBraceEscape:
    """The architect routinely embeds JSON output-shape examples in prompts.
    SafeTemplateRenderer rejects these because the literal `{` looks like a
    template variable. Layer 2 escapes them to {{ }} so the runner accepts
    the prompt verbatim."""

    def test_literal_json_braces_escaped(self) -> None:
        ast = _make_ast(subtype="coordinator")
        ast["root"]["config"]["user_prompt_template"] = (
            'Analyze {query}. Output: {"score": 0.5, "reasoning": "x"}'
        )
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "brace_escape" in kinds
        out = new["root"]["config"]["user_prompt_template"]
        assert '{query}' in out  # legitimate template var preserved
        assert '{{"score"' in out  # literal brace escaped
        assert '}}' in out

    def test_already_escaped_braces_untouched(self) -> None:
        ast = _make_ast(subtype="researcher")
        ast["root"]["config"]["system_prompt"] = (
            "Use {{var}} for templating; {{not}} touched."
        )
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "brace_escape" not in kinds
        assert (
            new["root"]["config"]["system_prompt"]
            == "Use {{var}} for templating; {{not}} touched."
        )

    def test_multiline_json_schema_escaped(self) -> None:
        """The exact failure shape captured from the runner trace."""
        ast = _make_ast(subtype="coordinator")
        ast["root"]["config"]["user_prompt_template"] = (
            'Return JSON: {\n  "complexity": "simple" | "moderate" | "complex",\n'
            '  "is_simple_query": false,\n  "extracted_scope": {\n'
            '    "entities": ["<Company>", "<TICKER>"]\n  }\n}'
        )
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "brace_escape" in kinds
        out = new["root"]["config"]["user_prompt_template"]
        # Two open braces (outer JSON + nested extracted_scope) → escaped.
        # Two matching close braces → escaped.
        assert out.count("{{") == 2
        assert out.count("}}") == 2
        # Every `{` is part of a `{{` (no unescaped literal braces left).
        for ch_idx, ch in enumerate(out):
            if ch == "{":
                assert out[ch_idx + 1] == "{" or (ch_idx > 0 and out[ch_idx - 1] == "{")

    def test_no_braces_no_fix(self) -> None:
        ast = _make_ast(subtype="researcher")
        ast["root"]["config"]["system_prompt"] = "Plain prompt without any braces."
        _, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "brace_escape" not in kinds

    def test_only_template_vars_no_fix(self) -> None:
        ast = _make_ast(subtype="researcher")
        ast["root"]["config"]["user_prompt_template"] = (
            "Research {query} for {company} and respond with {answer}."
        )
        _, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "brace_escape" not in kinds


# ---------------------------------------------------------------------------
# tool_consolidation — rewrite [web_search, ...] → [web_research, web_crawl]
# ---------------------------------------------------------------------------


class TestToolConsolidation:
    """Layer 2 consolidation rewrites legacy web_search/web_crawl bindings to
    the merged web_research + web_crawl pair. Eliminates the LLM-orchestration
    burden where the architect could forget to call web_crawl after search."""

    def test_legacy_pair_rewritten_to_merged(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search", "web_crawl"],
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            max_tool_calls=6,
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["tools"] == ["web_research", "web_crawl"]
        kinds = _fixes_by_kind(fixes)
        assert "tool_consolidation" in kinds
        # web_research auto-declared at top level
        assert any(t["name"] == "web_research" for t in new["tools"])

    def test_lonely_web_search_gets_crawl_companion(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search"],
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}}
            ],
            max_tool_calls=6,
        )
        new, fixes = normalize_ast(ast)
        assert new["root"]["config"]["tools"] == ["web_research", "web_crawl"]
        kinds = _fixes_by_kind(fixes)
        assert "tool_consolidation" in kinds

    def test_canonical_pair_passes_through(self) -> None:
        """If the architect already used [web_research, web_crawl] there's
        nothing to consolidate — no fix emitted."""
        ast = _make_ast(
            subtype="researcher",
            tools=["web_research", "web_crawl"],
            declared_tools=[
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            max_tool_calls=6,
        )
        new, fixes = normalize_ast(ast)
        kinds = _fixes_by_kind(fixes)
        assert "tool_consolidation" not in kinds

    def test_canonical_pair_with_stale_web_search_is_cleaned(self) -> None:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_research", "web_crawl", "web_search"],
            declared_tools=[
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
                {"kind": "web_search", "name": "web_search", "config": {}},
            ],
            max_tool_calls=6,
        )

        new, fixes = normalize_ast(ast)

        assert new["root"]["config"]["tools"] == ["web_research", "web_crawl"]
        kinds = _fixes_by_kind(fixes)
        assert "tool_consolidation" in kinds

    def test_non_research_subtype_skipped(self) -> None:
        """Synthesizers shouldn't get web_research even if they happen to
        have web_search bound (unusual but defensive)."""
        ast = _make_ast(
            subtype="synthesizer",
            tools=["web_search"],
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}}
            ],
        )
        new, fixes = normalize_ast(ast)
        # Subtype is synthesizer, so _auto_bind_retrieval skips it.
        # Consolidation still applies to any agent type — design choice.
        assert new["root"]["config"]["tools"] == ["web_research", "web_crawl"]
        kinds = _fixes_by_kind(fixes)
        assert "tool_consolidation" in kinds

    def test_preserves_other_tools(self) -> None:
        """A researcher with [web_search, vector_search] keeps vector_search."""
        ast = _make_ast(
            subtype="researcher",
            tools=["web_search", "vector_search"],
            declared_tools=[
                {"kind": "web_search", "name": "web_search", "config": {}},
                {"kind": "vector_search", "name": "vector_search", "config": {}},
            ],
            max_tool_calls=6,
        )
        new, fixes = normalize_ast(ast)
        tools_after = new["root"]["config"]["tools"]
        assert "vector_search" in tools_after
        assert "web_research" in tools_after
        assert "web_crawl" in tools_after
        assert "web_search" not in tools_after


# ---------------------------------------------------------------------------
# synthesizer grounding defaults — generic evidence-pool safety floor
# ---------------------------------------------------------------------------


class TestSynthesizerGroundingDefaults:
    def _evidence_pool_ast(
        self,
        *,
        synthesizer_config: dict[str, Any] | None = None,
        root_type: str = "sequence",
    ) -> dict[str, Any]:
        researcher = {
            "id": "researcher",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "model_tier": "analytical",
                "tools": ["web_research", "web_crawl"],
                "max_tool_calls": 6,
                "pool_writes": [
                    {"pool": "observations", "extract": "findings"},
                    {"pool": "sources", "extract": "sources"},
                ],
            },
            "children": [],
        }
        synthesizer = {
            "id": "synthesizer",
            "type": "agent",
            "label": "Synthesizer",
            "config": {
                "subtype": "synthesizer",
                "model_tier": "complex",
                "output_key": "report",
                **(synthesizer_config or {}),
            },
            "children": [],
        }
        if root_type == "plan_and_execute":
            root = {
                "id": "main",
                "type": "sequence",
                "label": "Main",
                "config": {},
                "children": [
                    {
                        "id": "plan",
                        "type": "plan_and_execute",
                        "label": "Plan",
                        "config": {"body": researcher},
                        "children": [],
                    },
                    synthesizer,
                ],
            }
        else:
            root = {
                "id": "main",
                "type": "sequence",
                "label": "Main",
                "config": {},
                "children": [researcher, synthesizer],
            }
        return {
            "root": root,
            "tools": [
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            "pools": [
                {"name": "observations", "dedup_key": "content_hash"},
                {"name": "sources", "dedup_key": "url"},
            ],
        }

    def _synth_config(self, ast: dict[str, Any]) -> dict[str, Any]:
        return ast["root"]["children"][-1]["config"]

    def test_missing_grounding_gets_reclaim_contract(self) -> None:
        ast = self._evidence_pool_ast()

        new, fixes = normalize_ast(ast)

        config = self._synth_config(new)
        assert config["grounding_mode"] == "reclaim"
        assert config["pool_inject"] == [
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ]
        assert config["output_schema"]["claim_disposition"] == {
            "abstained": "remove"
        }
        kinds = _fixes_by_kind(fixes)
        assert "synthesizer_grounding_default" in kinds
        assert "synthesizer_pool_inject_default" in kinds
        assert "synthesizer_output_schema_default" in kinds

    def test_plan_and_execute_topology_gets_same_contract(self) -> None:
        ast = self._evidence_pool_ast(root_type="plan_and_execute")

        new, _ = normalize_ast(ast)

        config = self._synth_config(new)
        assert config["grounding_mode"] == "reclaim"
        assert {item["pool"] for item in config["pool_inject"]} == {
            "observations",
            "sources",
        }

    def test_explicit_none_preserved_as_opt_out(self) -> None:
        ast = self._evidence_pool_ast(
            synthesizer_config={"grounding_mode": "none"}
        )

        new, fixes = normalize_ast(ast)

        config = self._synth_config(new)
        assert config["grounding_mode"] == "none"
        assert "output_schema" not in config
        kinds = _fixes_by_kind(fixes)
        assert "synthesizer_grounding_default" not in kinds
        assert "synthesizer_output_schema_default" not in kinds

    def test_no_evidence_pools_no_grounding_default(self) -> None:
        ast = _make_ast(subtype="synthesizer")

        new, fixes = normalize_ast(ast)

        assert "grounding_mode" not in new["root"]["config"]
        kinds = _fixes_by_kind(fixes)
        assert "synthesizer_grounding_default" not in kinds


# ---------------------------------------------------------------------------
# researcher prompt contract — generic out-of-box lane safety floor
# ---------------------------------------------------------------------------


class TestResearcherPromptContract:
    def _researcher_prompt_ast(self, template: str) -> dict[str, Any]:
        ast = _make_ast(
            subtype="researcher",
            tools=["web_research", "web_crawl"],
            declared_tools=[
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            max_tool_calls=6,
        )
        ast["root"]["id"] = "lane_6-researcher"
        ast["root"]["label"] = "Competitor analysis and relative positioning"
        ast["root"]["config"]["system_prompt"] = (
            "## Lane Specialization\n"
            "You are a domain-specialized researcher. Focus on the lane's "
            "assigned evidence questions, source quality, conflicts, and gaps. "
            "Return only findings supported by retrieved evidence and mark "
            "unknowns explicitly."
        )
        ast["root"]["config"]["user_prompt_template"] = template
        return ast

    def test_augments_partial_designer_prompt_without_replacing_substance(self) -> None:
        ast = self._researcher_prompt_ast(
            "Conduct a comprehensive competitor analysis for the specified company. "
            "Address these sub-questions:\n"
            "1. Who are the primary direct competitors and how do they compare?\n"
            "2. What are the company's competitive advantages and disadvantages?\n"
            "3. How does the company's valuation compare to competitors?\n"
            "4. What strategic moves are competitors making?\n"
            "5. Is the company gaining or losing market share?\n\n"
            "Output Sections:\n"
            "- Competitive Landscape Overview\n"
            "- Financial Benchmarking vs Peers\n"
            "- Competitive Advantages & Vulnerabilities\n"
            "- Relative Valuation Analysis\n"
            "- Strategic Positioning Assessment"
        )

        new, fixes = normalize_ast(ast)

        template = new["root"]["config"]["user_prompt_template"]
        assert template.startswith("## Investigation Brief")
        assert "You are investigating: **{query}**" in template
        assert "Conduct a comprehensive competitor analysis" in template
        assert "### Search strategy" in template
        assert "Data unavailable" in template
        assert "DO NOT improvise" in template
        kinds = _fixes_by_kind(fixes)
        assert "researcher_prompt_contract" in kinds

        prompt_advice = [
            item for item in _quality_advice(new)
            if item["path"] == "root.config.user_prompt_template"
        ]
        assert prompt_advice == []

    def test_wraps_substantive_mutated_prompt_missing_output_bullets(self) -> None:
        ast = self._researcher_prompt_ast(
            "Run a focused lane investigation for the assigned company and "
            "workstream. Use the lane's specialized system prompt as the "
            "semantic source of truth and do not substitute a generic overview.\n\n"
            "### Sub-questions\n"
            "1. Which current source materials are most authoritative for this lane?\n"
            "2. What concrete facts and dates do those sources establish?\n"
            "3. Which metrics or events materially change the lane interpretation?\n"
            "4. What conflicts or data gaps remain after retrieval?\n"
            "5. What final-report implications are supported by evidence?\n\n"
            "### Required output structure\n"
            "Return JSON fields named findings, evidence, caveats, and sources. "
            "Mark unresolved items Data unavailable and never invent figures.\n\n"
            "### Search strategy\n"
            "planned_queries should combine the user query with lane-specific "
            "terms, exact source names, and current date or period qualifiers."
        )

        new, fixes = normalize_ast(ast)

        template = new["root"]["config"]["user_prompt_template"]
        assert template.startswith("## Investigation Brief")
        assert "### Required output structure" in template
        assert "- **Evidence summary**" in template
        assert "- **Analysis and implications**" in template
        assert "- **Unknowns and caveats**" in template
        assert "### Designer-authored lane brief" in template
        assert "Return JSON fields named findings" in template
        kinds = _fixes_by_kind(fixes)
        assert "researcher_prompt_contract" in kinds
        assert "missing=output_bullets" in kinds["researcher_prompt_contract"][0].before

        prompt_advice = [
            item for item in _quality_advice(new)
            if item["path"] == "root.config.user_prompt_template"
        ]
        assert prompt_advice == []

    def test_empty_researcher_prompt_is_left_for_designer_advice(self) -> None:
        ast = self._researcher_prompt_ast("")

        new, fixes = normalize_ast(ast)

        template = new["root"]["config"]["user_prompt_template"]
        assert template == ""
        prompt_advice = [
            item for item in _quality_advice(new)
            if item["path"] == "root.config.user_prompt_template"
        ]
        assert prompt_advice
        kinds = _fixes_by_kind(fixes)
        assert "researcher_prompt_contract" not in kinds

    def test_static_parallel_lane_removes_plan_execute_prompt_variables(self) -> None:
        ast = {
            "root": {
                "id": "main",
                "type": "sequence",
                "label": "Main",
                "config": {},
                "children": [
                    {
                        "id": "parallel-lanes",
                        "type": "parallel",
                        "label": "Parallel Lanes",
                        "config": {},
                        "children": [
                            {
                                "id": "lane_fundamentals-researcher",
                                "type": "agent",
                                "label": "Lane: Fundamentals & Valuation",
                                "config": {
                                    "subtype": "researcher",
                                    "model_tier": "analytical",
                                    "tools": ["web_research", "web_crawl"],
                                    "max_tool_calls": 12,
                                    "input_keys": [
                                        "query",
                                        "coordination",
                                        "page_contents",
                                        "previous_observations",
                                        "search_results",
                                        "step_description",
                                        "step_title",
                                        "step_type",
                                    ],
                                    "system_prompt": "Specialized lane prompt. " * 30,
                                    "user_prompt_template": (
                                        "Execute the FUNDAMENTALS lane for {company_or_ticker}.\n\n"
                                        "Coordination/scope context:\n{coordination}\n\n"
                                        "Step details: {step_title} -- "
                                        "{step_description} ({step_type})\n"
                                        "Prior observations: {previous_observations}\n"
                                        "Research evidence: {search_results}\n"
                                        "Page contents: {page_contents}\n\n"
                                        "Produce a fundamentals brief addressing revenue, "
                                        "margin, earnings, free cash flow, valuation "
                                        "multiples versus peers and history, balance "
                                        "sheet leverage, liquidity, capital allocation, "
                                        "quality of earnings, and accounting red flags. "
                                        "Return evidence-backed findings, caveats, and "
                                        "source notes only."
                                    ),
                                },
                                "children": [],
                            }
                        ],
                    }
                ],
            },
            "tools": [
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            "pools": [],
        }

        new, fixes = normalize_ast(ast)

        config = new["root"]["children"][0]["children"][0]["config"]
        assert config["input_keys"] == ["query", "coordination"]
        template = config["user_prompt_template"]
        assert "{company_or_ticker}" not in template
        assert "for {query}" in template
        for forbidden in (
            "{step_title}",
            "{step_description}",
            "{step_type}",
            "{previous_observations}",
            "{search_results}",
            "{page_contents}",
        ):
            assert forbidden not in template
        assert template.startswith("## Investigation Brief")
        assert "### Designer-authored lane brief" in template
        assert "Produce a fundamentals brief" in template
        kinds = _fixes_by_kind(fixes)
        assert "static_parallel_lane_prompt" in kinds
        assert "static_parallel_lane_inputs" in kinds
        assert "researcher_prompt_contract" in kinds

    def test_plan_execute_researcher_keeps_step_variables(self) -> None:
        ast = {
            "root": {
                "id": "main",
                "type": "plan_and_execute",
                "label": "Plan",
                "config": {
                    "body": {
                        "id": "lane_body",
                        "type": "agent",
                        "label": "Adaptive Lane",
                        "config": {
                            "subtype": "researcher",
                            "model_tier": "analytical",
                            "tools": ["web_research", "web_crawl"],
                            "max_tool_calls": 12,
                            "input_keys": ["query", "current_step", "research_plan"],
                            "system_prompt": "Adaptive lane prompt. " * 30,
                            "user_prompt_template": (
                                "Execute the following research step:\n"
                                "Title: {step_title}\n"
                                "Description: {step_description}\n"
                            ),
                        },
                        "children": [],
                    }
                },
                "children": [],
            },
            "tools": [
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
            "pools": [],
        }

        new, fixes = normalize_ast(ast)

        config = new["root"]["config"]["body"]["config"]
        assert config["input_keys"] == ["query", "current_step", "research_plan"]
        assert "{step_title}" in config["user_prompt_template"]
        kinds = _fixes_by_kind(fixes)
        assert "static_parallel_lane_prompt" not in kinds
        assert "static_parallel_lane_inputs" not in kinds

    def test_config_error_handling_is_lifted_to_node_level(self) -> None:
        ast = _make_ast(subtype="researcher")
        ast["root"]["config"]["error_handling"] = {
            "on_error": "skip",
            "max_retries": 1,
        }

        new, fixes = normalize_ast(ast)

        assert "error_handling" not in new["root"]["config"]
        assert new["root"]["error_handling"] == {
            "on_error": "skip",
            "max_retries": 1,
        }
        kinds = _fixes_by_kind(fixes)
        assert "error_handling_lift" in kinds


# ---------------------------------------------------------------------------
# End-to-end: combined-defect AST (the captured bug shape)
# ---------------------------------------------------------------------------


class TestCombinedDefects:
    def test_captured_bug_produces_workable_workflow(self) -> None:
        """Mirrors the artifact at tests/_runs/20260518T055122-investment_research/
        designer/workflow.json — Opus emitted lane_researcher + standard tier +
        empty tools + referenced-but-undeclared sources pool."""
        ast: dict[str, Any] = {
            "root": {
                "type": "plan_and_execute",
                "label": "Investment lanes",
                "config": {
                    "body": {
                        "type": "parallel",
                        "label": "Lanes",
                        "children": [
                            {
                                "type": "agent",
                                "label": "Fundamentals",
                                "config": {
                                    "subtype": "lane_researcher",
                                    "model_tier": "standard",
                                    "tools": [],
                                    "pool_writes": "sources",
                                },
                            },
                            {
                                "type": "agent",
                                "label": "Risk",
                                "config": {
                                    "subtype": "lane_researcher",
                                    "model_tier": "standard",
                                    "tools": [],
                                    "pool_writes": "sources",
                                },
                            },
                        ],
                    },
                },
                "children": [],
            },
            "tools": [],
            "pools": [],
        }
        new, fixes = normalize_ast(ast)
        # All four major defect classes get fixed.
        kinds = _fixes_by_kind(fixes)
        assert "subtype_rewrite" in kinds
        assert "tier_rewrite" in kinds
        assert "auto_bind_retrieval" in kinds
        assert "auto_declare_pool" in kinds
        assert "set_minimum_max_tool_calls" in kinds
        # Both lane researchers got normalized.
        lanes = new["root"]["config"]["body"]["children"]
        for lane in lanes:
            assert lane["config"]["subtype"] == "researcher"
            assert lane["config"]["model_tier"] == "analytical"
            # Merged web_research is now the auto-bound retrieval tool.
            assert "web_research" in lane["config"]["tools"]
            assert "web_crawl" in lane["config"]["tools"]
            assert lane["config"]["max_tool_calls"] == 6
        # 'sources' pool now declared.
        assert any(p["name"] == "sources" for p in new["pools"])

    def test_input_not_mutated(self) -> None:
        """normalize_ast must deep-copy; the caller's AST stays untouched."""
        ast = _make_ast(subtype="lane_researcher", model_tier="standard")
        snapshot = copy.deepcopy(ast)
        normalize_ast(ast)
        assert ast == snapshot, "input AST was mutated"

    def test_clean_ast_produces_zero_fixes(self) -> None:
        """An AST the architect built correctly first time emits no fixes —
        the common case once Layer 4 prompt guardrails settle in.

        'Clean' now means the architect used the merged web_research +
        web_crawl pair, not the legacy web_search/web_crawl split."""
        ast = _make_ast(
            subtype="researcher",
            model_tier="analytical",
            tools=["web_research", "web_crawl"],
            max_tool_calls=10,
            declared_tools=[
                {"kind": "web_research", "name": "web_research", "config": {}},
                {"kind": "web_crawl", "name": "web_crawl", "config": {}},
            ],
        )
        ast["root"]["config"]["user_prompt_template"] = (
            "## Investigation Brief\n\n"
            "You are investigating: **{query}**\n\n"
            "### Sub-questions\n"
            "1. Which authoritative sources address the request?\n"
            "2. What citeable facts answer the core question?\n"
            "3. Which constraints or exceptions affect the answer?\n"
            "4. Where do sources conflict or leave gaps?\n"
            "5. Which findings are strong enough for the final deliverable?\n\n"
            "### Required output structure\n"
            "- **Evidence-backed findings**: source-backed facts.\n"
            "- **Coverage and conflicts**: agreements, disagreements, and gaps.\n"
            "- **Unsupported items**: unavailable or weakly supported claims.\n\n"
            "### Search strategy\n"
            "- Search for primary and authoritative sources.\n"
            "- Retrieve source text before relying on metadata.\n\n"
            "### Definition of done\n"
            "Mark missing evidence as \"Data unavailable\" -- DO NOT improvise."
        )
        _, fixes = normalize_ast(ast)
        assert fixes == []

    def test_non_dict_input_returns_empty(self) -> None:
        new, fixes = normalize_ast("not a dict")  # type: ignore[arg-type]
        assert new == "not a dict"
        assert fixes == []

    def test_fix_to_dict_round_trips(self) -> None:
        ast = _make_ast(subtype="lane_researcher")
        _, fixes = normalize_ast(ast)
        fix = fixes[0]
        d = fix.to_dict()
        assert set(d.keys()) == {"kind", "path", "before", "after", "rationale"}
