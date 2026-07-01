"""Unit tests for workflow_builder.py focused on pool defaults."""
from __future__ import annotations

import pytest

from deep_research.agent_designer.designer_types import (
    ToolBindingSpec,
    ToolDeclarationSpec,
    ToolPlan,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.workflow_builder import (
    _fallback_lane_user_prompt_template,
    _infer_ambiguity_axes,
    _is_corpus_only_assets,
    _plan_execute_synthesizer_directive,
    _query_diversification_block,
    _sources_dedup_key,
    _synthesizer_lane_coverage_directive,
    _tool_plan_bindings,
    _with_lane_user_prompt_contract,
    build_web_research_workflow,
)


def _corpus_assets() -> list[dict]:
    return [
        {"kind": "vector_index", "full_name": "main.foo.idx"},
        {"kind": "delta_table", "full_name": "main.foo.tbl"},
    ]


def _walk_nodes(node: dict):
    yield node
    config = node.get("config") if isinstance(node.get("config"), dict) else {}
    for nested_key in ("body", "evaluator", "planner"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            yield from _walk_nodes(nested)
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk_nodes(child)


def _web_brief() -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="web-only",
        topology="plan_and_execute",
        research_lanes=[
            {
                "description": "Web research lane.",
                "user_prompt_template": "Research {query}.",
            }
        ],
        tool_plan=ToolPlan(
            tools=[ToolDeclarationSpec(name="web", kind="web_search")]
        ),
    )


def _contract_brief() -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="contract",
        topology="plan_and_execute",
        user_goal="OfficeQA Treasury fiscal calendar compute answer",
        research_lanes=[
            {
                "description": "retrieve Treasury chunks then compute totals",
                "user_prompt_template": "Investigate {query}.",
            }
        ],
        tool_plan=ToolPlan(
            tools=[
                ToolDeclarationSpec(name="vector_search", kind="vector_search"),
                ToolDeclarationSpec(name="table_read", kind="table_read"),
                ToolDeclarationSpec(name="compute", kind="compute"),
            ],
            bindings=[
                ToolBindingSpec(
                    node_id="all_researchers",
                    tool_names=["vector_search", "table_read", "compute"],
                )
            ],
        ),
        tool_contract={
            "schema": "resolved_tool_contract.v1",
            "source": "prompt_grounding",
            "evidence_policy": "corpus_only",
            "required_capabilities": [
                "vector_search",
                "table_read",
                "compute",
            ],
            "ready_tool_kinds": ["vector_search", "table_read", "compute"],
            "prompt_obligations": {
                "required_terms": [
                    "officeqa",
                    "treasury",
                    "fiscal",
                    "calendar",
                    "compute",
                ],
                "synthesis_obligations": [
                    "Preserve the fiscal/calendar-year distinction."
                ],
                "planner_obligations": [
                    "Use named Databricks corpus resources before synthesis."
                ],
                "forbidden_tool_kinds": [
                    "web_search",
                    "web_crawl",
                    "web_research",
                ],
            },
        },
    )


def _contract_brief_without_forbidden_web() -> WorkflowDesignBrief:
    brief = _contract_brief()
    assert brief.tool_contract is not None
    brief.tool_contract.prompt_obligations.forbidden_tool_kinds = []
    brief.tool_contract.prompt_obligations.planner_obligations = [
        "Use named Databricks corpus resources before synthesis."
    ]
    return brief


def test_sources_dedup_key_corpus_only() -> None:
    assert _sources_dedup_key(_corpus_assets(), ToolPlan()) == "chunk_id"


def test_sources_dedup_key_web_signal_overrides_assets() -> None:
    assert (
        _sources_dedup_key(
            _corpus_assets(),
            ToolPlan(tools=[ToolDeclarationSpec(name="w", kind="web_search")]),
        )
        == "url"
    )


def test_sources_dedup_key_no_assets_no_tool_plan() -> None:
    assert _sources_dedup_key(None, None) == "url"


def test_sources_dedup_key_handles_missing_tool_plan() -> None:
    assert _sources_dedup_key(_corpus_assets(), None) == "chunk_id"


def _make_corpus_lane() -> dict:
    return {
        "description": "Treasury evidence retrieval.",
        "user_prompt_template": "Find evidence for {query}.",
    }


def test_build_workflow_plan_and_execute_corpus_chunk_id() -> None:
    brief = WorkflowDesignBrief(
        workflow_name="treasury",
        topology="plan_and_execute",
        research_lanes=[_make_corpus_lane()],
    )
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury",
        design_brief=brief,
        assets=_corpus_assets(),
    )
    pools = ast["pools"]
    sources_pool = next(p for p in pools if p["name"] == "sources")
    assert sources_pool["dedup_key"] == "chunk_id"


def test_build_workflow_parallel_lanes_corpus_chunk_id() -> None:
    brief = WorkflowDesignBrief(
        workflow_name="treasury",
        topology="parallel_lanes",
        research_lanes=[_make_corpus_lane()],
    )
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury",
        design_brief=brief,
        assets=_corpus_assets(),
    )
    sources_pool = next(p for p in ast["pools"] if p["name"] == "sources")
    assert sources_pool["dedup_key"] == "chunk_id"


def test_build_workflow_web_brief_uses_url_dedup() -> None:
    ast = build_web_research_workflow(
        intent="web research",
        design_brief=_web_brief(),
        assets=None,
    )
    sources_pool = next(p for p in ast["pools"] if p["name"] == "sources")
    assert sources_pool["dedup_key"] == "url"


def test_contract_core_prompt_precedes_designer_goal_and_passes_detector() -> None:
    from deep_research.agent_designer.semantic_validation import (
        detect_generic_synthesizer_prompt,
    )

    ast = build_web_research_workflow(
        intent="OfficeQA Treasury fiscal calendar compute answer",
        design_brief=_contract_brief(),
        assets=_corpus_assets(),
    )

    synthesizers = [
        node
        for node in _walk_nodes(ast["root"])
        if (node.get("config") or {}).get("subtype") == "synthesizer"
    ]
    assert synthesizers
    system_prompt = synthesizers[0]["config"]["system_prompt"]
    assert system_prompt.index("Workflow-Specific Evidence Contract") < system_prompt.index(
        "Designer Goal"
    )
    core = system_prompt.split("## Designer Goal", 1)[0].lower()
    assert "officeqa" in core
    assert "treasury" in core
    assert detect_generic_synthesizer_prompt(ast) == []


def test_plan_and_execute_gets_required_tool_kind_groups_from_contract() -> None:
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury fiscal calendar compute answer",
        design_brief=_contract_brief(),
        assets=_corpus_assets(),
    )

    plan_and_execute = next(
        node
        for node in _walk_nodes(ast["root"])
        if node.get("type") == "plan_and_execute"
    )
    assert plan_and_execute["config"]["required_tool_kind_groups"] == [
        ["vector_search"],
        ["table_search", "table_read", "table_load"],
        ["compute"],
    ]


def test_plan_and_execute_corpus_body_does_not_emit_lane_router() -> None:
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury fiscal calendar compute answer",
        design_brief=_contract_brief(),
        assets=_corpus_assets(),
    )

    plan_and_execute = next(
        node
        for node in _walk_nodes(ast["root"])
        if node.get("type") == "plan_and_execute"
    )
    body = plan_and_execute["config"]["body"]
    serialized_body = str(body)

    assert "research-lane-router" not in serialized_body
    assert "current_step.lane" not in serialized_body
    # Phase 0 (dataflow-enforcement plan): the body is the direct researcher —
    # the dead body reflector (whose control decision nothing read) was removed.
    assert body["type"] == "agent"
    assert body["id"] == "researcher"
    assert body["config"]["subtype"] == "researcher"


def test_corpus_only_contract_replaces_generic_web_planner_prompt() -> None:
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury fiscal calendar compute answer",
        design_brief=_contract_brief(),
        assets=_corpus_assets(),
    )

    plan_execute = next(node for node in _walk_nodes(ast["root"]) if node["id"] == "plan-and-execute")
    planner = plan_execute["config"]["planner"]
    system_prompt = planner["system_prompt"]
    user_prompt = planner["user_prompt_template"]

    assert "Databricks corpus research workflow" in system_prompt
    assert "Web search" not in system_prompt
    assert "Public information" not in system_prompt
    assert "Search the public web" not in system_prompt
    assert "Databricks corpus evidence plan" in user_prompt
    assert "vector_search" in user_prompt
    assert "table_read" in user_prompt
    assert "compute" in user_prompt
    assert "official documents" not in user_prompt


def test_corpus_only_contract_does_not_invent_no_web_policy() -> None:
    ast = build_web_research_workflow(
        intent="OfficeQA Treasury fiscal calendar compute answer",
        design_brief=_contract_brief_without_forbidden_web(),
        assets=_corpus_assets(),
    )

    text = str(ast)

    assert "Forbidden tool kinds" not in text
    assert "Prompt-forbidden web rule" not in text
    assert "do not fall back to public web evidence" not in text
    assert "Do not create URL, browser, or outside-source" not in text
    assert "Plan evidence steps against the named Databricks resources only" not in text


def test_is_corpus_only_assets_corpus_only_true() -> None:
    assets = [{"kind": "vector_index", "full_name": "x"}]
    assert _is_corpus_only_assets(assets, ToolPlan()) is True


def test_is_corpus_only_assets_web_signal_false() -> None:
    assets = [{"kind": "vector_index", "full_name": "x"}]
    plan = ToolPlan(tools=[ToolDeclarationSpec(name="w", kind="web_search")])
    assert _is_corpus_only_assets(assets, plan) is False


def test_fallback_lane_prompt_corpus_only_uses_retrieval_strategy() -> None:
    brief = WorkflowDesignBrief(workflow_name="x")
    text = _fallback_lane_user_prompt_template(
        lane_description="Lane focus",
        intent="OfficeQA Treasury",
        design_brief=brief,
        assets=[{"kind": "vector_index", "full_name": "main.foo.idx"}],
        tool_plan=ToolPlan(),
    )
    assert "Retrieval strategy" in text
    assert "vector_search" in text
    assert "chunk_id" in text
    # Forbidden legacy tokens for corpus-only:
    assert "official documents" not in text
    assert "Read the fetched" not in text


def test_fallback_lane_prompt_web_keeps_legacy_block() -> None:
    brief = WorkflowDesignBrief(workflow_name="x")
    text = _fallback_lane_user_prompt_template(
        lane_description="Lane focus",
        intent="research",
        design_brief=brief,
        assets=None,
        tool_plan=None,
    )
    assert "Search strategy" in text
    assert "official documents" in text
    assert "Read the fetched" in text


# ---------------------------------------------------------------------------
# PR3-A: Query Diversification block + ambiguity heuristic + Negative-existence
# discipline. All assertions are domain-generic — no Treasury/Army/1945 tokens.
# ---------------------------------------------------------------------------


# Bare tokens that must NOT appear in any builder-emitted prompt text. Each
# is a corpus-specific identifier from the OfficeQA failing case that the
# generic-vocabulary principle (memory ``feedback-generic-not-domain-specific``)
# forbids in framework code. Listed bare so future drift is caught by these
# tests across all five axes — not just period_basis.
_FORBIDDEN_DOMAIN_TOKENS = (
    "War Department",
    "Treasury",
    "Army",
    "1945",
    "1953",
    "OfficeQA",
)


def _ambiguous_brief(user_goal: str = "") -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="x",
        user_goal=user_goal,
    )


def test_infer_ambiguity_axes_picks_period_basis_from_year_token() -> None:
    axes = _infer_ambiguity_axes(intent="What are the totals in 1945?")
    assert "period_basis" in axes
    assert "temporal_scope" in axes


def test_infer_ambiguity_axes_picks_unit_basis_from_dollar() -> None:
    axes = _infer_ambiguity_axes(intent="What is the figure in million $?")
    assert "unit_basis" in axes


def test_infer_ambiguity_axes_picks_geographic_scope() -> None:
    axes = _infer_ambiguity_axes(intent="Which country leads the region?")
    assert "geographic_scope" in axes


def test_infer_ambiguity_axes_picks_entity_scope_with_year_and_institution() -> None:
    axes = _infer_ambiguity_axes(intent="What did the department do in 1945?")
    assert "entity_scope" in axes


def test_infer_ambiguity_axes_empty_for_unambiguous() -> None:
    axes = _infer_ambiguity_axes(intent="hello world")
    assert axes == []


def test_infer_ambiguity_axes_explicit_override_filtered() -> None:
    axes = _infer_ambiguity_axes(
        intent="hello", explicit_axes=["period_basis", "not_a_real_axis"]
    )
    assert axes == ["period_basis"]


def test_query_diversification_block_renders_per_axis_directives() -> None:
    text = _query_diversification_block(["period_basis", "unit_basis"])
    assert "Query diversification" in text
    assert "period_basis" in text
    assert "fiscal year" in text.lower()
    assert "calendar year" in text.lower()
    assert "unit_basis" in text
    assert "millions" in text
    # No tokens for axes not requested:
    assert "entity_scope" not in text


def test_query_diversification_block_empty_for_no_axes() -> None:
    assert _query_diversification_block([]) == ""


def test_fallback_lane_prompt_period_basis_adds_diversification() -> None:
    brief = _ambiguous_brief(user_goal="What are totals in 1945?")
    text = _fallback_lane_user_prompt_template(
        lane_description="lane focus",
        intent="What are totals in 1945?",
        design_brief=brief,
        assets=[{"kind": "vector_index", "full_name": "main.foo.idx"}],
        tool_plan=ToolPlan(),
    )
    assert "Query diversification" in text
    assert "fiscal year" in text.lower()
    assert "calendar year" in text.lower()
    for tok in _FORBIDDEN_DOMAIN_TOKENS:
        assert tok not in text, f"forbidden corpus-specific token leaked: {tok!r}"


def test_fallback_lane_prompt_no_ambiguity_omits_diversification() -> None:
    brief = WorkflowDesignBrief(workflow_name="x", user_goal="say hi")
    text = _fallback_lane_user_prompt_template(
        lane_description="lane focus",
        intent="say hi",
        design_brief=brief,
        assets=None,
        tool_plan=None,
    )
    assert "Query diversification" not in text


def test_with_lane_user_prompt_contract_period_basis_adds_diversification() -> None:
    brief = _ambiguous_brief(user_goal="totals in 2020")
    text = _with_lane_user_prompt_contract(
        description="lane focus",
        designer_template="Investigate {query}.",
        assets=[{"kind": "vector_index", "full_name": "main.foo.idx"}],
        tool_plan=ToolPlan(),
        intent="totals in 2020",
        design_brief=brief,
    )
    assert "Query diversification" in text
    assert "fiscal year" in text.lower()
    assert "calendar year" in text.lower()


def test_with_lane_user_prompt_contract_no_ambiguity_omits_block() -> None:
    brief = WorkflowDesignBrief(workflow_name="x", user_goal="hi")
    text = _with_lane_user_prompt_contract(
        description="lane focus",
        designer_template="Investigate {query}.",
        assets=None,
        tool_plan=None,
        intent="hi",
        design_brief=brief,
    )
    assert "Query diversification" not in text


def test_with_lane_user_prompt_contract_explicit_axes_override() -> None:
    brief = WorkflowDesignBrief(workflow_name="x", user_goal="hi")
    text = _with_lane_user_prompt_contract(
        description="lane focus",
        designer_template="Investigate {query}.",
        assets=None,
        tool_plan=None,
        intent="hi",
        design_brief=brief,
        ambiguity_axes=["period_basis"],
    )
    assert "Query diversification" in text
    assert "fiscal year" in text.lower()
    assert "calendar year" in text.lower()


def test_synthesizer_lane_coverage_directive_contains_negative_existence_rule() -> None:
    lanes = [{"id": "lane_1", "description": "first lane"}]
    text = _synthesizer_lane_coverage_directive(lanes)
    assert "Negative-existence claim discipline" in text
    assert "not retrieved by any lane" in text
    for tok in _FORBIDDEN_DOMAIN_TOKENS:
        assert tok not in text, f"forbidden corpus-specific token leaked: {tok!r}"


def test_plan_execute_synthesizer_directive_contains_negative_existence_rule() -> None:
    lanes = [{"id": "lane_1", "description": "first lane"}]
    text = _plan_execute_synthesizer_directive(lanes)
    assert "Negative-existence claim discipline" in text
    assert "not retrieved by any lane" in text
    for tok in _FORBIDDEN_DOMAIN_TOKENS:
        assert tok not in text, f"forbidden corpus-specific token leaked: {tok!r}"


def test_query_diversification_uses_only_generic_vocabulary() -> None:
    """All five axes' rendered text must be domain-agnostic."""
    for axis in (
        "period_basis",
        "entity_scope",
        "temporal_scope",
        "unit_basis",
        "geographic_scope",
    ):
        block = _query_diversification_block([axis])
        for tok in _FORBIDDEN_DOMAIN_TOKENS:
            assert tok not in block, (
                f"axis={axis} leaked corpus-specific token {tok!r}: {block!r}"
            )


# ---------------------------------------------------------------------------
# PR3-B Layer 1: task_signature threaded through build_web_research_workflow
# ---------------------------------------------------------------------------


_VALID_SIG_PIPELINED = {
    "asset_signature": "corpus_only",
    "retrieval_pattern": "pipelined_retrieve_read_compute",
    "question_class": "numeric_aggregation",
    "question_ambiguity": ["period_basis"],
    "primary_evidence_kind": "structured_tables",
    "expected_output_shape": "single_number",
}


_VALID_SIG_INDEPENDENT = {
    "asset_signature": "corpus_only",
    "retrieval_pattern": "independent_lanes",
    "question_class": "comparative_analysis",
    "question_ambiguity": [],
    "primary_evidence_kind": "text_chunks",
    "expected_output_shape": "structured_report",
}


_VALID_SIG_WEB_OPEN = {
    "asset_signature": "web_only",
    "retrieval_pattern": "open_research",
    "question_class": "open_research",
    "question_ambiguity": [],
    "primary_evidence_kind": "web_articles",
    "expected_output_shape": "paragraph",
}


def _make_generic_lane() -> dict:
    """Lane spec whose description is domain-agnostic (no forbidden tokens)."""
    return {
        "description": "Primary evidence retrieval lane.",
        "user_prompt_template": "Find evidence for {query}.",
    }


def test_build_workflow_signature_forces_plan_and_execute() -> None:
    # brief defaults to parallel_lanes — signature must override.
    brief = WorkflowDesignBrief(
        workflow_name="x",
        topology="parallel_lanes",
        research_lanes=[_make_generic_lane()],
    )
    ast = build_web_research_workflow(
        intent="hi",
        design_brief=brief,
        assets=_corpus_assets(),
        task_signature=_VALID_SIG_PIPELINED,
    )
    # plan_and_execute topology emits a plan_and_execute node somewhere.
    serialized = str(ast)
    assert "plan_and_execute" in serialized
    # Lane prompts must include the period_basis directive AND avoid
    # corpus-specific tokens in framework-emitted text. (User-authored
    # lane descriptions can carry whatever vocabulary the brief defines;
    # we use a generic lane here so the framework-text-only invariant
    # is actually testable.)
    for tok in _FORBIDDEN_DOMAIN_TOKENS:
        assert tok not in serialized, (
            f"corpus-specific token {tok!r} leaked into generated AST"
        )
    assert "fiscal year" in serialized.lower()
    assert "calendar year" in serialized.lower()


def test_build_workflow_signature_forces_parallel_lanes() -> None:
    brief = WorkflowDesignBrief(
        workflow_name="x",
        topology="plan_and_execute",  # signature must override to parallel_lanes
        research_lanes=[_make_generic_lane()],
    )
    ast = build_web_research_workflow(
        intent="hi",
        design_brief=brief,
        assets=_corpus_assets(),
        task_signature=_VALID_SIG_INDEPENDENT,
    )
    # No plan_and_execute node when topology was forced to parallel_lanes.
    serialized = str(ast)
    # The parallel_lanes topology emits a top-level parallel node, not a
    # plan_and_execute one — check the root path is sequence-with-parallel.
    assert "plan_and_execute" not in serialized


def test_build_workflow_signature_invalid_raises_signature_error() -> None:
    """Plan v2.1 M11: failure-closed for invalid signatures.

    Previously this builder silently fell back to brief.topology on
    invalid TaskSignature payloads — the exact bypass that lets brief
    vocabulary override the classifier (the Investment failure mode).
    Now invalid signatures raise SignatureError; the designer flow halts
    with a clear classification failure instead of producing a wrong AST.
    """
    from deep_research.agent_designer.task_signature import SignatureError

    brief = WorkflowDesignBrief(
        workflow_name="x",
        topology="parallel_lanes",
        research_lanes=[_make_generic_lane()],
    )
    bad_sig = dict(_VALID_SIG_PIPELINED)
    bad_sig["asset_signature"] = "garbage_signature"
    with pytest.raises(SignatureError, match="task_signature payload failed validation"):
        build_web_research_workflow(
            intent="hi",
            design_brief=brief,
            assets=_corpus_assets(),
            task_signature=bad_sig,
        )


def test_build_workflow_signature_threads_no_ambiguity_axes() -> None:
    # When the signature carries an empty question_ambiguity list, the lane
    # prompts must NOT inject the Query Diversification block (heuristic
    # detection is bypassed by the explicit empty list).
    brief = WorkflowDesignBrief(
        workflow_name="x",
        topology="parallel_lanes",
        research_lanes=[_make_generic_lane()],
    )
    ast = build_web_research_workflow(
        intent="hi",
        design_brief=brief,
        assets=_corpus_assets(),
        task_signature=_VALID_SIG_INDEPENDENT,
    )
    serialized = str(ast)
    assert "Query diversification" not in serialized


# ---------------------------------------------------------------------------
# _tool_plan_bindings — P4-1 regression for the silent-empty-fallback defect.
# When the architect's tool_plan omits a node id (e.g., cross-lane-researcher)
# the helper used to return `[]` and the researcher ran with zero tools,
# producing the "Insufficient Evidence" empty-report symptom on deployed
# treasury / vector-index agents. See plan Phase 4.
# ---------------------------------------------------------------------------


def _brief_with_tool_plan(tool_plan: ToolPlan | None) -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="test",
        topology="plan_and_execute",
        research_lanes=[],
        tool_plan=tool_plan,
    )


def test_tool_plan_bindings_no_tool_plan_returns_default() -> None:
    """When ``tool_plan`` is None, the helper falls back to caller default."""
    brief = _brief_with_tool_plan(None)
    result = _tool_plan_bindings(
        brief, node_id="cross-lane-researcher",
        default=["web_research"], researcher=True,
    )
    assert result == ["web_research"]


def test_tool_plan_bindings_exact_node_match_returns_bound() -> None:
    """A binding whose node_id matches is used verbatim."""
    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="vector_search", kind="vector_search"),
            ToolDeclarationSpec(name="table_read", kind="table_read"),
        ],
        bindings=[
            ToolBindingSpec(
                node_id="cross-lane-researcher",
                tool_names=["vector_search"],
            ),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    assert result == ["vector_search"]


def test_tool_plan_bindings_wildcard_researchers_alias() -> None:
    """The 'researchers' wildcard binding applies to any researcher node_id."""
    plan = ToolPlan(
        tools=[ToolDeclarationSpec(name="vector_search", kind="vector_search")],
        bindings=[
            ToolBindingSpec(node_id="researchers", tool_names=["vector_search"]),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    assert result == ["vector_search"]


def test_tool_plan_bindings_no_binding_match_falls_back_to_declared_corpus_tools() -> None:
    """REGRESSION for the user's 2026-05-25 treasury bug.

    The architect bound `vector_search` to `lane_1-researcher` but not to
    the cross-lane fallback (and didn't use a wildcard). The lane router
    routed to the cross-lane node, which used to receive `tools=[]`. After
    the fix it falls back to the declared evidence tools.
    """
    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="vector_search", kind="vector_search"),
            ToolDeclarationSpec(name="table_read", kind="table_read"),
        ],
        bindings=[
            ToolBindingSpec(node_id="lane_1-researcher", tool_names=["vector_search"]),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],  # generic web default — must NOT win
        researcher=True,
    )
    # Fallback picks declared evidence tools, not the web-flavored default.
    assert result == ["vector_search", "table_read"]


def test_tool_plan_bindings_web_only_workflow_fallback() -> None:
    """For a web-only workflow with no matching binding, fall back to declared
    web tools (matches the historical default for that mode)."""
    plan = ToolPlan(
        tools=[ToolDeclarationSpec(name="web_research", kind="web_research")],
        bindings=[
            ToolBindingSpec(node_id="lane_1-researcher", tool_names=["web_research"]),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    assert result == ["web_research"]


def test_tool_plan_bindings_mixed_workflow_fallback_includes_both() -> None:
    """Mixed (corpus + web) workflow's cross-lane fallback gets both kinds."""
    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="vector_search", kind="vector_search"),
            ToolDeclarationSpec(name="web_research", kind="web_research"),
        ],
        bindings=[
            ToolBindingSpec(node_id="lane_1-researcher", tool_names=["vector_search"]),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    # Order follows tool_plan.tools declaration order.
    assert result == ["vector_search", "web_research"]


def test_tool_plan_bindings_no_evidence_tools_declared_falls_back_to_default() -> None:
    """If tool_plan declares only non-evidence tools (e.g., compute) and no
    binding matches, fall back to caller default — never empty."""
    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="compute", kind="compute"),
        ],
        bindings=[
            ToolBindingSpec(node_id="lane_1-researcher", tool_names=["compute"]),
        ],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    assert result == ["web_research"]


def test_tool_plan_bindings_never_returns_empty_when_default_nonempty() -> None:
    """Defense-in-depth: every code path returns at least one tool when the
    caller supplied a non-empty default. The silent-empty regression is the
    bug we are fixing."""
    plan = ToolPlan(
        tools=[],
        bindings=[],
    )
    result = _tool_plan_bindings(
        _brief_with_tool_plan(plan),
        node_id="cross-lane-researcher",
        default=["web_research"],
        researcher=True,
    )
    assert result == ["web_research"]


# ---------------------------------------------------------------------------
# Structural-gate regression — P4-2 confirms the gate catches the failing
# pattern from the user's 2026-05-25 treasury bug (researcher with tools=[]
# AND pool_writes targeting the sources pool). The gate exists at
# structural_gate.py:160; this test pins it in so the rule isn't accidentally
# weakened.
# ---------------------------------------------------------------------------


def test_structural_gate_rejects_researcher_with_no_evidence_tools() -> None:
    """A researcher node with ``tools=[]`` that writes to the sources pool
    must be flagged by the structural gate."""
    from deep_research.agent_designer.structural_gate import (
        detect_tool_access_contract,
    )
    ast = {
        "id": "designer-draft",
        "tools": [
            {"name": "vector_search", "kind": "vector_search"},
        ],
        "root": {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "cross-lane-researcher",
                    "type": "agent",
                    "label": "Cross-Lane Researcher",
                    "config": {
                        "subtype": "researcher",
                        "tools": [],  # <-- the bug
                        "pool_writes": [
                            {"pool": "sources", "extract": "sources"},
                        ],
                    },
                },
            ],
        },
    }
    errors = detect_tool_access_contract(ast)
    messages = [str(e.message) for e in errors]
    assert any(
        "no bound executable evidence tools" in m.lower()
        or "no bound" in m.lower()
        for m in messages
    ), f"Expected gate to flag empty-tools researcher, got: {messages}"


def test_structural_gate_passes_researcher_with_evidence_tools() -> None:
    """Sanity: a researcher correctly bound to a declared evidence tool
    passes the gate (no false positives from the P4-2 rule)."""
    from deep_research.agent_designer.structural_gate import (
        detect_tool_access_contract,
    )
    ast = {
        "id": "designer-draft",
        "tools": [
            {"name": "vector_search", "kind": "vector_search"},
        ],
        "root": {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "researcher",
                    "type": "agent",
                    "config": {
                        "subtype": "researcher",
                        "tools": ["vector_search"],
                        "pool_writes": [
                            {"pool": "sources", "extract": "sources"},
                        ],
                    },
                },
            ],
        },
    }
    errors = detect_tool_access_contract(ast)
    no_evidence_errors = [
        e for e in errors
        if "no bound" in str(e.message).lower()
        or "no bound executable evidence tools" in str(e.message).lower()
    ]
    assert no_evidence_errors == [], (
        f"Gate falsely flagged a properly-bound researcher: {no_evidence_errors}"
    )
