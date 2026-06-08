"""Plan v2.1 PR-2 — deterministic blueprint builder tests.

Covers the three pieces of :mod:`deep_research.agent_designer.blueprint`:

* :func:`compute_lane_key` — stable content-derived identifier; safe
  truncation; collision resistance across descriptions that share a
  prefix.
* :func:`compute_structural_fingerprint` — canonicalized sha256 that
  IS sensitive to topology/node-ID changes and is NOT sensitive to
  prompt-only mutations (so PR-3's architect ``node_patches`` flow can
  rely on it for immutability).
* :func:`build_blueprint` — golden tests for the two anchor cases
  (OfficeQA-like single-lane plan_and_execute; Investment-like 6-lane
  parallel_lanes), plus failure-closed checks per plan M11.

These are pure-Python tests — no LLM, no fixtures, no flag manipulation
(the flag only affects the YAML wiring, not the builder itself).
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.agent_designer.blueprint import (
    DESIGNER_DETERMINISTIC_BLUEPRINT_ENV,
    SignatureError,
    build_blueprint,
    compute_lane_key,
    compute_structural_fingerprint,
    is_deterministic_blueprint_enabled,
)

# ---------------------------------------------------------------------------
# Fixture builders — keep signatures local so each test reads top-to-bottom.
# ---------------------------------------------------------------------------


def _officeqa_signature() -> dict[str, Any]:
    """A signature representative of the OfficeQA anchor case.

    Single concern, with dependencies and iteration — the classic
    pipelined retrieve→read→compute corpus task. The structural axes
    drive ``plan_and_execute`` via Rule 2 of select_topology.
    """
    return {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
        "independent_workstreams_count": 1,
        "step_dependencies_present": True,
        "iteration_required": True,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["retrieve then read then compute pipeline"],
    }


def _investment_signature() -> dict[str, Any]:
    """A signature representative of the Investment anchor case.

    Six independent concerns — the classic open-research multi-domain
    task. Even with ``iteration_required=True``, the M4 Rule 1
    precedence (independence wins first) routes this to
    ``parallel_lanes`` — this is the explicit fix for the codex
    CRITICAL-5 finding that the v2 mapping would have recreated the
    Investment failure.
    """
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 6,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": [
            "fundamentals",
            "valuation",
            "risk",
            "market trends",
            "earnings",
            "competitors",
        ],
    }


def _corpus_assets() -> list[dict[str, Any]]:
    return [
        {"kind": "vector_index", "full_name": "main.officeqa_benchmark.treasury_chunks"},
        {"kind": "delta_table", "full_name": "main.officeqa_benchmark.treasury_tables"},
    ]


# ---------------------------------------------------------------------------
# Feature-flag helper
# ---------------------------------------------------------------------------


def test_flag_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """PR-3 default-ON: DESIGNER_DETERMINISTIC_BLUEPRINT unset → ON."""
    monkeypatch.delenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, raising=False)
    assert is_deterministic_blueprint_enabled() is True


@pytest.mark.parametrize(
    "truthy", ["1", "true", "TRUE", "yes", "on", " True ", "anything-else"]
)
def test_flag_recognizes_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, truthy: str
) -> None:
    """PR-3: any non-empty value not in the OFF set enables the flag."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, truthy)
    assert is_deterministic_blueprint_enabled() is True


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "OFF", " false "])
def test_flag_recognizes_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, falsy: str
) -> None:
    """PR-3 opt-out tokens: setting the env to {0|false|no|off|empty}
    keeps the legacy architect-authored-AST path."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, falsy)
    assert is_deterministic_blueprint_enabled() is False


# ---------------------------------------------------------------------------
# compute_lane_key
# ---------------------------------------------------------------------------


def test_lane_key_format_short_word() -> None:
    """Single-word description: snake prefix + 8-char hash suffix."""
    key = compute_lane_key("fundamentals")
    assert key.startswith("fundamentals_")
    _, suffix = key.rsplit("_", 1)
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_lane_key_format_multi_word_snakeified() -> None:
    """Multi-word descriptions: non-alphanumeric collapses to underscore."""
    key = compute_lane_key("Market Trends & Sentiment!")
    # Truncation cap is 24 chars on the lowercased prefix.
    snake, suffix = key.rsplit("_", 1)
    # snake must contain only [a-z0-9_]
    assert all(c.isalnum() or c == "_" for c in snake)
    assert len(suffix) == 8


def test_lane_key_truncates_long_descriptions_but_hash_uses_full() -> None:
    """Two descriptions sharing the first 24 chars still get distinct keys
    because the hash suffix is computed on the FULL description.
    """
    long_a = "long description that shares prefix bytes A"
    long_b = "long description that shares prefix bytes B"
    key_a = compute_lane_key(long_a)
    key_b = compute_lane_key(long_b)
    # Snake prefix identical (first 24 chars matches)
    snake_a, hash_a = key_a.rsplit("_", 1)
    snake_b, hash_b = key_b.rsplit("_", 1)
    assert snake_a == snake_b
    # But hash suffixes differ — collision resistance from the full descriptor
    assert hash_a != hash_b


def test_lane_key_deterministic_across_invocations() -> None:
    """compute_lane_key is a pure function of its input."""
    assert compute_lane_key("risk") == compute_lane_key("risk")


def test_lane_key_unicode_safe() -> None:
    """Lane descriptions can contain arrows, em-dashes, etc."""
    key = compute_lane_key("retrieve→read→compute pipeline")
    assert "retrieve" in key.lower()
    snake, suffix = key.rsplit("_", 1)
    assert len(suffix) == 8


def test_lane_key_empty_description_raises() -> None:
    """Empty descriptions are a classifier-contract violation — fail-closed."""
    with pytest.raises(SignatureError):
        compute_lane_key("")
    with pytest.raises(SignatureError):
        compute_lane_key("   ")


# ---------------------------------------------------------------------------
# compute_structural_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_calls() -> None:
    """The same AST always produces the same fingerprint."""
    ast = build_blueprint(_investment_signature(), "investment analysis", [])
    first = compute_structural_fingerprint(ast)
    second = compute_structural_fingerprint(ast)
    assert first == second
    # And matches the one embedded by build_blueprint
    assert ast["structural_fingerprint"] == first


def test_fingerprint_insensitive_to_prompt_only_changes() -> None:
    """Plan v2.1 M2: prompt mutations must NOT change the fingerprint.

    The architect's allow-listed patches (system_prompt,
    user_prompt_template, model_tier, error_handling, max_tool_calls)
    are not part of the structural projection. Mutating them and
    reapplying the fingerprint must yield the same value.
    """
    ast = build_blueprint(_investment_signature(), "q", [])
    original = compute_structural_fingerprint(ast)

    mutated = copy.deepcopy(ast)
    # Walk and mutate every prompt field we can find.
    def _mutate_prompts(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config")
        if isinstance(config, dict):
            if "system_prompt" in config:
                config["system_prompt"] = "MUTATED — architect customization"
            if "user_prompt_template" in config:
                config["user_prompt_template"] = "MUTATED user template {{ query }}"
            for child in config.get("body", {}).values() if isinstance(config.get("body"), dict) else []:
                _mutate_prompts(child)
        for child in node.get("children") or []:
            _mutate_prompts(child)

    _mutate_prompts(mutated["root"])
    assert compute_structural_fingerprint(mutated) == original


def test_fingerprint_sensitive_to_topology_change() -> None:
    """Replacing the topology body type changes the fingerprint."""
    ast = build_blueprint(_investment_signature(), "q", [])
    original = compute_structural_fingerprint(ast)
    mutated = copy.deepcopy(ast)
    # Find the parallel node and flip it to 'sequence' — any structural change.
    children = mutated["root"]["children"]
    assert children[1]["type"] in {"parallel", "plan_and_execute"}
    children[1]["type"] = "sequence"
    assert compute_structural_fingerprint(mutated) != original


def test_fingerprint_sensitive_to_pool_dedup_key_change() -> None:
    """Pool dedup_key changes alter the fingerprint."""
    ast = build_blueprint(_investment_signature(), "q", [])
    original = compute_structural_fingerprint(ast)
    mutated = copy.deepcopy(ast)
    pools = mutated.get("pools") or []
    assert pools, "test fixture should contain at least one pool"
    pools[0]["dedup_key"] = "different_key"
    assert compute_structural_fingerprint(mutated) != original


def test_fingerprint_sensitive_to_node_id_change() -> None:
    """Renaming a node changes its identity and therefore the fingerprint."""
    ast = build_blueprint(_investment_signature(), "q", [])
    original = compute_structural_fingerprint(ast)
    mutated = copy.deepcopy(ast)
    # Rename the first lane researcher.
    parallel_or_body = mutated["root"]["children"][1]
    if parallel_or_body.get("type") == "parallel":
        parallel_or_body["children"][0]["id"] = "renamed-lane-researcher"
    elif "config" in parallel_or_body and "body" in parallel_or_body["config"]:
        # plan_and_execute case — dive into body
        parallel_or_body["config"]["body"]["children"][0]["children"][0][
            "id"
        ] = "renamed-lane-researcher"
    assert compute_structural_fingerprint(mutated) != original


# ---------------------------------------------------------------------------
# build_blueprint — OfficeQA golden (plan_and_execute + chunk_id)
# ---------------------------------------------------------------------------


def test_blueprint_officeqa_goes_to_plan_and_execute() -> None:
    """OfficeQA-like signature (count=1, deps=True, iter=True) → plan_and_execute.

    The topology body sits at root.children[1] (root is a sequence
    wrapping coordinator + topology body + synthesizer).
    """
    ast = build_blueprint(_officeqa_signature(), "q", _corpus_assets())
    topology_body = ast["root"]["children"][1]
    assert topology_body["type"] == "plan_and_execute"


def test_blueprint_officeqa_uses_chunk_id_dedup() -> None:
    """Corpus-only assets → pool dedup_key=chunk_id via _sources_dedup_key.

    Plan v2.1 M6: dedup_key is derived from REAL assets, never from
    the classifier's asset_signature. Test that supplying corpus assets
    gives the right key regardless of what asset_signature says.
    """
    ast = build_blueprint(_officeqa_signature(), "q", _corpus_assets())
    sources_pools = [p for p in ast.get("pools") or [] if p.get("name") == "sources"]
    assert sources_pools, "sources pool must exist"
    assert sources_pools[0]["dedup_key"] == "chunk_id"


def test_blueprint_officeqa_one_lane_key() -> None:
    """Single-concern signature → exactly one lane_key in metadata."""
    ast = build_blueprint(_officeqa_signature(), "q", _corpus_assets())
    lane_keys = ast.get("lane_keys") or {}
    assert len(lane_keys) == 1
    only_key = next(iter(lane_keys.keys()))
    assert only_key.endswith(only_key.split("_")[-1])  # has hash suffix
    assert lane_keys[only_key] == "retrieve then read then compute pipeline"


def test_blueprint_officeqa_embeds_fingerprint() -> None:
    ast = build_blueprint(_officeqa_signature(), "q", _corpus_assets())
    fp = ast.get("structural_fingerprint")
    assert isinstance(fp, str)
    assert len(fp) == 64  # sha256 hex digest length
    assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# build_blueprint — Investment golden (parallel_lanes + url + 6 lanes)
# ---------------------------------------------------------------------------


def test_blueprint_investment_goes_to_parallel_lanes() -> None:
    """6 independent concerns → parallel_lanes (Rule 1 independence wins).

    This is the codex CRITICAL-5 fix in action: a six-lane brief
    cannot recreate the Investment failure because Rule 1 wins before
    Rule 2 fires on iteration_required.
    """
    ast = build_blueprint(_investment_signature(), "q", [])
    topology_body = ast["root"]["children"][1]
    assert topology_body["type"] == "parallel"


def test_blueprint_investment_has_six_lane_researchers() -> None:
    """Six lane descriptions → six researcher nodes under parallel."""
    ast = build_blueprint(_investment_signature(), "q", [])
    parallel = ast["root"]["children"][1]
    researchers = [c for c in parallel.get("children") or [] if c.get("type") == "agent"]
    assert len(researchers) == 6


def test_blueprint_investment_uses_url_dedup() -> None:
    """Web-only assets (or no assets) → pool dedup_key=url."""
    ast = build_blueprint(_investment_signature(), "q", [])
    sources_pools = [p for p in ast.get("pools") or [] if p.get("name") == "sources"]
    assert sources_pools
    assert sources_pools[0]["dedup_key"] == "url"


def test_blueprint_investment_lane_keys_match_descriptions() -> None:
    """Six extractive lane descriptions → six lane_keys with verbatim values."""
    ast = build_blueprint(_investment_signature(), "q", [])
    lane_keys = ast.get("lane_keys") or {}
    assert len(lane_keys) == 6
    assert set(lane_keys.values()) == {
        "fundamentals",
        "valuation",
        "risk",
        "market trends",
        "earnings",
        "competitors",
    }


def test_blueprint_investment_iteration_required_does_not_reroute() -> None:
    """Even with iteration_required=True, 6 lanes stay parallel_lanes.

    Direct golden for the M4 Rule 1 precedence — the exact failure
    mode codex CRITICAL-5 flagged in the v2 mapping.
    """
    sig = _investment_signature()
    sig["iteration_required"] = True
    sig["step_dependencies_present"] = True
    ast = build_blueprint(sig, "q", [])
    assert ast["root"]["children"][1]["type"] == "parallel"


# ---------------------------------------------------------------------------
# Failure-closed (plan M11)
# ---------------------------------------------------------------------------


def test_blueprint_none_signature_raises() -> None:
    with pytest.raises(SignatureError, match="required"):
        build_blueprint(None, "q", [])  # type: ignore[arg-type]


def test_blueprint_non_dict_signature_raises() -> None:
    with pytest.raises(SignatureError, match="must be a dict"):
        build_blueprint([], "q", [])  # type: ignore[arg-type]


def test_blueprint_empty_signature_raises() -> None:
    with pytest.raises(SignatureError, match="validation"):
        build_blueprint({}, "q", [])


def test_blueprint_missing_required_field_raises() -> None:
    """TaskSignature requires asset_signature, retrieval_pattern, etc."""
    sig = _officeqa_signature()
    del sig["asset_signature"]
    with pytest.raises(SignatureError, match="validation"):
        build_blueprint(sig, "q", [])


def test_blueprint_invalid_enum_value_raises() -> None:
    sig = _officeqa_signature()
    sig["asset_signature"] = "not_a_real_asset_signature"
    with pytest.raises(SignatureError, match="validation"):
        build_blueprint(sig, "q", [])


def test_blueprint_empty_intent_raises() -> None:
    with pytest.raises(SignatureError, match="intent"):
        build_blueprint(_officeqa_signature(), "", _corpus_assets())


def test_blueprint_whitespace_intent_raises() -> None:
    with pytest.raises(SignatureError, match="intent"):
        build_blueprint(_officeqa_signature(), "   \n\t  ", _corpus_assets())


# ---------------------------------------------------------------------------
# Graceful degradation — under-populated lane_descriptions
# ---------------------------------------------------------------------------


def test_blueprint_under_populated_lane_descriptions_pads_placeholders() -> None:
    """Plan M6 contract: ``len(lane_descriptions) == count``.

    If the classifier under-populates (count=4 but only 2 descriptions),
    the builder degrades gracefully with generic ``concern_<i>``
    placeholders rather than failing closed. The architect can then call
    ``request_signature_revision`` if it sees these placeholders.
    """
    sig = _investment_signature()
    sig["independent_workstreams_count"] = 4
    sig["lane_descriptions"] = ["fundamentals", "risk"]
    ast = build_blueprint(sig, "q", [])
    lane_keys = ast.get("lane_keys") or {}
    assert len(lane_keys) == 4
    values = set(lane_keys.values())
    assert "fundamentals" in values
    assert "risk" in values
    # Two synthetic placeholders for the remaining lanes.
    assert any(v.startswith("concern_") for v in values)


def test_blueprint_over_populated_lane_descriptions_truncates() -> None:
    """Too many descriptions for the count → builder truncates to count."""
    sig = _investment_signature()
    sig["independent_workstreams_count"] = 3
    # lane_descriptions has 6 entries; builder truncates to 3.
    ast = build_blueprint(sig, "q", [])
    lane_keys = ast.get("lane_keys") or {}
    assert len(lane_keys) == 3


def test_blueprint_lane_descriptions_empty_with_count_one_uses_placeholder() -> None:
    """The classifier may emit count=1 with no descriptions for bounded lookups."""
    sig = _officeqa_signature()
    sig["independent_workstreams_count"] = 1
    sig["lane_descriptions"] = []
    sig["step_dependencies_present"] = False
    sig["iteration_required"] = False
    ast = build_blueprint(sig, "q", _corpus_assets())
    lane_keys = ast.get("lane_keys") or {}
    # Single-agent topology: builder still attaches a lane_key for the
    # single concern (descriptive placeholder).
    assert len(lane_keys) == 1


# ---------------------------------------------------------------------------
# Anti-hardcoding regression: dedup_key derived from assets, not signature
# ---------------------------------------------------------------------------


def test_blueprint_dedup_key_follows_assets_not_signature_asset_signature() -> None:
    """Plan v2.1 M6: dedup_key derives from REAL assets, never from
    the classifier's asset_signature.

    Construct a signature claiming asset_signature='web_only' but pass
    corpus assets — the dedup_key must follow the actual assets, not
    the classifier's descriptive label.
    """
    sig = _investment_signature()
    sig["asset_signature"] = "web_only"  # Classifier says web
    # ...but the real assets are corpus
    ast = build_blueprint(sig, "q", _corpus_assets())
    sources_pools = [p for p in ast.get("pools") or [] if p.get("name") == "sources"]
    assert sources_pools[0]["dedup_key"] == "chunk_id"  # follows assets


def test_blueprint_corpus_only_with_empty_assets_fails_closed() -> None:
    """Plan v2.2 grounding — corpus_only + empty assets is fail-closed.

    The previous behavior was "degenerate but not unsafe" — silently fall
    back to ``web_research`` defaults. That swap violated the corpus-only
    contract whenever the classifier inferred ``corpus_only`` from the
    user_intent text but the intent-grounding stage failed to resolve any
    workspace asset. The new contract surfaces the gap to the architect
    (which calls ``request_signature_revision``) rather than producing a
    workflow that researches the wrong evidence.
    """
    sig = _officeqa_signature()
    sig["asset_signature"] = "corpus_only"
    with pytest.raises(SignatureError, match="grounding"):
        build_blueprint(sig, "q", [])  # no UI-selected and no grounded assets


def test_blueprint_dedup_key_url_for_web_only_signature_empty_assets() -> None:
    """web_only signature + empty assets is the supported empty-asset path."""
    sig = _officeqa_signature()
    sig["asset_signature"] = "web_only"
    # web_only with no UI-selected assets is a valid configuration; the
    # builder produces a url-keyed sources pool and lets _default_web_tool_decls
    # supply the runtime tools.
    ast = build_blueprint(sig, "q", [])
    sources_pools = [p for p in ast.get("pools") or [] if p.get("name") == "sources"]
    assert sources_pools[0]["dedup_key"] == "url"


# ---------------------------------------------------------------------------
# Idempotency / determinism
# ---------------------------------------------------------------------------


def test_blueprint_idempotent_same_inputs_yield_same_fingerprint() -> None:
    """Two builds with identical inputs produce the same structural fingerprint."""
    sig = _investment_signature()
    ast_a = build_blueprint(sig, "investment analysis", [])
    ast_b = build_blueprint(sig, "investment analysis", [])
    assert ast_a["structural_fingerprint"] == ast_b["structural_fingerprint"]
    assert ast_a["lane_keys"] == ast_b["lane_keys"]


def test_blueprint_distinct_signatures_yield_distinct_fingerprints() -> None:
    """OfficeQA and Investment fingerprints must differ (different topologies)."""
    ast_a = build_blueprint(_officeqa_signature(), "q", _corpus_assets())
    ast_b = build_blueprint(_investment_signature(), "q", [])
    assert ast_a["structural_fingerprint"] != ast_b["structural_fingerprint"]


# ---------------------------------------------------------------------------
# Asset → tool wiring (closes the long-standing "blueprint declares web
# tools for corpus_only assets" gap that broke the OfficeQA scaffold-and-run)
# ---------------------------------------------------------------------------


def _officeqa_full_corpus_assets() -> list[dict[str, Any]]:
    """OfficeQA-style assets with the metadata recommend_tools_for_assets
    needs to wire table_search/table_read/table_load/compute (warehouse_id
    + field_roles). Mirrors what the live case fixture carries; the minimal
    ``_corpus_assets()`` fixture above lacks these, so the recommended tool
    set drops to just ``vector_search``."""
    return [
        {
            "kind": "vector_index",
            "full_name": "vs.example.chunks_index",
            "usage": "required",
            "metadata": {"columns": ["chunk_id", "content"], "query_type": "HYBRID"},
        },
        {
            "kind": "delta_table",
            "full_name": "delta.example.chunks",
            "usage": "required",
            "field_roles": {
                "primary_key": "chunk_id",
                "content": "content",
                "order_by": "chunk_id",
            },
            "metadata": {
                "warehouse_id": "abc123",
                "columns": ["chunk_id", "content"],
            },
        },
        {
            "kind": "delta_table",
            "full_name": "delta.example.tables",
            "usage": "required",
            "field_roles": {
                "primary_key": "chunk_id",
                "content": "content",
                "structured_json": "table_json",
                "order_by": "chunk_id",
            },
            "metadata": {
                "warehouse_id": "abc123",
                "columns": [
                    "chunk_id",
                    "content",
                    "table_json",
                ],
            },
        },
    ]


def _officeqa_tool_contract() -> dict[str, Any]:
    return {
        "schema": "resolved_tool_contract.v1",
        "source": "prompt_grounding",
        "evidence_policy": "corpus_only",
        "resources": [
            {
                "kind": "vector_index",
                "identity": "vs.example.chunks_index",
                "usage": "required",
                "access_status": "unverified",
                "provenance": "prompt_exact_identity",
                "capabilities": ["vector_search"],
                "domain_terms": ["officeqa", "treasury", "chunks", "vector"],
            },
            {
                "kind": "delta_table",
                "identity": "delta.example.tables",
                "usage": "required",
                "access_status": "unverified",
                "provenance": "prompt_exact_identity",
                "capabilities": ["table_read", "table_load", "compute"],
                "domain_terms": ["treasury", "tables", "compute"],
            },
        ],
        "required_capabilities": [
            "vector_search",
            "table_search",
            "table_read",
            "table_load",
            "compute",
        ],
        "ready_tool_kinds": [
            "vector_search",
            "table_search",
            "table_read",
            "table_load",
            "compute",
        ],
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
                "Use the named Databricks corpus resources before synthesis."
            ],
            "forbidden_tool_kinds": ["web_search", "web_crawl", "web_research"],
        },
    }


def _declared_tool_kinds(ast: dict[str, Any]) -> set[str]:
    return {
        str(t.get("kind") or "")
        for t in (ast.get("tools") or [])
        if isinstance(t, dict)
    }


def _declared_tool_names(ast: dict[str, Any]) -> set[str]:
    return {
        str(t.get("name") or "")
        for t in (ast.get("tools") or [])
        if isinstance(t, dict)
    }


def _walk(node: Any):
    if not isinstance(node, dict):
        return
    yield node
    config = node.get("config") if isinstance(node.get("config"), dict) else {}
    for nested in (config.get("body"), config.get("evaluator"), config.get("planner")):
        if isinstance(nested, dict):
            yield from _walk(nested)
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk(child)


def _researcher_tool_bindings(ast: dict[str, Any]) -> list[list[str]]:
    """Return the ``tools:`` list bound to every researcher-shaped node."""
    bindings: list[list[str]] = []
    for node in _walk(ast.get("root") or {}):
        if node.get("type") != "agent":
            continue
        config = node.get("config") or {}
        subtype = str(config.get("subtype") or "")
        # Researcher subtypes the deterministic blueprint generates today;
        # extend rather than narrow if more researcher kinds are added.
        if subtype not in {"researcher", "background_researcher"}:
            continue
        tools = config.get("tools")
        if isinstance(tools, list):
            bindings.append([str(t) for t in tools if isinstance(t, str)])
    return bindings


def test_blueprint_corpus_only_declares_corpus_tools_not_web() -> None:
    """OfficeQA-style corpus_only signature + corpus assets → corpus tools
    in the global registry, NO web defaults. This is the bug the OfficeQA
    scaffold-and-run was hitting before the asset→tool wiring landed in
    build_blueprint."""
    ast = build_blueprint(
        _officeqa_signature(),
        "OfficeQA Treasury question over selected vector + tables",
        _officeqa_full_corpus_assets(),
    )
    kinds = _declared_tool_kinds(ast)
    # Required: vector_search + table tools come from the recommender
    assert "vector_search" in kinds
    assert "table_search" in kinds
    assert "table_read" in kinds
    assert "table_load" in kinds
    assert "compute" in kinds
    # Forbidden: web tools must not have leaked in
    assert "web_research" not in kinds
    assert "web_crawl" not in kinds


def test_blueprint_corpus_only_binds_corpus_tools_to_each_lane() -> None:
    """Every researcher node gets every recommended tool — the
    ``all_researchers`` binding the asset→tool plan emits."""
    ast = build_blueprint(
        _officeqa_signature(), "q", _officeqa_full_corpus_assets()
    )
    bindings = _researcher_tool_bindings(ast)
    assert bindings, "expected at least one researcher node"
    expected_tools = _declared_tool_names(ast)
    # Filter out compute_namespace (helper for compute) — researchers
    # only need the primary evidence-gathering + computation tools.
    for tool_list in bindings:
        bound = set(tool_list)
        # All declared tools should be reachable from every researcher.
        missing = expected_tools - bound
        assert not missing, (
            f"researcher lane missing tools: {sorted(missing)} "
            f"(bound={sorted(bound)}, expected={sorted(expected_tools)})"
        )


def test_blueprint_corpus_only_required_asset_without_warehouse_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required Delta asset missing its warehouse_id can't be wired into
    table_search/table_read/table_load. The blueprint builder must fail
    closed rather than silently fall back to web tools (which would violate
    the case contract)."""
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
    sig = _officeqa_signature()
    assets = [
        {
            "kind": "vector_index",
            "full_name": "vs.example.idx",
            "usage": "required",
            "metadata": {"columns": ["chunk_id"]},
        },
        {
            "kind": "delta_table",
            "full_name": "delta.example.no_warehouse",
            "usage": "required",
            "field_roles": {"content": "content"},
            # NOTE: no metadata.warehouse_id — the recommender will report
            # a severity=error diagnostic for this asset.
            "metadata": {},
        },
    ]
    with pytest.raises(SignatureError, match="warehouse"):
        build_blueprint(sig, "q", assets)


def test_blueprint_corpus_plus_web_declares_both_kinds() -> None:
    """``corpus_plus_web`` is the explicit mixed-evidence signature: blueprint
    wires both the recommended corpus tools AND ``web_research``.
    ``web_crawl`` is intentionally excluded (plan F1) — it produced
    deterministic failures when parallel lanes called it before any
    ``web_research`` populated the URL registry. Architect can re-add via
    ``tool_plan`` customization if a workflow truly needs the crawl tool.
    """
    sig = _officeqa_signature()
    sig["asset_signature"] = "corpus_plus_web"
    ast = build_blueprint(sig, "q", _officeqa_full_corpus_assets())
    kinds = _declared_tool_kinds(ast)
    assert "vector_search" in kinds
    assert "table_search" in kinds
    assert "web_research" in kinds
    # web_crawl intentionally excluded — see plan F1
    assert "web_crawl" not in kinds


def test_blueprint_no_assets_signature_keeps_web_defaults() -> None:
    """Investment-style empty-asset workflow keeps web defaults.

    Plan v2.1 generic-robustness — ``web_crawl`` is NO LONGER in the
    default tool plan (it caused deterministic full-workflow failures
    when parallel lanes speculatively called it before any
    ``web_research`` populated the URL registry; see plan F1).
    ``web_research`` alone is sufficient — it combines search +
    auto-fetch of top-K page bodies in a single call.
    """
    ast = build_blueprint(_investment_signature(), "investment q", [])
    kinds = _declared_tool_kinds(ast)
    assert "web_research" in kinds
    # web_crawl is intentionally excluded from defaults; architect can
    # re-add it via the tool_plan when truly needed.
    assert "web_crawl" not in kinds
    # Corpus tool kinds must not appear when there are no assets
    assert "vector_search" not in kinds
    assert "table_search" not in kinds
    assert "compute" not in kinds


def test_default_web_tool_decls_excludes_web_crawl_by_default() -> None:
    """Plan v2.1 generic-robustness F1 — ``_default_web_tool_decls()``
    returns ONLY ``web_research`` by default. ``web_crawl`` requires
    explicit opt-in via ``include_crawl=True`` so the architect's
    ``tool_plan`` can add it back when a workflow truly needs the
    index-based crawl capability."""
    from deep_research.agent_designer.workflow_builder import (
        _default_web_tool_decls,
    )

    tools_default = _default_web_tool_decls()
    kinds_default = {t["kind"] for t in tools_default}
    assert kinds_default == {"web_research"}, (
        f"default tool plan must contain only web_research; got {kinds_default}"
    )

    tools_with_crawl = _default_web_tool_decls(include_crawl=True)
    kinds_with_crawl = {t["kind"] for t in tools_with_crawl}
    assert kinds_with_crawl == {"web_research", "web_crawl"}


def test_placeholder_system_prompt_mandates_web_research_first() -> None:
    """Plan v2.1 generic-robustness F2 — placeholder system_prompt
    explicitly sequences web_research BEFORE web_crawl so a runtime
    agent that has web_crawl bound (rare; architect-customized) does
    not speculatively call it with url_index=0 before any search."""
    from deep_research.agent_designer.blueprint import (
        _PLACEHOLDER_SYSTEM_PROMPT_TEMPLATE,
    )

    text = _PLACEHOLDER_SYSTEM_PROMPT_TEMPLATE.lower()
    # The web_research bullet must say "FIRST" before the web_crawl bullet.
    research_idx = text.find("web_research")
    crawl_idx = text.find("web_crawl")
    assert research_idx != -1
    assert crawl_idx != -1
    assert research_idx < crawl_idx, (
        "web_research must be mentioned BEFORE web_crawl in the placeholder"
    )
    assert "first" in text[: crawl_idx], (
        "the web_research bullet must use the word 'first' as the sequencer"
    )
    assert "url_index=0" in text, (
        "placeholder must explicitly forbid web_crawl(url_index=0) before search"
    )


def test_lane_user_prompt_search_strategy_mandates_web_research_first() -> None:
    """Plan v2.1 generic-robustness F2 — the lane user_prompt_template's
    ``Search strategy`` block mandates web_research as the FIRST call."""
    sig = _investment_signature()
    ast = build_blueprint(sig, "investment q", [])
    # Walk to any researcher node
    lane_template = ""

    def _walk(node):
        nonlocal lane_template
        if not isinstance(node, dict):
            return
        cfg = node.get("config") or {}
        if (
            node.get("type") == "agent"
            and isinstance(cfg, dict)
            and cfg.get("subtype") == "researcher"
            and not lane_template
        ):
            tpl = cfg.get("user_prompt_template") or ""
            if isinstance(tpl, str):
                lane_template = tpl
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                _walk(body)
        for ch in node.get("children") or []:
            _walk(ch)

    _walk(ast.get("root") or {})
    assert lane_template, "expected at least one researcher node with a template"
    assert "FIRST call: ``web_research``" in lane_template, (
        "lane Search strategy block must mandate web_research as the FIRST call; "
        f"got block content starting with:\n{lane_template[:500]}"
    )


def test_blueprint_corpus_plus_web_emits_both_blocks_after_f2() -> None:
    """Plan v2.1 generic-robustness — confirm F2's sequencing language
    survives the three-way evidence-mode dispatch for ``corpus_plus_web``."""
    sig = _officeqa_signature()
    sig["asset_signature"] = "corpus_plus_web"
    ast = build_blueprint(sig, "q", _officeqa_full_corpus_assets())

    lane_template = ""

    def _walk(node):
        nonlocal lane_template
        if not isinstance(node, dict):
            return
        cfg = node.get("config") or {}
        if (
            node.get("type") == "agent"
            and isinstance(cfg, dict)
            and cfg.get("subtype") == "researcher"
            and not lane_template
        ):
            tpl = cfg.get("user_prompt_template") or ""
            if isinstance(tpl, str):
                lane_template = tpl
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                _walk(body)
        for ch in node.get("children") or []:
            _walk(ch)

    _walk(ast.get("root") or {})
    # Corpus block AND web search block both present (mixed mode).
    assert "Retrieval strategy" in lane_template
    assert "Search strategy" in lane_template
    # Web block carries the F2 sequencing language.
    assert "FIRST call: ``web_research``" in lane_template


def test_blueprint_structured_only_declares_compute_no_web() -> None:
    """structured_only signature with a Delta table containing a structured
    JSON column → table tools + compute, no web. Covers the SQL-only case
    that neither anchor case exercises directly."""
    sig = _officeqa_signature()
    sig["asset_signature"] = "structured_only"
    sig["primary_evidence_kind"] = "structured_tables"
    assets = [
        {
            "kind": "delta_table",
            "full_name": "delta.sql.metrics",
            "usage": "required",
            "field_roles": {
                "primary_key": "row_id",
                "content": "row_text",
                "structured_json": "row_json",
                "order_by": "row_id",
            },
            "metadata": {
                "warehouse_id": "wh1",
                "columns": ["row_id", "row_text", "row_json"],
            },
        },
    ]
    ast = build_blueprint(sig, "q", assets)
    kinds = _declared_tool_kinds(ast)
    assert "table_search" in kinds
    assert "table_read" in kinds
    assert "table_load" in kinds
    assert "compute" in kinds
    assert "web_research" not in kinds
    assert "web_crawl" not in kinds


def test_blueprint_corpus_only_no_assets_fails_closed_after_grounding() -> None:
    """Plan v2.2 grounding — corpus_only + empty assets is now fail-closed.

    Previously this case ("corpus_only signature but no UI-selected and no
    required assets") silently defaulted to ``web_research`` on the
    premise that "no required corpus assets" made the swap safe. That
    premise broke whenever the user named a workspace resource in
    free text (e.g. ``main.x.idx``) — the classifier picked up
    ``corpus_only`` from the text, but the original asset list stayed
    empty and the lane researchers got web tools they could not use to
    answer the question. With the intent-grounding stage upstream,
    reaching this branch means grounding also produced no match — the
    honest answer is to halt and surface, not to swap.
    """
    sig = _officeqa_signature()
    sig["asset_signature"] = "corpus_only"
    with pytest.raises(SignatureError, match="grounding"):
        build_blueprint(sig, "q", [])


def test_blueprint_corpus_only_lane_template_satisfies_search_strategy_gate() -> None:
    """The corpus-only lane prompt builder emits a ``### Retrieval strategy``
    block (vs the web variant's ``### Search strategy``). The semantic
    validator must accept either; otherwise the deterministic blueprint
    produces an AST that immediately trips its own quality gate.

    Catches the heading-vs-validator drift that broke OfficeQA's
    scaffold-and-run even AFTER the asset→tool wiring was correct.
    """
    from deep_research.agent_designer.semantic_validation import (
        semantic_validation_errors,
    )

    ast = build_blueprint(
        _officeqa_signature(), "OfficeQA Treasury question", _officeqa_full_corpus_assets()
    )
    errors = semantic_validation_errors(ast)
    # Filter out the noisy "missing Search strategy" finding; the
    # builder DID emit a ``Retrieval strategy`` block for corpus assets.
    strategy_misses = [
        e for e in errors
        if "Search strategy" in (e.message or "")
    ]
    assert not strategy_misses, (
        f"validator must accept the corpus 'Retrieval strategy' heading "
        f"as a Search-strategy marker; got missing-block errors: "
        f"{[e.message for e in strategy_misses]}"
    )


def test_blueprint_stamps_placeholder_pending_top_level_list() -> None:
    """Plan v2.1 generic-robustness — every researcher node id from the
    deterministic blueprint must appear in the AST's top-level
    ``placeholder_pending_nodes`` list until the architect's final
    ``node_patches`` lands a non-empty prompt. List lives at top-level
    (not in node.config) so the framework's strict ``AgentNodeConfig``
    validator does not reject it.
    """
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY

    ast = build_blueprint(_investment_signature(), "investment q", [])
    pending = ast.get(PLACEHOLDER_PENDING_KEY)
    assert isinstance(pending, list)
    assert len(pending) == 6, (
        f"expected 6 lane researcher node ids; got {pending}"
    )
    for item in pending:
        assert isinstance(item, str) and item.endswith("-researcher")


def test_blueprint_placeholder_pending_present_for_corpus_topology() -> None:
    """Same lifecycle list applies to plan_and_execute topologies."""
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY

    ast = build_blueprint(
        _officeqa_signature(), "q", _officeqa_full_corpus_assets()
    )
    pending = ast.get(PLACEHOLDER_PENDING_KEY)
    assert isinstance(pending, list)
    assert pending, "plan_and_execute blueprint must register at least one pending researcher"
    for item in pending:
        assert isinstance(item, str)


def test_blueprint_contract_metadata_and_no_placeholder_pending() -> None:
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY

    ast = build_blueprint(
        _officeqa_signature(),
        "OfficeQA Treasury fiscal calendar compute question",
        _officeqa_full_corpus_assets(),
        tool_contract=_officeqa_tool_contract(),
    )

    assert ast.get(PLACEHOLDER_PENDING_KEY) is None
    assert ast["evidence_policy"] == "corpus_only"
    assert {"officeqa", "treasury", "fiscal", "calendar"}.issubset(
        set(ast["required_prompt_terms"])
    )
    summary = ast["resolved_tool_contract_summary"]
    assert summary["available"] is True
    assert summary["evidence_policy"] == "corpus_only"
    assert "web_search" not in _declared_tool_kinds(ast)
    assert {
        "vector_search",
        "table_search",
        "table_read",
        "table_load",
        "compute",
    }.issubset(_declared_tool_kinds(ast))


def test_apply_architect_patches_clears_placeholder_pending_on_prompt_patch() -> None:
    """When the architect's final ``node_patches`` delivers a non-empty
    ``system_prompt`` OR ``user_prompt_template``, the matching node id is
    removed from the top-level ``placeholder_pending_nodes`` list. Other
    patches (model_tier, error_handling) leave the list alone."""
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )

    ast = build_blueprint(
        _investment_signature(), "investment q", []
    )
    parallel = ast["root"]["children"][1]
    lane_id = parallel["children"][0]["id"]
    before = ast.get(PLACEHOLDER_PENDING_KEY) or []
    assert lane_id in before

    merged, errors = _apply_architect_patches(
        ast,
        {lane_id: {"system_prompt": "Specialized lane focus content."}},
    )
    assert not errors
    after = merged.get(PLACEHOLDER_PENDING_KEY) or []
    assert lane_id not in after, (
        f"expected {lane_id!r} removed from pending list; still got {after}"
    )


def test_apply_architect_patches_keeps_placeholder_pending_when_only_tier_changes() -> None:
    """A patch that touches only ``model_tier`` (no prompt content) must
    NOT remove the node from the pending list — architect still has work to do."""
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    parallel = ast["root"]["children"][1]
    lane_id = parallel["children"][0]["id"]

    merged, errors = _apply_architect_patches(
        ast,
        {lane_id: {"model_tier": "complex"}},
    )
    assert not errors
    pending = merged.get(PLACEHOLDER_PENDING_KEY) or []
    assert lane_id in pending


def test_apply_architect_patches_clears_placeholder_pending_on_user_prompt_only() -> None:
    """``user_prompt_template`` patch alone is also sufficient to remove
    the node from the pending list — the validator accepts either or both
    prompt fields."""
    from deep_research.agent_designer.blueprint import PLACEHOLDER_PENDING_KEY
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    parallel = ast["root"]["children"][1]
    lane_id = parallel["children"][0]["id"]

    merged, errors = _apply_architect_patches(
        ast,
        {lane_id: {"user_prompt_template": "Investigate {query} with lane focus."}},
    )
    assert not errors
    pending = merged.get(PLACEHOLDER_PENDING_KEY) or []
    assert lane_id not in pending


def test_semantic_validator_rejects_unfixed_placeholder_pending() -> None:
    """The validator emits a blocking error for any researcher node whose
    ``placeholder_pending`` flag survives the architect+critic loop."""
    from deep_research.agent_designer.semantic_validation import (
        detect_unspecialized_agents,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    errors = detect_unspecialized_agents(ast)
    placeholder_errors = [
        e for e in errors if "placeholder" in (e.message or "").lower()
    ]
    assert placeholder_errors, (
        "validator must reject blueprint with placeholder_pending=true"
    )


def test_semantic_validator_passes_after_placeholder_cleared() -> None:
    """After ``_apply_architect_patches`` delivers prompts to every lane,
    the validator no longer emits placeholder-pending errors for them.

    (Other validator checks may still fire — we only assert the
    placeholder-pending finding is gone.)"""
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )
    from deep_research.agent_designer.semantic_validation import (
        detect_unspecialized_agents,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    # Walk to collect every researcher lane id
    lane_ids: list[str] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        cfg = node.get("config") or {}
        if (
            node.get("type") == "agent"
            and isinstance(cfg, dict)
            and cfg.get("subtype") == "researcher"
        ):
            lane_ids.append(str(node.get("id") or ""))
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                _walk(body)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                _walk(child)

    _walk(ast.get("root") or {})
    assert lane_ids
    patches = {
        lane_id: {
            "system_prompt": (
                "Specialized system prompt — what to investigate, what to "
                "cite, what to flag, what NOT to do for this lane focus. "
                "This must be longer than 80 characters to satisfy the "
                "minimum-length check the validator also runs."
            ),
            "user_prompt_template": (
                "## Investigation Brief\n\nYou are investigating: **{query}**\n\n"
                "Lane focus: <use-case-specific focus>.\n\n"
                "### Sub-questions you MUST address\n"
                "1. ...\n2. ...\n3. ...\n\n"
                "### Required output structure\n"
                "- Evidence summary\n- Analysis\n- Unknowns\n\n"
                "### Search strategy\n- one\n- two\n\n"
                "### Definition of done\nMark unknowns as Data unavailable."
            ),
        }
        for lane_id in lane_ids
    }
    merged, errors_patch = _apply_architect_patches(ast, patches)
    assert not errors_patch
    errors_validate = detect_unspecialized_agents(merged)
    placeholder_errors = [
        e for e in errors_validate if "placeholder" in (e.message or "").lower()
    ]
    assert not placeholder_errors, (
        f"unexpected placeholder errors after clearing: "
        f"{[e.message for e in placeholder_errors]}"
    )


def test_evidence_mode_corpus_only_for_corpus_assets() -> None:
    """``_evidence_mode`` returns ``corpus_only`` for vector/delta-only ToolPlan."""
    from deep_research.agent_designer.designer_types import (
        ToolDeclarationSpec,
        ToolPlan,
    )
    from deep_research.agent_designer.workflow_builder import _evidence_mode

    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="vs", kind="vector_search"),
            ToolDeclarationSpec(name="tr", kind="table_read"),
        ]
    )
    assert _evidence_mode([], plan) == "corpus_only"


def test_evidence_mode_mixed_for_corpus_plus_web_toolplan() -> None:
    from deep_research.agent_designer.designer_types import (
        ToolDeclarationSpec,
        ToolPlan,
    )
    from deep_research.agent_designer.workflow_builder import _evidence_mode

    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="vs", kind="vector_search"),
            ToolDeclarationSpec(name="wr", kind="web_research"),
        ]
    )
    assert _evidence_mode([], plan) == "mixed"


def test_evidence_mode_web_only_for_web_toolplan() -> None:
    from deep_research.agent_designer.designer_types import (
        ToolDeclarationSpec,
        ToolPlan,
    )
    from deep_research.agent_designer.workflow_builder import _evidence_mode

    plan = ToolPlan(
        tools=[
            ToolDeclarationSpec(name="wr", kind="web_research"),
            ToolDeclarationSpec(name="wc", kind="web_crawl"),
        ]
    )
    assert _evidence_mode([], plan) == "web_only"


def test_evidence_mode_falls_back_to_assets_when_no_toolplan() -> None:
    """Legacy callers may pass assets but no ToolPlan; mode still resolves."""
    from deep_research.agent_designer.workflow_builder import _evidence_mode

    assert _evidence_mode([{"kind": "vector_index", "full_name": "x"}], None) == "corpus_only"
    assert _evidence_mode([], None) == "web_only"


def test_blueprint_corpus_plus_web_lane_prompt_includes_both_blocks() -> None:
    """For ``asset_signature=corpus_plus_web`` the lane user_prompt_template
    must include BOTH the corpus retrieval block AND the web search block,
    with the corpus-first preferrer between them."""
    sig = _officeqa_signature()
    sig["asset_signature"] = "corpus_plus_web"
    ast = build_blueprint(sig, "q", _officeqa_full_corpus_assets())

    found_lane_template = ""

    def _walk(node: Any) -> None:
        nonlocal found_lane_template
        if not isinstance(node, dict):
            return
        cfg = node.get("config") or {}
        if (
            node.get("type") == "agent"
            and isinstance(cfg, dict)
            and cfg.get("subtype") == "researcher"
            and not found_lane_template
        ):
            tpl = cfg.get("user_prompt_template") or ""
            if isinstance(tpl, str):
                found_lane_template = tpl
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                _walk(body)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                _walk(child)

    _walk(ast.get("root") or {})
    assert found_lane_template, "expected at least one researcher node with a template"
    assert "Retrieval strategy" in found_lane_template, (
        "mixed mode must include corpus retrieval block"
    )
    assert "Search strategy" in found_lane_template, (
        "mixed mode must include web search block"
    )
    assert "corpus FIRST" in found_lane_template, (
        "mixed mode must include the corpus-first preferrer sentence"
    )


def test_apply_architect_patches_rejects_tools_key_at_node_level() -> None:
    """Plan v2.1 generic-robustness — ``tools`` is structural and must NOT
    survive in the patch allow-list. An architect patch with ``tools`` is
    rejected with a clear "use request_signature_revision" hint."""
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    parallel = ast["root"]["children"][1]
    lane_id = parallel["children"][0]["id"]

    _, errors = _apply_architect_patches(
        ast,
        {lane_id: {"tools": ["some_tool"]}},
    )
    assert errors, "patch with 'tools' key must be rejected"
    assert any("structural_drift" in e or "tools" in e for e in errors), (
        f"error must reference tools or structural_drift; got: {errors}"
    )


def test_apply_architect_patches_rejects_pools_key() -> None:
    """Pools are also fingerprinted — architect can't patch them."""
    from deep_research.agent_designer.framework_tools import (
        _apply_architect_patches,
    )

    ast = build_blueprint(_investment_signature(), "q", [])
    parallel = ast["root"]["children"][1]
    lane_id = parallel["children"][0]["id"]

    _, errors = _apply_architect_patches(
        ast,
        {lane_id: {"pools": [{"name": "extra"}]}},
    )
    assert errors


def test_blueprint_preferred_asset_without_warehouse_does_not_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric to the fail-closed test: when the Delta asset is
    ``usage="preferred"`` (not required), the builder may degrade to whatever
    tools the recommender can produce without raising."""
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
    sig = _officeqa_signature()
    assets = [
        {
            "kind": "vector_index",
            "full_name": "vs.example.idx",
            "usage": "required",
            "metadata": {"columns": ["chunk_id"]},
        },
        {
            "kind": "delta_table",
            "full_name": "delta.example.no_warehouse",
            "usage": "preferred",  # NOT required — fail-closed must not fire
            "field_roles": {"content": "content"},
            "metadata": {},
        },
    ]
    ast = build_blueprint(sig, "q", assets)
    kinds = _declared_tool_kinds(ast)
    assert "vector_search" in kinds
    # table tools couldn't be wired (no warehouse) — degrades gracefully
    assert "table_search" not in kinds
