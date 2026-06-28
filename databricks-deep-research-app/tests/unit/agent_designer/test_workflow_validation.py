"""Unit tests for the unified workflow validator service (US-101).

Covers the contract that makes the gate reliable: content-addressed caching
(skip-if-unchanged), never caching a fallback/skipped verdict, the semantic
projection's stability under non-semantic churn (generated ids, save-time tool
config materialization) and its sensitivity to real semantic change, and the
build-loop adapter.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.agent_designer import workflow_validation as wv
from deep_research.agent_designer.workflow_critic import (
    AgentFinding,
    CoverageGap,
    CritiqueResult,
    OutputGap,
)
from deep_research.agent_designer.workflow_validation import (
    ValidationSource,
    WorkflowValidationResult,
    semantic_projection,
    to_critic_verdict,
    validate_workflow,
)


def _ast() -> dict[str, Any]:
    return {
        "root": {
            "type": "agent",
            "id": "node-aaa-111",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "system_prompt": "Research the treasury corpus and answer precisely.",
                "model_tier": "analytical",
                "tools": ["web_search"],
            },
            "children": [],
        },
        "tools": [{"name": "web_search", "kind": "web", "config": {}}],
    }


class _FakeCache:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str, str], WorkflowValidationResult] = {}
        self.puts = 0

    async def get(
        self, *, validator_version: str, intent_hash: str, semantic_hash: str
    ) -> WorkflowValidationResult | None:
        return self.store.get((validator_version, intent_hash, semantic_hash))

    async def put(self, result: WorkflowValidationResult) -> None:
        self.puts += 1
        self.store[
            (result.validator_version, result.intent_hash, result.semantic_hash)
        ] = result


def _patch_critic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict: str = "pass",
    is_fallback: bool = False,
    counter: list[int] | None = None,
) -> None:
    async def _fake(**_: Any) -> tuple[CritiqueResult, bool]:
        if counter is not None:
            counter.append(1)
        return CritiqueResult(verdict=verdict, summary=f"verdict={verdict}"), is_fallback

    monkeypatch.setattr(wv, "critique_workflow_against_intent_ex", _fake)


# --- skip paths -------------------------------------------------------------


@pytest.mark.asyncio
async def test_skipped_when_no_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    _patch_critic(monkeypatch, counter=calls)
    result = await validate_workflow(
        definition=_ast(), intent="   ", llm=object(), cache=_FakeCache()
    )
    assert result.verdict == "skipped"
    assert result.source is ValidationSource.SKIPPED
    assert result.cacheable is False
    assert calls == []  # no LLM call


@pytest.mark.asyncio
async def test_skipped_when_no_llm() -> None:
    result = await validate_workflow(definition=_ast(), intent="find X", llm=None)
    assert result.verdict == "skipped"
    assert result.source is ValidationSource.SKIPPED
    assert result.cacheable is False


# --- cache behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_then_cache_hit_skips_second_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    _patch_critic(monkeypatch, verdict="pass", counter=calls)
    cache = _FakeCache()

    first = await validate_workflow(
        definition=_ast(), intent="find X", llm=object(), cache=cache
    )
    assert first.source is ValidationSource.FRESH
    assert first.cacheable is True
    assert cache.puts == 1
    assert len(calls) == 1

    second = await validate_workflow(
        definition=_ast(), intent="find X", llm=object(), cache=cache
    )
    assert second.source is ValidationSource.CACHE
    assert second.cache_hit is True
    assert second.verdict == "pass"
    assert len(calls) == 1  # the LLM was NOT called again


@pytest.mark.asyncio
async def test_fallback_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_critic(monkeypatch, verdict="needs_revision", is_fallback=True)
    cache = _FakeCache()
    result = await validate_workflow(
        definition=_ast(), intent="find X", llm=object(), cache=cache
    )
    assert result.source is ValidationSource.FALLBACK
    assert result.cacheable is False
    assert cache.puts == 0  # a non-judgment must never poison the cache


# --- semantic projection ----------------------------------------------------


def test_projection_stable_under_id_churn() -> None:
    a = _ast()
    b = copy.deepcopy(a)
    b["root"]["id"] = "node-zzz-999"  # regenerated UUID, no semantic change
    assert semantic_projection(a, "find X", None) == semantic_projection(b, "find X", None)


def test_projection_invariant_under_materialize_for_save() -> None:
    """The documented risk: save validates the post-materialize definition while
    the build validates the in-progress AST. The projection must be invariant so
    an unchanged workflow stays a cache hit."""
    from deep_research.agent_designer.ast_normalizer import (
        apply_web_search_provider_defaults,
    )
    from deep_research.agent_designer.catalog_service import CatalogService

    a = _ast()
    before = semantic_projection(a, "find X", None)

    materialized = CatalogService().materialize_for_save(copy.deepcopy(a))
    assert semantic_projection(materialized, "find X", None) == before

    filled = copy.deepcopy(a)
    apply_web_search_provider_defaults(filled)  # stamps endpoint/model into tool config
    assert semantic_projection(filled, "find X", None) == before


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda d: d["root"]["config"].update(system_prompt="totally different"),
            id="system_prompt",
        ),
        pytest.param(
            lambda d: d["root"]["config"].update(model_tier="complex"), id="model_tier"
        ),
        pytest.param(
            lambda d: d["root"]["config"].update(tools=["web_search", "vector_search"]),
            id="tools_bound",
        ),
        pytest.param(
            lambda d: d["tools"].append({"name": "vector_search", "kind": "vector_index", "config": {}}),
            id="tool_decl_kind",
        ),
        pytest.param(
            lambda d: d["root"]["config"].update(subtype="synthesizer"), id="subtype"
        ),
    ],
)
def test_projection_changes_on_semantic_change(mutate: Any) -> None:
    a = _ast()
    before = semantic_projection(a, "find X", None)
    b = copy.deepcopy(a)
    mutate(b)
    assert semantic_projection(b, "find X", None) != before


def test_projection_changes_on_intent_change() -> None:
    a = _ast()
    assert semantic_projection(a, "find X", None) != semantic_projection(a, "find Y", None)


# --- build-loop adapter -----------------------------------------------------


def _result(verdict: str, **kw: Any) -> WorkflowValidationResult:
    return WorkflowValidationResult(
        verdict=verdict,  # type: ignore[arg-type]
        summary="s",
        semantic_hash="h",
        intent_hash="i",
        validator_version=wv.VALIDATOR_VERSION,
        source=ValidationSource.FRESH,
        **kw,
    )


def test_to_critic_verdict_pass_approves() -> None:
    assert to_critic_verdict(_result("pass")).approve is True


def test_to_critic_verdict_skipped_approves() -> None:
    assert to_critic_verdict(_result("skipped")).approve is True


def test_to_critic_verdict_fail_rejects_with_directives() -> None:
    from deep_research.agent_designer.critic_types import CriticDirective

    res = _result(
        "fail",
        directives=[CriticDirective(node_path="root", issue="x", suggested_action="y")],
    )
    verdict = to_critic_verdict(res)
    assert verdict.approve is False
    assert len(verdict.directives) == 1


def test_directives_from_critique_maps_findings_and_gaps() -> None:
    critique = CritiqueResult(
        verdict="needs_revision",
        summary="s",
        agent_findings=[
            AgentFinding(
                node_path="root.children[0]",
                label="L",
                severity="fail",
                finding="off-topic",
                suggested_action="update_block: rewrite",
            ),
            AgentFinding(
                node_path="root.children[1]",
                label="M",
                severity="minor",
                finding="sharpen",
                suggested_action="update_block: tweak",
            ),
        ],
        coverage_gaps=[CoverageGap(aspect="pricing", rationale="no agent covers it")],
        output_gaps=[OutputGap(required_output="table", rationale="no synthesizer")],
    )
    directives = wv.directives_from_critique(critique)
    assert len(directives) == 4
    sev = {d.issue: d.severity for d in directives}
    assert sev["off-topic"] == "blocking"
    assert sev["sharpen"] == "advisory"
