from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import deep_research.agent_designer.tool_contract as tool_contract_module
from deep_research.agent_designer.designer_types import (
    ResolvedToolContract,
    ResourceSemanticExtraction,
    ResourceSemanticItem,
)
from deep_research.agent_designer.prompt_grounding import ground_prompt
from deep_research.agent_designer.tool_contract import (
    extract_resource_semantics_structured,
    project_resolved_tool_contract,
    sanitized_resolved_tool_contract_summary,
    validate_resource_semantics,
)

_OFFICEQA_RESOURCE_INTENT = """
Build an OfficeQA-style research assistant for Databricks assets. Infer the
needed tool configuration from these resource names:
- Vector index: main.officeqa_benchmark.treasury_chunks_vs_index
- Delta table for exact Treasury document chunk reads:
  main.officeqa_benchmark.treasury_chunks
- Delta table for structured Treasury bulletin tables:
  main.officeqa_benchmark.treasury_tables

Use the vector index to find candidate Treasury document chunks, use the
Delta tables for exact text/table reads, use compute for numeric table
calculations, and synthesize a concise answer with the fiscal/calendar-year
distinction when the evidence supports it.
"""
_OFFICEQA_INTENT = _OFFICEQA_RESOURCE_INTENT + "Do not use public web tools.\n"


def test_schema_aliases_accept_and_emit_public_contract_keys() -> None:
    semantics = ResourceSemanticExtraction.model_validate(
        {"schema": "resource_semantics.v1"}
    )
    contract = ResolvedToolContract.model_validate(
        {"schema": "resolved_tool_contract.v1"}
    )

    assert semantics.schema_ == "resource_semantics.v1"
    assert contract.schema_ == "resolved_tool_contract.v1"
    assert semantics.model_dump(mode="json", by_alias=True)["schema"] == (
        "resource_semantics.v1"
    )
    assert contract.model_dump(mode="json", by_alias=True)["schema"] == (
        "resolved_tool_contract.v1"
    )
    assert "schema_" not in semantics.model_dump(mode="json", by_alias=True)
    assert "schema_" not in contract.model_dump(mode="json", by_alias=True)


async def _officeqa_grounding(default_warehouse_id: str | None = "wh-officeqa"):
    return await ground_prompt(
        intent=_OFFICEQA_INTENT,
        existing_assets=[],
        discovery=None,
        default_warehouse_id=default_warehouse_id,
    )


@pytest.mark.asyncio
async def test_officeqa_contract_is_corpus_only_and_forbids_web_when_prompt_says_so() -> None:
    grounding = await _officeqa_grounding()

    contract = project_resolved_tool_contract(grounding, intent=_OFFICEQA_INTENT)

    assert contract is not None
    assert contract.evidence_policy == "corpus_only"
    assert {
        "vector_search",
        "table_search",
        "table_read",
        "table_load",
        "compute",
    }.issubset(set(contract.ready_tool_kinds))
    assert set(contract.prompt_obligations.forbidden_tool_kinds) == {
        "web_search",
        "web_crawl",
        "web_research",
    }
    assert {
        "main.officeqa_benchmark.treasury_chunks_vs_index",
        "main.officeqa_benchmark.treasury_chunks",
        "main.officeqa_benchmark.treasury_tables",
    }.issubset({resource.identity for resource in contract.resources})


@pytest.mark.asyncio
async def test_corpus_only_contract_does_not_forbid_web_without_prompt_policy() -> None:
    grounding = await ground_prompt(
        intent=_OFFICEQA_RESOURCE_INTENT,
        existing_assets=[],
        discovery=None,
        default_warehouse_id="wh-officeqa",
    )

    contract = project_resolved_tool_contract(
        grounding,
        intent=_OFFICEQA_RESOURCE_INTENT,
    )

    assert contract is not None
    assert contract.evidence_policy == "corpus_only"
    assert contract.prompt_obligations.forbidden_tool_kinds == []
    assert not any(
        "public web" in obligation.casefold()
        for obligation in contract.prompt_obligations.planner_obligations
    )


@pytest.mark.asyncio
async def test_contract_derives_fiscal_calendar_obligation_without_semantics() -> None:
    grounding = await ground_prompt(
        intent=_OFFICEQA_RESOURCE_INTENT,
        existing_assets=[],
        discovery=None,
        default_warehouse_id="wh-officeqa",
    )

    contract = project_resolved_tool_contract(
        grounding,
        intent=_OFFICEQA_RESOURCE_INTENT,
    )

    assert contract is not None
    assert any(
        "fiscal/calendar-year distinction" in obligation
        for obligation in contract.prompt_obligations.synthesis_obligations
    )


@pytest.mark.asyncio
async def test_structured_semantic_terms_are_validated_and_projected() -> None:
    grounding = await _officeqa_grounding()
    semantics = ResourceSemanticExtraction(
        resources=[
            ResourceSemanticItem(
                identity="main.officeqa_benchmark.treasury_chunks_vs_index",
                role_description="semantic lookup over Treasury chunks",
                domain_terms=["officeqa", "treasury", "vector", "chunks"],
                intended_operations=["find candidate document chunks"],
            )
        ],
        task_domain_terms=["fiscal", "calendar", "compute"],
        answer_obligations=[
            "Preserve the fiscal/calendar-year distinction when evidence supports it."
        ],
    )

    contract = project_resolved_tool_contract(
        grounding,
        intent=_OFFICEQA_INTENT,
        semantics=semantics,
    )

    assert contract is not None
    terms = set(contract.prompt_obligations.required_terms)
    assert {"officeqa", "treasury", "fiscal", "calendar", "compute"}.issubset(terms)
    assert contract.resources[0].role_description
    assert contract.prompt_obligations.synthesis_obligations


@pytest.mark.asyncio
async def test_verbose_semantics_keep_resource_anchors_in_required_terms() -> None:
    """Scaffold-flake regression: when the advisory extraction emits a full
    slate (>= _MAX_TERMS) of semantic phrases, the deterministic resource /
    capability anchors must still survive in required_terms (and in the
    summary's required_terms[:12]) — otherwise the contract drops the named
    corpus and the live scaffold gate fails on LLM verbosity."""
    grounding = await _officeqa_grounding()
    semantics = ResourceSemanticExtraction(
        resources=[
            ResourceSemanticItem(
                identity="main.officeqa_benchmark.treasury_chunks_vs_index",
                role_description="semantic lookup over Treasury chunks",
                domain_terms=[
                    "candidate ranking", "vector search", "document chunks",
                    "semantic retrieval", "exact retrieval", "treasury corpus",
                ],
                intended_operations=["find candidate document chunks"],
            )
        ],
        task_domain_terms=[
            "calendar year distinction", "evidence synthesis",
            "officeqa-style research assistant", "question answering",
            "treasury documents", "fiscal year distinction", "answer drafting",
        ],
        answer_obligations=["Preserve the fiscal/calendar-year distinction."],
    )

    contract = project_resolved_tool_contract(
        grounding, intent=_OFFICEQA_INTENT, semantics=semantics,
    )
    assert contract is not None

    required_terms = set(contract.prompt_obligations.required_terms)
    assert {"officeqa", "treasury", "chunks", "vector", "compute"} & required_terms, (
        f"resource anchors truncated out of required_terms: "
        f"{contract.prompt_obligations.required_terms}"
    )
    # The scaffold checks the sanitized summary (required_terms[:12]).
    summary = sanitized_resolved_tool_contract_summary(contract)
    assert {"officeqa", "treasury", "chunks", "vector", "compute"} & set(
        summary["required_terms"]
    )


@pytest.mark.asyncio
async def test_structured_llm_semantics_call_accepts_valid_response() -> None:
    grounding = await _officeqa_grounding()

    class _FakeLLM:
        async def complete(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                structured=ResourceSemanticExtraction(
                    resources=[
                        ResourceSemanticItem(
                            identity=(
                                "main.officeqa_benchmark.treasury_chunks_vs_index"
                            ),
                            role_description="vector retrieval over Treasury chunks",
                            domain_terms=["treasury", "vector"],
                            intended_operations=["semantic lookup"],
                        )
                    ],
                    task_domain_terms=["officeqa", "calendar"],
                    answer_obligations=["Use evidence before answering."],
                )
            )

    semantics, diagnostics = await extract_resource_semantics_structured(
        llm=_FakeLLM(),
        intent=_OFFICEQA_INTENT,
        grounding=grounding,
    )

    assert diagnostics == []
    assert semantics is not None
    assert semantics.task_domain_terms == ["officeqa", "calendar"]
    assert semantics.resources[0].identity == (
        "main.officeqa_benchmark.treasury_chunks_vs_index"
    )


@pytest.mark.asyncio
async def test_semantic_validation_discards_invented_resources_and_config() -> None:
    grounding = await _officeqa_grounding()
    raw = {
        "resources": [
            {
                "identity": "main.officeqa_benchmark.treasury_chunks_vs_index",
                "domain_terms": ["treasury"],
                "warehouse_id": "should-not-be-trusted",
            },
            {
                "identity": "main.officeqa_benchmark.invented_resource",
                "domain_terms": ["invented"],
            },
        ],
        "task_domain_terms": ["officeqa"],
        "answer_obligations": ["Use SELECT * FROM private.table"],
    }

    semantics, diagnostics = validate_resource_semantics(grounding, raw)

    assert semantics is not None
    assert semantics.resources == []
    assert semantics.answer_obligations == []
    assert {diagnostic["code"] for diagnostic in diagnostics} == {
        "semantic_executable_config_discarded",
        "semantic_resource_discarded",
    }


@pytest.mark.asyncio
async def test_invalid_semantics_fall_back_to_deterministic_terms() -> None:
    grounding = await _officeqa_grounding()

    contract = project_resolved_tool_contract(
        grounding,
        intent=_OFFICEQA_INTENT,
        semantics={"resources": [{"identity": "invented"}]},
    )

    assert contract is not None
    assert {"officeqa", "treasury", "chunks"}.issubset(
        set(contract.prompt_obligations.required_terms)
    )


@pytest.mark.asyncio
async def test_missing_warehouse_contract_carries_blocking_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)

    grounding = await _officeqa_grounding(default_warehouse_id=None)

    contract = project_resolved_tool_contract(grounding, intent=_OFFICEQA_INTENT)

    assert contract is not None
    assert grounding.safe_to_build_blueprint is False
    diagnostic_codes = {diagnostic.get("code") for diagnostic in contract.diagnostics}
    assert {"missing_warehouse_id", "safe_blueprint_blocked"}.issubset(
        diagnostic_codes
    )
    assert "table_read" not in contract.ready_tool_kinds


def test_contract_summary_is_prompt_safe_and_framework_code_not_officeqa_hardcoded() -> None:
    source = inspect.getsource(tool_contract_module).casefold()

    assert "officeqa" not in source
    assert "treasury" not in source
    assert sanitized_resolved_tool_contract_summary(None) == {
        "schema": "resolved_tool_contract.v1",
        "available": False,
    }
