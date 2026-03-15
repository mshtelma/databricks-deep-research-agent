from __future__ import annotations

from databricks_deep_research.agents.query_policy import (
    EvidenceContract,
    EvidenceQuality,
    QueryPolicyRegistry,
    RetrievalIntent,
    RetrievalNeed,
)
from databricks_deep_research.tools.protocol import SourceInfo, ToolDefinition, ToolResult


def _need(intent: RetrievalIntent = RetrievalIntent.fact_lookup, contract: EvidenceContract = EvidenceContract.ranked_sources) -> RetrievalNeed:
    return RetrievalNeed(
        root_query="What does Kroger revenue growth look like?",
        step_text="Analyze Kroger revenue growth from internal data",
        step_title="Analyze revenue growth",
        entities=["Kroger"],
        focus_terms=["revenue", "growth"],
        intent=intent,
        evidence_contract=contract,
    )


def test_registry_resolves_vector_policy() -> None:
    registry = QueryPolicyRegistry()
    definition = ToolDefinition(name="vector_search", description="", parameters={}, source_kind="vector_index")
    plan = registry.plan("vector_index", definition, _need(RetrievalIntent.document_retrieval, EvidenceContract.quoted_document_content), {"query": "Kroger earnings"})
    assert plan.query_strategy == "vector_entity_artifact"
    assert "research index first then" not in plan.rendered_query_text


def test_registry_resolves_sql_policy() -> None:
    registry = QueryPolicyRegistry()
    definition = ToolDefinition(name="genie", description="", parameters={}, source_kind="sql_analytics")
    plan = registry.plan("sql_analytics", definition, _need(RetrievalIntent.metric_slice, EvidenceContract.numeric_table), {"question": "How is Kroger doing?"})
    assert plan.query_strategy == "genie_metric_slice"
    assert "Show" in plan.rendered_query_text or "Break down" in plan.alternate_argument_sets[0]["question"]


def test_vector_assessment_accepts_metadata_as_low_value() -> None:
    registry = QueryPolicyRegistry()
    definition = ToolDefinition(name="vector_search", description="", parameters={}, source_kind="vector_index")
    raw_sources = [{"url": "enterprise://vector_search/test/1", "title": "Transcript", "snippet": "Header only", "content": None}]
    outcome = registry.assess("vector_index", definition, ToolResult(content="Found transcript", sources=[SourceInfo(url="enterprise://vector_search/test/1", title="Transcript", snippet="Header only")]), _need(RetrievalIntent.quote_extraction, EvidenceContract.quoted_document_content), raw_sources)
    assert outcome.accepted_low_value_sources
    assert outcome.evidence_quality == EvidenceQuality.metadata_only
    assert outcome.needs_adaptation is True


def test_sql_assessment_marks_availability_only_as_insufficient() -> None:
    registry = QueryPolicyRegistry()
    definition = ToolDefinition(name="genie", description="", parameters={}, source_kind="sql_analytics")
    raw_sources = [{"url": "enterprise://genie/warehouse", "title": "Warehouse", "snippet": "Data available"}]
    outcome = registry.assess("sql_analytics", definition, ToolResult(content="Query accepted and data exists in the warehouse.", sources=[SourceInfo(url="enterprise://genie/warehouse", title="Warehouse", snippet="Data available")]), _need(RetrievalIntent.metric_slice, EvidenceContract.numeric_table), raw_sources)
    assert outcome.failure_mode.value == "availability_only"
    assert outcome.sufficient_for_step is False
    assert outcome.needs_adaptation is True
