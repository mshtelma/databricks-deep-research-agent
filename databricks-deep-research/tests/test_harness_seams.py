from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig, PoolWriteConfig
from databricks_deep_research.agents.execution.state_projection import project_research_state
from databricks_deep_research.agents.harness import (
    _build_pool_batches,
    _normalize_research_output,
    _project_research_state,
)
from databricks_deep_research.workflow.state import WorkflowState


def test_normalize_research_output_plain_text_sets_observation_fields() -> None:
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    normalized = _normalize_research_output("hello world", config, [])
    assert normalized is not None
    assert normalized.state_text == "hello world"
    assert normalized.observation_text == "hello world"
    assert normalized.findings_text == "hello world"
    assert normalized.research_status == "ok"


def test_build_pool_batches_creates_text_and_source_batches() -> None:
    config = AgentNodeConfig(
        subtype="researcher",
        output_key="findings",
        pool_writes=[
            PoolWriteConfig(pool="observations", extract="findings"),
            PoolWriteConfig(pool="sources", extract="sources"),
        ],
    )
    normalized = _normalize_research_output(
        {
            "findings": "fact",
            "sources": [{"url": "https://example.com", "title": "Example"}],
        },
        config,
        [],
    )
    assert normalized is not None
    batches = _build_pool_batches(normalized, config.pool_writes, config.output_key)
    # Normalizer serializes the full dict — "fact" is inside the JSON
    assert "fact" in batches[("observations", "findings")].items[0]
    assert batches[("sources", "sources")].items == [{"url": "https://example.com", "title": "Example"}]


def test_project_research_state_writes_structured_payload() -> None:
    state = WorkflowState(query="q")
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    normalized = _normalize_research_output({"findings": "abc", "search_queries": ["x"]}, config, [])
    assert normalized is not None
    state_output, structured = _project_research_state("node", config, state, normalized)
    # Normalizer now serializes full dict — "abc" is inside the JSON
    assert "abc" in state_output
    assert "abc" in structured.get("findings", "")
    assert state.get("research_status") is not None


def test_normalize_output_excludes_sources_from_state_text() -> None:
    """Sources should be stripped from state_text but still in normalized.sources."""
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    parsed = {
        "operands": [{"id": "op1", "value": 42}],
        "sources": [{"url": "https://example.com", "title": "Ex"}],
        "sources_found": 1,
        "sources_used": [],
    }
    normalized = _normalize_research_output(parsed, config, [])
    assert normalized is not None
    # Sources excluded from state_text
    assert '"sources"' not in normalized.state_text
    assert '"sources_found"' not in normalized.state_text
    assert '"sources_used"' not in normalized.state_text
    # But operands preserved
    assert '"operands"' in normalized.state_text
    assert "42" in normalized.state_text
    # Sources still available via the sources field
    assert len(normalized.sources) == 1
    assert normalized.sources[0]["url"] == "https://example.com"


def test_project_research_state_module_helper_matches_harness_shim() -> None:
    state = WorkflowState(query="q")
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    normalized = _normalize_research_output({"findings": "abc", "search_queries": ["x"]}, config, [])
    assert normalized is not None
    state_output, structured = project_research_state("node", config, state, normalized)
    assert "abc" in state_output
    assert "abc" in structured.get("findings", "")
    assert state.get("research_status") is not None
