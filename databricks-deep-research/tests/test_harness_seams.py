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
    assert batches[("observations", "findings")].items == ["fact"]
    assert batches[("sources", "sources")].items == [{"url": "https://example.com", "title": "Example"}]


def test_project_research_state_writes_structured_payload() -> None:
    state = WorkflowState(query="q")
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    normalized = _normalize_research_output({"findings": "abc", "search_queries": ["x"]}, config, [])
    assert normalized is not None
    state_output, structured = _project_research_state("node", config, state, normalized)
    assert state_output == "abc"
    assert structured["findings"] == "abc"
    assert structured["observation"] == "abc"
    assert state.get("research_status") is not None


def test_project_research_state_module_helper_matches_harness_shim() -> None:
    state = WorkflowState(query="q")
    config = AgentNodeConfig(subtype="researcher", output_key="findings")
    normalized = _normalize_research_output({"findings": "abc", "search_queries": ["x"]}, config, [])
    assert normalized is not None
    state_output, structured = project_research_state("node", config, state, normalized)
    assert state_output == "abc"
    assert structured["findings"] == "abc"
    assert structured["observation"] == "abc"
    assert state.get("research_status") is not None
