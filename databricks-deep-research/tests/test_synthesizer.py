"""Tests for the synthesizer builtin subtype (T069)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Force registration of all builtins
import databricks_deep_research.agents.builtins  # noqa: F401
from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.grounding import resolve_grounding_mode
from databricks_deep_research.agents.harness import execute_agent
from databricks_deep_research.citation.pipeline import VerificationEvent
from databricks_deep_research.citation.types import (
    ClaimInfo,
    EvidenceInfo,
    VerificationSummaryInfo,
)
from databricks_deep_research.events.types import (
    ClaimGeneratedEvent,
    ClaimVerifiedEvent,
    VerificationSummaryEvent,
)
from databricks_deep_research.llm.client import LLMResponse
from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
from databricks_deep_research.workflow.state import WorkflowState


class TestSynthesizerPostProcess:
    def test_grounding_mode_defaults_to_none_without_legacy_schema(self) -> None:
        config = AgentNodeConfig(subtype="synthesizer")
        assert resolve_grounding_mode(config) == "none"

    def test_reclaim_post_process_prefers_framework_state(self) -> None:
        config = AgentNodeConfig(
            subtype="synthesizer",
            output_schema={"synthesis_mode": "reclaim"},
        )
        state = WorkflowState(query="test")
        state.runtime_store = TypedRuntimeStateStore(query="test", workflow_id="wf", workflow_name="wf")
        state.runtime_store.publish_verification_payload(
            producer_node_id="synth",
            payload={
                "claims": [{"claim_text": "Revenue grew.", "citation_keys": ["0"], "claim_role": "fact"}],
                "verifications": [],
                "corrections": [],
                "numeric_claims": [],
                "verification_summary": {
                    "total_claims": 1,
                    "verified_claims": 1,
                    "corrected_citations": 0,
                    "removed_claims": 0,
                    "softened_claims": 0,
                    "overall_confidence": 1.0,
                    "analysis_summary": {"total_claims": 0},
                },
                "analysis_summary": {"total_claims": 0},
            },
        )

        builtin = get_builtin("synthesizer")
        assert builtin is not None and builtin.post_process is not None

        events = builtin.post_process("synth", "report", config, state)

        assert any(isinstance(event, ClaimGeneratedEvent) for event in events)
        assert any(isinstance(event, VerificationSummaryEvent) for event in events)


@pytest.mark.asyncio
async def test_execute_agent_reclaim_runs_pipeline_and_writes_state() -> None:
    config = AgentNodeConfig(
        subtype="synthesizer",
        model_tier="analytical",
        output_key="report",
        system_prompt="You are a verifier.",
        user_prompt_template="Write a report about {query}.",
        output_schema={
            "synthesis_mode": "reclaim",
            "target_word_count": 200,
            "generation_mode": "strict",
        },
    )
    state = WorkflowState(query="How are Kroger earnings?")

    observations_pool = PoolState(PoolConfig(name="observations"))
    observations_pool.add("Kroger reported Q3 results.")
    sources_pool = PoolState(PoolConfig(name="sources", dedup_content_hash=False))
    sources_pool.add(
        {
            "url": "enterprise://vector_search/earnings/0",
            "title": "Kroger Q3 2025 Results",
            "snippet": "Kroger reported identical sales growth of 2.6%.",
            "content": "Kroger reported identical sales growth of 2.6%.",
            "source_type": "vector_search",
        }
    )
    pools = {"observations": observations_pool, "sources": sources_pool}
    state.pools = pools

    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="fallback content", usage={"total_tokens": 5})
    )

    fake_pipeline = MagicMock()
    fake_pipeline.last_evidence_pool = [
        MagicMock(
            source_url="enterprise://vector_search/earnings/0",
            source_title="",
        )
    ]
    fake_pipeline.last_generated_claims = [
        ClaimInfo(
            claim_text="Kroger reported identical sales growth of 2.6%.",
            claim_type="numeric",
            position_start=0,
            position_end=48,
            evidence=EvidenceInfo(
                source_url="enterprise://vector_search/earnings/0",
                quote_text="Kroger reported identical sales growth of 2.6%.",
            ),
            confidence_level="high",
            verification_verdict="supported",
            verification_reasoning="Directly stated in the source.",
            verification_method="numeric_qa",
            citation_key="Source",
            citation_keys=["Source"],
            claim_role="fact",
        )
    ]
    fake_pipeline.last_verification_summary = VerificationSummaryInfo(
        total_claims=1,
        supported_count=1,
        partial_count=0,
        unsupported_count=0,
        contradicted_count=0,
        abstained_count=0,
        unsupported_rate=0.0,
        contradicted_rate=0.0,
        warning=False,
            citation_corrections=0,
        )

    async def _run_full_pipeline(**_: object):
        yield "Kroger reported identical sales growth of 2.6% [Source]."
        yield VerificationEvent(
            event_type="claim_verified",
            data={
                "claim_index": 0,
                "verdict": "supported",
                "confidence": 0.9,
                "claim_role": "fact",
                "verification_method": "numeric_qa",
                "evidence_preview": "Kroger reported identical sales growth of 2.6%.",
            },
        )
        yield VerificationEvent(
            event_type="verification_summary",
            data={"total_claims": 1, "supported": 1, "analysis_summary": {"total_claims": 0}},
        )

    fake_pipeline.run_full_pipeline = _run_full_pipeline

    with patch(
        "databricks_deep_research.agents.builtins.synthesizer._build_reclaim_pipeline",
        return_value=fake_pipeline,
    ):
        output = await execute_agent("synth", config, state, llm, tools=[], pools=pools)

    assert output.content == "Kroger reported identical sales growth of 2.6% [0]."
    assert state.get("claims")[0]["citation_keys"] == ["0"]
    assert state.get("claims")[0]["claim_role"] == "fact"
    assert state.get("verification_summary")["total_claims"] == 1
    assert state.get("analysis_summary")["total_claims"] == 0
    assert any(isinstance(event, ClaimGeneratedEvent) for event in output.events)
    assert any(isinstance(event, ClaimVerifiedEvent) for event in output.events)
    assert any(isinstance(event, VerificationSummaryEvent) for event in output.events)
    llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_agent_none_mode_skips_citation_pipeline() -> None:
    config = AgentNodeConfig(
        subtype="synthesizer",
        model_tier="analytical",
        output_key="report",
        system_prompt="You are a synthesizer.",
        user_prompt_template="Write a report about {query}.",
        grounding_mode="none",
    )
    state = WorkflowState(query="How are Kroger earnings?")
    pools = {
        "observations": PoolState(PoolConfig(name="observations")),
        "sources": PoolState(PoolConfig(name="sources", dedup_content_hash=False)),
    }
    pools["observations"].add("Kroger reported Q3 results.")
    pools["sources"].add({"url": "https://example.com/kroger", "title": "Kroger"})

    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="Plain report", usage={"total_tokens": 5})
    )

    with patch(
        "databricks_deep_research.agents.builtins.synthesizer._build_reclaim_pipeline",
        side_effect=AssertionError("citation pipeline should not run"),
    ):
        output = await execute_agent("synth", config, state, llm, tools=[], pools=pools)

    assert output.content == "Plain report"
    assert not any(isinstance(event, ClaimVerifiedEvent) for event in output.events)
    llm.complete.assert_awaited()
