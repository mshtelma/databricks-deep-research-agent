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


class TestVerificationCitationKeyThreading:
    """Live ``claim_verified`` events must carry the NUMERIC citation keys that
    match the rendered (numeric) report, so the UI can color markers before
    persistence. Regression guard for the "grey-until-reload" bug.
    """

    def test_numeric_citation_keys_maps_named_to_numeric(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import (
            _numeric_citation_keys,
        )

        key_to_numeric = {"Arxiv": "0", "Github": "1"}
        # named primary + list → numeric, order preserved
        assert _numeric_citation_keys(
            "Arxiv", ["Arxiv", "Github"], key_to_numeric
        ) == ["0", "1"]
        # single key, no list
        assert _numeric_citation_keys("Github", None, key_to_numeric) == ["1"]
        # unmapped key falls back to itself (mirrors the report rewrite)
        assert _numeric_citation_keys("Unknown", None, key_to_numeric) == ["Unknown"]
        # dedup after mapping
        assert _numeric_citation_keys(None, ["Arxiv", "Arxiv"], key_to_numeric) == ["0"]
        # empty input
        assert _numeric_citation_keys(None, None, key_to_numeric) == []

    def test_normalize_verification_records_emits_numeric_keys(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import (
            _normalize_verification_records,
        )

        records = _normalize_verification_records(
            [
                {
                    "claim_index": 0,
                    "verdict": "supported",
                    "citation_key": "Arxiv",
                    "citation_keys": ["Arxiv", "Github"],
                }
            ],
            {"Arxiv": "0", "Github": "1"},
        )
        assert records[0]["citation_key"] == "0"
        assert records[0]["citation_keys"] == ["0", "1"]
        assert records[0]["verdict"] == "supported"

    def test_extract_claim_verified_events_carries_numeric_keys(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import (
            _extract_claim_verified_events,
        )

        events = _extract_claim_verified_events(
            "synth",
            "2026-01-01T00:00:00Z",
            [
                {
                    "claim_index": 0,
                    "verdict": "supported",
                    "confidence": 0.9,
                    "citation_key": "0",
                    "citation_keys": ["0", "1"],
                }
            ],
        )
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ClaimVerifiedEvent)
        assert event.citation_key == "0"
        assert event.citation_keys == ["0", "1"]


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

    # 4.2: a deterministic ## Sources section is appended to the report body
    # (default-on for deep_research). The verified claim prose is unchanged; the
    # cited source is rendered from the evidence pool + numeric marker.
    assert output.content.startswith(
        "Kroger reported identical sales growth of 2.6% [0]."
    )
    assert "## Sources" in output.content
    assert (
        "[0] [Kroger Q3 2025 Results](enterprise://vector_search/earnings/0)"
        in output.content
    )
    assert state.get("claims")[0]["citation_keys"] == ["0"]
    assert state.get("claims")[0]["claim_role"] == "fact"
    assert state.get("verification_summary")["total_claims"] == 1
    assert state.get("analysis_summary")["total_claims"] == 0
    assert any(isinstance(event, ClaimGeneratedEvent) for event in output.events)
    assert any(isinstance(event, ClaimVerifiedEvent) for event in output.events)
    assert any(isinstance(event, VerificationSummaryEvent) for event in output.events)
    llm.complete.assert_not_awaited()


class TestPlaceholderTitle:
    """Tests for _is_placeholder_title and its effect on _normalize_source."""

    def test_is_placeholder_title_rejects_untitled(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import _is_placeholder_title

        assert _is_placeholder_title("Untitled") is True
        assert _is_placeholder_title("unknown") is True
        assert _is_placeholder_title("N/A") is True
        assert _is_placeholder_title("") is True
        assert _is_placeholder_title("  ") is True
        assert _is_placeholder_title("AB") is True  # len < 3
        assert _is_placeholder_title("doc_42") is True
        assert _is_placeholder_title("row_0") is True
        assert _is_placeholder_title("Vector Search Result 7") is True

    def test_is_placeholder_title_allows_real_titles(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import _is_placeholder_title

        assert _is_placeholder_title("Kroger Q3 2025 Earnings Report") is False
        assert _is_placeholder_title("API Reference Guide") is False
        assert _is_placeholder_title("Product Documentation") is False

    def test_normalize_source_placeholder_title_not_used_as_snippet(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import _normalize_source

        source = {
            "url": "https://example.com",
            "title": "Untitled",
            "content": None,
            "snippet": "",
        }
        result = _normalize_source(source)

        assert result is not None
        # "Untitled" should NOT become the snippet
        assert result["snippet"] == ""

    def test_normalize_source_real_title_used_as_snippet(self) -> None:
        from databricks_deep_research.agents.builtins.synthesizer import _normalize_source

        source = {
            "url": "https://example.com",
            "title": "Kroger Q3 Earnings Report",
            "content": None,
            "snippet": "",
        }
        result = _normalize_source(source)

        assert result is not None
        # Real title should be used as snippet fallback
        assert result["snippet"] == "Kroger Q3 Earnings Report"


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
