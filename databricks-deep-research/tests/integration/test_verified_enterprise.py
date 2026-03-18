"""Integration test: enterprise Vector Search with reclaim-mode citation verification.

Combines real enterprise VS retrieval with the reclaim synthesizer that emits
verification events (ClaimGeneratedEvent, ClaimVerifiedEvent, VerificationSummaryEvent).
This covers the end-to-end path: real enterprise data -> pool accumulation ->
reclaim synthesizer -> citation markers.

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- Access to the VS index configured via FRAMEWORK_TEST_VS_INDEX
  (default: main.dbdemos_ai_agent.earnings_vs_index)

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration/test_verified_enterprise.py -v -s --timeout=600
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from databricks_deep_research.events.types import (
    CitationCorrectedEvent,
    ClaimGeneratedEvent,
    ClaimVerifiedEvent,
    NumericClaimDetectedEvent,
    PlanAndExecuteExitEvent,
    SynthesisStartedEvent,
    VerificationSummaryEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from tests.helpers import event_summary, print_citation_details
from tests.integration.conftest import RealVectorSearchTool, requires_databricks
from tests.integration.enterprise_helpers import (
    EnterpriseDatasetProfile,
    assert_enterprise_baseline,
    assert_reclaim_enterprise_output,
    build_enterprise_registry,
    print_enterprise_case_diagnostics,
    run_enterprise_case,
)

MIN_SUPPORTED_RATE = 0.20
MIN_SUPPORTED_OR_PARTIAL_RATE = 0.70
MAX_CONTRADICTED_RATE = 0.10
MIN_DIVERSE_SOURCES_IF_EARLY_EXIT = 4
MIN_DIVERSE_CITATIONS_IF_EARLY_EXIT = 3

_QUARTER_RE = re.compile(r"\bq[1-4]\b", re.IGNORECASE)
_FINANCE_VALUE_RE = re.compile(
    r"[\$€£¥]\s*\(?\d"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)\b"
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|trillion)\b",
    re.IGNORECASE,
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _looks_like_incidental_period_marker(raw_value: str, claim_text: str) -> bool:
    """Detect quarter/year tokens that were misparsed as finance values."""
    normalized = raw_value.strip().strip("()").replace(",", "")
    if normalized not in {"1", "2", "3", "4", "2024", "2025", "2026"}:
        return False
    if not _QUARTER_RE.search(claim_text):
        return False
    return bool(_FINANCE_VALUE_RE.search(claim_text))


def _looks_like_truncated_range_value(raw_value: str, claim_text: str) -> bool:
    """Detect malformed currency values created from range text like ``$4.75 to $4.80``."""
    if not re.search(r"[\$€£¥]\d+(?:\.\d+)?\s+[tTbBmMkK]$", raw_value.strip()):
        return False
    return bool(
        re.search(
            r"[\$€£¥]\s*\d+(?:\.\d+)?\s+to\s+[\$€£¥]\s*\d+(?:\.\d+)?",
            claim_text,
            re.IGNORECASE,
        )
    )


@pytest.mark.integration
class TestVerifiedEnterpriseWorkflow:
    """Enterprise VS retrieval + reclaim-mode synthesis with citation verification."""

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_verified_enterprise_research(
        self,
        llm_client: FrameworkLLMClient,
        real_vector_search_tool: RealVectorSearchTool,
        examples_dir: Path,
    ) -> None:
        """Real VS data flows through reclaim synthesizer producing cited report."""
        profile = EnterpriseDatasetProfile(
            label="Reclaim earnings VS",
            mode="reclaim",
            query="how are the Kroger earnings?",
            expected_tool_name="vector_search",
            required_keywords=("kroger", "earnings", "revenue", "financial", "profit"),
            required_term_groups=(
                ("kroger",),
                ("earnings", "revenue", "sales"),
                ("profit", "margin", "guidance", "financial"),
            ),
            min_report_length=200,
        )
        result = await run_enterprise_case(
            workflow_path=examples_dir / "verified_enterprise_research.yaml",
            llm_client=llm_client,
            profile=profile,
            enterprise_tools=[real_vector_search_tool],
            tool_registry=build_enterprise_registry(real_vector_search_tool),
        )
        state = result.state
        events = result.events
        elapsed = result.elapsed

        # ---------------------------------------------------------------
        # Hard assertions
        # ---------------------------------------------------------------
        assert_enterprise_baseline(result)
        assert_reclaim_enterprise_output(result)

        tool_calls = result.tool_calls
        vs_calls = result.matching_tool_calls
        sources_pool = result.sources_pool
        assert sources_pool is not None
        enterprise_urls = result.enterprise_urls
        report_str = result.report_str
        synthesis_events = [
            event for event in events if isinstance(event, SynthesisStartedEvent)
        ]

        assert "[" in report_str and "]" in report_str, (
            "Reclaim-mode report should contain [N] citation markers. "
            f"Report preview: {report_str[:400]}"
        )

        claim_generated = [
            e for e in events if isinstance(e, ClaimGeneratedEvent)
        ]
        claim_verified = [
            e for e in events if isinstance(e, ClaimVerifiedEvent)
        ]
        citation_corrected = [
            e for e in events if isinstance(e, CitationCorrectedEvent)
        ]
        numeric_claims = [
            e for e in events if isinstance(e, NumericClaimDetectedEvent)
        ]
        verification_summaries = [
            e for e in events if isinstance(e, VerificationSummaryEvent)
        ]
        plan_exit_events = [
            e for e in events if isinstance(e, PlanAndExecuteExitEvent)
        ]

        assert len(claim_generated) >= 1, (
            "Reclaim-mode framework execution should emit ClaimGeneratedEvent. "
            f"Event summary: {event_summary(events)}"
        )
        assert len(verification_summaries) >= 1, (
            "Reclaim-mode framework execution should emit VerificationSummaryEvent. "
            f"Event summary: {event_summary(events)}"
        )
        assert len(plan_exit_events) >= 1, (
            "Enterprise workflow should emit PlanAndExecuteExitEvent for research coverage checks"
        )

        verification_summary = state.get("verification_summary")
        assert isinstance(verification_summary, dict) and verification_summary, (
            "Synthesizer should persist structured verification_summary in state"
        )
        verification_details = state.get("verification_details")
        assert isinstance(verification_details, dict) and verification_details, (
            "Synthesizer should persist structured verification_details payload in state"
        )
        analysis_summary = state.get("analysis_summary")
        assert isinstance(analysis_summary, dict), (
            "Synthesizer should persist structured analysis_summary in state"
        )

        claims_payload = verification_details.get("claims", [])
        corrections_payload = verification_details.get("corrections", [])
        numeric_payload = verification_details.get("numeric_claims", [])
        verifications_payload = verification_details.get("verifications", [])
        assert isinstance(claims_payload, list)
        assert isinstance(corrections_payload, list)
        assert isinstance(numeric_payload, list)
        assert isinstance(verifications_payload, list)
        assert verification_summary.get("analysis_summary") == analysis_summary
        assert all("claim_role" in claim for claim in claims_payload), (
            "Every persisted claim should carry a claim_role for fact/analysis/free lanes"
        )
        assert all(
            "<analysis>" not in str(claim.get("claim_text", ""))
            and "<free>" not in str(claim.get("claim_text", ""))
            for claim in claims_payload
        ), "Persisted claim texts should be stripped of reclaim-only block tags"

        # ---------------------------------------------------------------
        # Quality gates
        # ---------------------------------------------------------------
        total_claims = int(verification_summary.get("total_claims", 0))
        supported_count = int(
            verification_summary.get(
                "supported_count",
                verification_summary.get("verified_claims", 0),
            )
        )
        partial_count = int(verification_summary.get("partial_count", 0))
        contradicted_count = int(verification_summary.get("contradicted_count", 0))
        abstained_count = int(verification_summary.get("abstained_count", 0))
        denominator = total_claims - abstained_count if total_claims > abstained_count else total_claims
        supported_rate = float(
            verification_summary.get(
                "supported_rate",
                _ratio(supported_count, denominator),
            )
        )
        supported_or_partial_rate = _ratio(
            supported_count + partial_count,
            denominator,
        )
        contradicted_rate = _ratio(contradicted_count, denominator)

        assert supported_or_partial_rate >= MIN_SUPPORTED_OR_PARTIAL_RATE, (
            "Verification coverage is too weak for enterprise runs. "
            f"supported+partial={supported_count + partial_count}/{denominator} "
            f"({supported_or_partial_rate:.1%})"
        )
        assert contradicted_rate <= MAX_CONTRADICTED_RATE, (
            "Contradicted-claim rate is too high for enterprise runs. "
            f"contradicted={contradicted_count}/{denominator} "
            f"({contradicted_rate:.1%})"
        )
        assert supported_rate >= MIN_SUPPORTED_RATE, (
            "Supported-claim rate is too low for this integration path. "
            f"supported={supported_count}/{denominator} ({supported_rate:.1%}) "
            f"summary_warning={verification_summary.get('warning')}"
        )

        analysis_contradicted = int(analysis_summary.get("contradicted_count", 0))
        analysis_unsupported = int(analysis_summary.get("unsupported_count", 0))
        assert analysis_contradicted == 0, (
            "Final output should not retain contradicted analysis sentences. "
            f"analysis_summary={analysis_summary}"
        )
        assert analysis_unsupported == 0, (
            "Unsupported analysis should be hedged or removed before final output. "
            f"analysis_summary={analysis_summary}"
        )

        if supported_rate < 0.30:
            print(
                f"WARNING: supported-claim rate is still low at {supported_rate:.1%}"
            )

        expected_warning = (
            supported_rate < MIN_SUPPORTED_RATE
            or float(verification_summary.get("unsupported_rate", 0.0)) > 0.30
            or contradicted_rate > MAX_CONTRADICTED_RATE
        )
        assert bool(verification_summary.get("warning", False)) == expected_warning, (
            "verification_summary.warning should reflect low-supported/high-unsupported/"
            "high-contradiction runs"
        )

        replace_corrections = [
            correction
            for correction in corrections_payload
            if correction.get("action") == "replace"
        ]
        for correction in replace_corrections:
            corrected_key = str(correction.get("corrected_key", "") or "")
            original_key = str(correction.get("original_key", "") or "")
            assert corrected_key, (
                "Citation replacements must record a non-empty corrected_key. "
                f"Bad correction: {correction}"
            )
            if original_key:
                assert corrected_key != original_key, (
                    "Citation replacements must change the citation key when the "
                    "original key is known. "
                    f"Bad correction: {correction}"
                )
        assert not any(
            "vector_search-" in str(correction.get("original_key", "")).lower()
            or "vector_search-" in str(correction.get("corrected_key", "")).lower()
            for correction in corrections_payload
        ), (
            "User-facing correction payloads should never leak evidence-pool keys like "
            "'Vector_search-38'. "
            f"Corrections: {corrections_payload}"
        )

        claims_by_index = dict(enumerate(claims_payload))
        incidental_numeric = []
        malformed_numeric = []
        for numeric_claim in numeric_payload:
            claim_index = int(numeric_claim.get("claim_index", -1))
            claim_text = str(
                claims_by_index.get(claim_index, {}).get("claim_text", "")
            )
            numeric_value = str(numeric_claim.get("numeric_value", "") or "")
            if _looks_like_incidental_period_marker(numeric_value, claim_text):
                incidental_numeric.append(
                    {
                        "claim_index": claim_index,
                        "numeric_value": numeric_value,
                        "claim_text": claim_text,
                    }
                )
            if _looks_like_truncated_range_value(numeric_value, claim_text):
                malformed_numeric.append(
                    {
                        "claim_index": claim_index,
                        "numeric_value": numeric_value,
                        "claim_text": claim_text,
                    }
                )
        assert not incidental_numeric, (
            "Numeric extraction should not emit quarter/year markers when a more "
            "salient finance value exists. "
            f"Bad numeric claims: {incidental_numeric}"
        )
        assert not malformed_numeric, (
            "Numeric extraction should not truncate guidance ranges into malformed "
            "values like '$4.75 t'. "
            f"Bad numeric claims: {malformed_numeric}"
        )

        unique_citation_keys = {
            str(citation_key)
            for claim in claims_payload
            for citation_key in claim.get("citation_keys", []) or []
            if citation_key not in (None, "")
        }
        assert not any(
            str(citation_key).lower().startswith("vector_search-")
            for citation_key in unique_citation_keys
        ), (
            "Claim citation keys should be normalized to source-pool indices, not raw "
            "evidence keys. "
            f"claim_citations={sorted(unique_citation_keys)}"
        )

        high_routed_fact_verifications = [
            verification
            for verification in verifications_payload
            if verification.get("claim_role") == "fact"
            and verification.get("routing_confidence_level") == "high"
        ]
        assert high_routed_fact_verifications, (
            "At least one obvious fact claim should route high and exercise the quick path. "
            f"verifications={verifications_payload[:10]}"
        )
        last_plan_exit = plan_exit_events[-1]
        if last_plan_exit.total_items_processed < 2:
            assert len(set(enterprise_urls)) >= MIN_DIVERSE_SOURCES_IF_EARLY_EXIT, (
                "If research exits before two items, source diversity must already be high. "
                f"unique_enterprise_sources={len(set(enterprise_urls))}"
            )
            assert len(unique_citation_keys) >= MIN_DIVERSE_CITATIONS_IF_EARLY_EXIT, (
                "If research exits before two items, the synthesized report must still "
                "cite a diverse evidence set. "
                f"unique_citations={len(unique_citation_keys)}"
            )

        # ---------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------
        summary = event_summary(events)
        observations_pool = result.observations_pool
        obs_count = observations_pool.count() if observations_pool else 0

        print(f"\n{'=' * 60}")
        print(f"Verified Enterprise Research completed in {elapsed:.1f}s")
        print(f"{'=' * 60}")
        print(f"Total events: {len(events)}")
        print(f"Event summary: {summary}")

        print("\n--- TOOL CALLS ---")
        print(f"Total tool calls: {len(tool_calls)}")
        print(f"Vector search calls: {len(vs_calls)}")
        tool_dist: dict[str, int] = {}
        for tc in tool_calls:
            tool_dist[tc.tool_name] = tool_dist.get(tc.tool_name, 0) + 1
        print(f"Tool distribution: {tool_dist}")

        print("\n--- POOLS ---")
        print(f"Sources pool: {sources_pool.count()} items")
        print(f"Observations pool: {obs_count} items")
        print(f"Enterprise URLs: {len(enterprise_urls)}")

        print("\n--- VERIFICATION EVENTS ---")
        print(f"ClaimGeneratedEvent:      {len(claim_generated)}")
        print(f"ClaimVerifiedEvent:       {len(claim_verified)}")
        print(f"CitationCorrectedEvent:   {len(citation_corrected)}")
        print(f"NumericClaimDetectedEvent: {len(numeric_claims)}")
        print(f"VerificationSummaryEvent: {len(verification_summaries)}")
        print(f"SynthesisStartedEvent:    {len(synthesis_events)}")

        if verification_summaries:
            vs_event = verification_summaries[0]
            print("\n--- VERIFICATION SUMMARY ---")
            print(f"Total claims:       {vs_event.total_claims}")
            print(f"Verified claims:    {vs_event.verified_claims}")
            print(f"Corrected citations: {vs_event.corrected_citations}")
            print(f"Removed claims:     {vs_event.removed_claims}")
            print(f"Softened claims:    {vs_event.softened_claims}")
            print(f"Overall confidence: {vs_event.overall_confidence:.2f}")
            print(f"Supported count:    {supported_count}")
            print(f"Partial count:      {partial_count}")
            print(f"Supported rate:     {supported_rate:.1%}")
            print(f"Supported+partial:  {supported_or_partial_rate:.1%}")
            print(f"Contradicted rate:  {contradicted_rate:.1%}")

        print(f"\n--- REPORT PREVIEW ({len(report_str)} chars) ---")
        print(report_str[:1000])

        print_citation_details(state, events)
        print_enterprise_case_diagnostics(result)
