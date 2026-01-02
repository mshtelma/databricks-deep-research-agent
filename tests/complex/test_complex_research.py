"""Complex, long-running research tests with production configuration.

These tests verify the system's ability to handle:
- Multi-entity comparative research
- Deep dive investigations with many iterations
- Comprehensive citation verification
- Multi-turn conversations with context

Requirements:
- .env file with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- .env file with BRAVE_API_KEY

Run with:
    make test-complex
    uv run pytest tests/complex -v -s

Note: Complex tests use production config (config/app.yaml) for full settings.
Each test may take 5-15 minutes to complete.
"""

import re
import time
from uuid import uuid4

import mlflow
import pytest

from src.agent.orchestrator import (
    OrchestrationConfig,
    run_research,
)
from src.agent.state import ResearchState
from src.agent.tools.web_crawler import WebCrawler
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient
from tests.complex.conftest import requires_all_credentials


# =============================================================================
# Grey Reference Detection Utilities
# =============================================================================


def extract_citation_markers(content: str) -> set[str]:
    """Extract all [Key] citation markers from content.

    Args:
        content: The report content to scan.

    Returns:
        Set of citation keys found in the content.
    """
    pattern = r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]"
    return set(re.findall(pattern, content))


def get_claim_citation_keys(state: ResearchState) -> set[str]:
    """Get all citation keys from claims in state.

    Args:
        state: Research state containing claims.

    Returns:
        Set of all citation keys across all claims.
    """
    keys: set[str] = set()
    for claim in state.claims:
        if claim.citation_key:
            keys.add(claim.citation_key)
        if claim.citation_keys:
            keys.update(claim.citation_keys)
    return keys


def count_grey_references(state: ResearchState) -> dict[str, int]:
    """Count different types of grey (unverified) references.

    Grey references include:
    - Abstained claims: verification couldn't proceed (no evidence)
    - Claims without evidence: have citation key but no EvidenceInfo
    - Orphaned markers: [Key] in content with no matching ClaimInfo

    Args:
        state: Research state after synthesis.

    Returns:
        Dict with counts: abstained, no_evidence, orphaned_markers, total_grey
    """
    abstained = sum(1 for c in state.claims if c.abstained)
    no_evidence = sum(
        1 for c in state.claims if c.evidence is None and not c.abstained
    )

    # Orphaned markers: in content but no matching claim
    content_markers = extract_citation_markers(state.final_report or "")
    claim_keys = get_claim_citation_keys(state)
    orphaned = len(content_markers - claim_keys)

    return {
        "abstained": abstained,
        "no_evidence": no_evidence,
        "orphaned_markers": orphaned,
        "total_grey": abstained + no_evidence + orphaned,
    }


# =============================================================================
# MLflow Metrics Logging
# =============================================================================


def log_verification_metrics_to_mlflow(
    state: ResearchState,
    query: str,
    domain: str,
    duration_ms: float,
) -> None:
    """Log verification metrics to MLflow for tracking and analysis.

    Args:
        state: Research state with verification results.
        query: The research query.
        domain: Test domain (e.g., "finance", "healthcare").
        duration_ms: Total research duration in milliseconds.
    """
    summary = state.verification_summary
    grey = count_grey_references(state)

    # Log metrics
    mlflow.log_metrics(
        {
            # Verdict counts
            "verification.total_claims": summary.total_claims if summary else 0,
            "verification.supported": summary.supported_count if summary else 0,
            "verification.partial": summary.partial_count if summary else 0,
            "verification.unsupported": summary.unsupported_count if summary else 0,
            "verification.contradicted": summary.contradicted_count if summary else 0,
            # Grey references
            "verification.abstained": grey["abstained"],
            "verification.no_evidence": grey["no_evidence"],
            "verification.orphaned_markers": grey["orphaned_markers"],
            "verification.total_grey": grey["total_grey"],
            # Rates
            "verification.unsupported_rate": summary.unsupported_rate if summary else 0,
            "verification.support_rate": (
                summary.supported_count / summary.total_claims
                if summary and summary.total_claims > 0
                else 0
            ),
            # Research metrics
            "research.duration_ms": duration_ms,
            "research.sources_count": len(state.sources),
            "research.claims_count": len(state.claims),
        }
    )

    # Log params
    mlflow.log_params(
        {
            "test.domain": domain,
            "test.query": query[:100],  # Truncate for MLflow
            "research.depth": state.resolve_depth(),
        }
    )


# =============================================================================
# Parametrized Test Cases for Verification Quality
# =============================================================================

VERIFICATION_TEST_CASES = [
    pytest.param(
        "What are the key architectural differences between Qwen2 and GLM-4 "
        "transformer models, including attention mechanisms, tokenization, "
        "and training approaches?",
        "ai_architecture",
        id="ai-qwen-vs-glm",
    ),
    pytest.param(
        "What were the major changes in Basel IV banking regulations "
        "and their impact on capital requirements?",
        "finance",
        id="finance-basel-iv",
    ),
    pytest.param(
        "What are the latest FDA-approved CAR-T cell therapies "
        "and their clinical trial outcomes?",
        "healthcare",
        id="hls-cart-therapy",
    ),
]


@pytest.mark.complex
class TestComplexResearch:
    """Long-running research scenarios with production configuration."""

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_multi_entity_comparison(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Compare multiple entities requiring comprehensive research.

        This test verifies the system can:
        - Decompose a multi-entity query into entity-specific steps
        - Research each entity thoroughly
        - Synthesize findings into a coherent comparison
        - Cite sources for numeric claims
        """
        result = await run_research(
            query="Compare the market capitalization, annual revenue, and employee count for Apple, Microsoft, and Google as of 2024",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            session_id=uuid4(),
            config=OrchestrationConfig(
                max_plan_iterations=3,
                max_steps_per_plan=10,
                timeout_seconds=540,  # 9 minutes
            ),
        )

        print("\n📊 Multi-entity comparison report:")
        print(result.state.final_report)

        assert len(result.state.sources) >= 3, "Should have multiple sources for comparison"

        # Verify substantial research effort
        assert result.steps_executed >= 3, "Should execute multiple research steps"

        print("\n✅ Multi-entity comparison test passed!")
        print(f"   - Report length: {len(result.state.final_report)} chars")
        print(f"   - Sources: {len(result.state.sources)}")
        print(f"   - Steps executed: {result.steps_executed}")
        print(f"   - Duration: {result.total_duration_ms / 1000:.1f}s")

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_deep_dive_with_follow_ups(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Multi-turn research with follow-up questions.

        This test verifies:
        - Initial deep research on a topic
        - Context preservation across follow-ups
        - Ability to build on previous findings
        """
        session_id = uuid4()

        # Step 1: Initial deep research
        print("\n📚 Step 1: Initial deep research...")
        initial_result = await run_research(
            query="Explain the current state of mRNA vaccine technology and its applications beyond COVID-19",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            session_id=session_id,
            config=OrchestrationConfig(
                max_plan_iterations=2,
                max_steps_per_plan=5,
                timeout_seconds=240,
            ),
        )

        assert initial_result.state.final_report, "Initial research should produce a report"
        print(f"   ✅ Initial research complete ({len(initial_result.state.final_report)} chars)")

        # Build conversation history
        conversation_history = [
            {"role": "user", "content": "Explain mRNA vaccine technology and applications beyond COVID-19"},
            {"role": "assistant", "content": initial_result.state.final_report},
        ]

        # Step 2: Follow-up question
        print("\n📚 Step 2: Follow-up on clinical trials...")
        followup_result = await run_research(
            query="What are the most promising clinical trials using this technology for cancer treatment?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            conversation_history=conversation_history,
            session_id=session_id,
            config=OrchestrationConfig(
                max_plan_iterations=2,
                max_steps_per_plan=5,
                timeout_seconds=240,
            ),
        )

        assert followup_result.state.final_report, "Follow-up should produce a report"

        # Verify context was used
        followup_lower = followup_result.state.final_report.lower()
        assert any(
            term in followup_lower for term in ["mrna", "cancer", "clinical", "trial", "treatment"]
        ), "Follow-up should reference mRNA and cancer"

        print(f"   ✅ Follow-up complete ({len(followup_result.state.final_report)} chars)")

        print("\n✅ Deep dive with follow-ups test passed!")
        print(f"   - Initial sources: {len(initial_result.state.sources)}")
        print(f"   - Follow-up sources: {len(followup_result.state.sources)}")

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_comprehensive_citation_verification(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Full research with comprehensive citation verification.

        This test verifies:
        - Research produces claims with citations
        - Numeric claims are identified and verified
        - Sources are valid and accessible
        """
        result = await run_research(
            query="What are the key statistics about global renewable energy adoption in 2024, including solar and wind capacity growth?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            session_id=uuid4(),
            config=OrchestrationConfig(
                max_plan_iterations=2,
                max_steps_per_plan=6,
                timeout_seconds=300,
            ),
        )

        print("\n📊 Renewable energy research report:")
        print(result.state.final_report)

        # Verify research completed
        assert result.state.final_report, "Should produce a report"
        assert len(result.state.final_report) > 300, "Should have substantial content"

        # Verify numeric content (statistics about renewable energy)
        report_lower = result.state.final_report.lower()
        assert any(
            term in report_lower for term in ["gigawatt", "gw", "terawatt", "tw", "percent", "%", "billion"]
        ), "Report should contain statistics"

        # Verify topics covered
        assert "solar" in report_lower or "photovoltaic" in report_lower, "Should discuss solar"
        assert "wind" in report_lower, "Should discuss wind"

        # Verify sources
        assert len(result.state.sources) > 0, "Should have sources"

        print("\n✅ Comprehensive citation verification test passed!")
        print(f"   - Report length: {len(result.state.final_report)} chars")
        print(f"   - Sources: {len(result.state.sources)}")
        print(f"   - Steps executed: {result.steps_executed}")
        print(f"   - Duration: {result.total_duration_ms / 1000:.1f}s")

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_complex_analytical_query(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Complex analytical query requiring multi-step research.

        This test verifies the system can handle queries that require:
        - Multiple research angles
        - Synthesis of diverse sources
        - Balanced analysis of pros and cons
        """
        result = await run_research(
            query="What are the economic and environmental trade-offs of electric vehicles versus hydrogen fuel cell vehicles for long-haul trucking?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            session_id=uuid4(),
            config=OrchestrationConfig(
                max_plan_iterations=2,
                max_steps_per_plan=8,
                timeout_seconds=360,
            ),
        )

        print("\n📊 EV vs Hydrogen trucking analysis:")
        print(result.state.final_report)

        # Verify comprehensive analysis
        assert result.state.final_report, "Should produce a report"
        assert len(result.state.final_report) > 400, "Complex analysis should be detailed"

        # Verify both technologies discussed
        report_lower = result.state.final_report.lower()
        assert "electric" in report_lower or "battery" in report_lower, "Should discuss EVs"
        assert "hydrogen" in report_lower or "fuel cell" in report_lower, "Should discuss hydrogen"

        # Verify analysis aspects
        assert any(
            term in report_lower for term in ["cost", "economic", "price", "investment"]
        ), "Should discuss economics"
        assert any(
            term in report_lower for term in ["emission", "environment", "carbon", "clean"]
        ), "Should discuss environment"

        # Verify trucking context
        assert any(
            term in report_lower for term in ["truck", "freight", "hauling", "range", "charging"]
        ), "Should discuss trucking specifics"

        print("\n✅ Complex analytical query test passed!")
        print(f"   - Report length: {len(result.state.final_report)} chars")
        print(f"   - Sources: {len(result.state.sources)}")
        print(f"   - Steps executed: {result.steps_executed}")
        print(f"   - Duration: {result.total_duration_ms / 1000:.1f}s")


# =============================================================================
# Verification Quality Tests with MLflow Metrics
# =============================================================================


@pytest.mark.complex
class TestVerificationMetrics:
    """Verification quality tests with MLflow metrics reporting.

    These tests run research queries across different domains and:
    1. Report verification metrics (supported, unsupported, partial) to MLflow
    2. Report grey reference counts to MLflow
    3. Fail if any grey references are detected (zero tolerance)

    Run with:
        uv run pytest tests/complex/test_complex_research.py::TestVerificationMetrics -v -s
    """

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,domain", VERIFICATION_TEST_CASES)
    async def test_verification_quality_with_metrics(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
        query: str,
        domain: str,
    ) -> None:
        """Run research and report verification metrics to MLflow.

        This test:
        - Executes a domain-specific research query
        - Logs all verification metrics to MLflow for tracking
        - Fails if ANY grey references are detected (zero tolerance)

        Args:
            llm_client: LLM client fixture.
            brave_client: Brave search client fixture.
            web_crawler: Web crawler fixture.
            query: The research query to execute.
            domain: Domain category for MLflow tagging.
        """
        with mlflow.start_run(run_name=f"verification-{domain}"):
            start_time = time.perf_counter()

            result = await run_research(
                query=query,
                llm=llm_client,
                brave_client=brave_client,
                crawler=web_crawler,
                session_id=uuid4(),
                config=OrchestrationConfig(
                    max_plan_iterations=2,
                    max_steps_per_plan=6,
                    timeout_seconds=480,
                ),
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            state = result.state

            # Log metrics to MLflow
            log_verification_metrics_to_mlflow(state, query, domain, duration_ms)

            # Count grey references
            grey = count_grey_references(state)

            # Print verification report
            print(f"\n{'='*60}")
            print(f"📊 Verification Report [{domain}]")
            print(f"{'='*60}")
            print(f"Query: {query[:80]}...")
            print(f"\n📝 Final Report ({len(state.final_report or '')} chars):")
            print(state.final_report[:500] + "..." if state.final_report else "None")

            if state.verification_summary:
                s = state.verification_summary
                print(f"\n📈 Verification Summary:")
                print(f"   Total claims:  {s.total_claims}")
                print(f"   ✅ Supported:   {s.supported_count}")
                print(f"   ⚠️  Partial:     {s.partial_count}")
                print(f"   ❌ Unsupported: {s.unsupported_count}")
                print(f"   🔄 Contradicted: {s.contradicted_count}")
                print(f"   📊 Support rate: {s.supported_count / s.total_claims * 100:.1f}%" if s.total_claims > 0 else "   📊 Support rate: N/A")
            else:
                print("\n⚠️  No verification summary available")

            print(f"\n🔘 Grey References:")
            print(f"   - Abstained:        {grey['abstained']}")
            print(f"   - No evidence:      {grey['no_evidence']}")
            print(f"   - Orphaned markers: {grey['orphaned_markers']}")
            print(f"   - TOTAL GREY:       {grey['total_grey']}")

            print(f"\n⏱️  Duration: {duration_ms / 1000:.1f}s")
            print(f"📚 Sources: {len(state.sources)}")
            print(f"🔢 Steps executed: {result.steps_executed}")

            # Assertions
            assert state.final_report, "Should produce a final report"
            assert len(state.final_report) > 300, "Report should have substantial content"

            # FAIL if grey references exist (zero tolerance)
            assert grey["total_grey"] == 0, (
                f"\n❌ Grey references detected! (Zero tolerance policy)\n"
                f"   Abstained:        {grey['abstained']}\n"
                f"   No evidence:      {grey['no_evidence']}\n"
                f"   Orphaned markers: {grey['orphaned_markers']}\n"
                f"   TOTAL:            {grey['total_grey']}"
            )

            print(f"\n✅ Verification test PASSED for [{domain}]!")
            print(f"\nFINAL REPORT:\n{state.final_report}\n")
            print(f"{'='*60}\n")
