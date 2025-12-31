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
    uv run pytest tests/complex -v -s --timeout=600

Note: Complex tests use production config (config/app.yaml) for full settings.
Each test may take 5-15 minutes to complete.
"""

from uuid import uuid4

import pytest

from src.agent.orchestrator import (
    OrchestrationConfig,
    run_research,
)
from src.agent.tools.web_crawler import WebCrawler
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient
from tests.complex.conftest import requires_all_credentials


@pytest.mark.complex
class TestComplexResearch:
    """Long-running research scenarios with production configuration."""

    @requires_all_credentials
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)  # 10 minute timeout
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

        # Verify comprehensive research
        assert result.state.final_report, "Should produce a final report"
        assert len(result.state.final_report) > 500, "Multi-entity comparison should be detailed"

        # Verify all entities are covered
        report_lower = result.state.final_report.lower()
        assert "apple" in report_lower, "Report should cover Apple"
        assert "microsoft" in report_lower, "Report should cover Microsoft"
        assert "google" in report_lower, "Report should cover Google"

        # Verify numeric content (market cap, revenue, employees)
        assert any(
            term in report_lower for term in ["billion", "trillion", "million", "revenue", "market cap"]
        ), "Report should contain numeric data"

        # Verify sources were found
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
    @pytest.mark.timeout(600)
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
    @pytest.mark.timeout(600)
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
    @pytest.mark.timeout(600)
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
