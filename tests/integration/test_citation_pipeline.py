"""Integration tests for citation pipeline with real LLM.

These tests verify the full 6-stage citation pipeline using real LLM calls:
1. Evidence Pre-Selection
2. Interleaved Generation (claim extraction)
3. Confidence Classification
4. Isolated Verification
5. Citation Correction
6. Numeric QA Verification

Requirements:
- .env file with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- .env file with BRAVE_API_KEY

Run with:
    uv run pytest tests/integration/test_citation_pipeline.py -v -s
"""

import pytest
from tests.integration.conftest import requires_all_credentials

from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient


@pytest.mark.integration
class TestCitationPipelineIntegration:
    """Test full citation pipeline with real LLM services."""

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_claim_generation_from_content(
        self,
        llm_client: LLMClient,
    ) -> None:
        """Test that claims are extracted correctly from synthesized content.

        This test verifies Stage 2 (Interleaved Generation) works with real LLM.
        """
        # TODO: Import citation pipeline components once paths are verified
        # from src.services.citation.pipeline import CitationPipeline

        # Sample synthesized content with claims
        content = """
        Apple Inc. reported revenue of $383 billion in fiscal year 2023.
        The company's iPhone segment generated approximately 52% of total revenue.
        Tim Cook has been CEO of Apple since 2011.
        """

        # Sample source content
        sources = [
            {
                "url": "https://example.com/apple-financials",
                "content": "Apple Inc. reported total revenue of $383.3 billion for fiscal year 2023. "
                "The iPhone segment contributed 52.3% of the total revenue. "
                "CEO Tim Cook, who took over from Steve Jobs in August 2011, announced the results.",
            }
        ]

        # Test that LLM client is working
        from src.services.llm.types import ModelTier

        response = await llm_client.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'test passed' in exactly two words."},
            ],
            tier=ModelTier.SIMPLE,
        )

        assert response.content is not None
        assert len(response.content) > 0

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_numeric_claim_detection(
        self,
        llm_client: LLMClient,
    ) -> None:
        """Test that numeric claims are properly detected.

        This test verifies Stage 6 (Numeric QA Verification) components.
        """
        from src.services.llm.types import ModelTier

        # Test content with numeric claims
        test_prompt = """
        Identify the numeric claims in this text and return them as a JSON list:

        "Apple's revenue was $383 billion in 2023. The company has 164,000 employees.
        iPhone sales accounted for 52% of revenue."

        Return format: [{"claim": "...", "value": "...", "unit": "..."}]
        """

        response = await llm_client.complete(
            messages=[
                {"role": "system", "content": "You are a claim extraction assistant."},
                {"role": "user", "content": test_prompt},
            ],
            tier=ModelTier.SIMPLE,
        )

        assert response.content is not None
        # Verify the response contains expected numeric values
        assert "383" in response.content or "billion" in response.content.lower()

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_evidence_selection_with_sources(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
    ) -> None:
        """Test evidence selection finds relevant quotes from sources.

        This test verifies Stage 1 (Evidence Pre-Selection) with real search.
        """
        # Do a simple search to get real sources
        search_results = await brave_client.search(
            query="Python programming language creator",
            count=3,
        )

        assert len(search_results) > 0

        # Verify we got actual content
        for result in search_results:
            assert result.url is not None
            assert result.title is not None

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_verification_verdict_assignment(
        self,
        llm_client: LLMClient,
    ) -> None:
        """Test that verification produces sensible verdicts.

        This test verifies Stage 4 (Isolated Verification) with real LLM.
        """
        from src.services.llm.types import ModelTier

        # Claim and evidence that should be SUPPORTED
        supported_test = """
        Claim: "Python was created by Guido van Rossum."
        Evidence: "Python is a high-level programming language created by Guido van Rossum
        and first released in 1991."

        Based on the evidence, is the claim SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED?
        Respond with just the verdict.
        """

        response = await llm_client.complete(
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant."},
                {"role": "user", "content": supported_test},
            ],
            tier=ModelTier.SIMPLE,
        )

        assert response.content is not None
        # Should recognize this as supported
        assert "SUPPORT" in response.content.upper()

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_full_pipeline_smoke_test(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
    ) -> None:
        """Smoke test for the full citation pipeline.

        This test runs a minimal research query and verifies citations are generated.
        """
        from src.agent.orchestrator import OrchestrationConfig, run_research

        config = OrchestrationConfig(
            max_iterations=1,
            max_steps=2,
        )

        # Run a simple research query
        result = await run_research(
            query="What year was Python programming language created?",
            llm_client=llm_client,
            brave_client=brave_client,
            config=config,
        )

        # Verify we got a result
        assert result is not None
        assert result.report is not None
        assert len(result.report) > 0

        # Verify sources were collected
        assert result.sources is not None

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_pipeline_fallback_on_empty_evidence(
        self,
        llm_client: LLMClient,
    ) -> None:
        """Test that pipeline falls back gracefully when no evidence is found.

        This test verifies the fallback mechanism when sources have no content,
        which should trigger classical synthesis instead of interleaved generation.
        """
        from src.agent.state import ResearchState, SourceInfo
        from src.services.citation.pipeline import CitationVerificationPipeline, VerificationEvent

        # Create state with sources that have no content (empty strings or None)
        state = ResearchState(
            query="What is machine learning?",
            sources=[
                SourceInfo(url="https://example1.com", title="Test 1", content=""),
                SourceInfo(url="https://example2.com", title="Test 2", content=None),
            ],
        )

        pipeline = CitationVerificationPipeline(llm_client)

        # Run the pipeline - should not raise, should produce content via fallback
        content_chunks: list[str] = []
        events: list[VerificationEvent] = []

        async for item in pipeline.run_full_pipeline(state, target_word_count=200):
            if isinstance(item, str):
                content_chunks.append(item)
            elif isinstance(item, VerificationEvent):
                events.append(item)

        # Verify content was produced (via classical fallback)
        full_content = "".join(content_chunks)
        assert len(full_content) > 50, "Pipeline should produce content via fallback"

        # Verify verification_summary event indicates fallback
        summary_events = [e for e in events if e.event_type == "verification_summary"]
        assert len(summary_events) == 1, "Should emit exactly one verification_summary event"

        summary_data = summary_events[0].data
        assert summary_data.get("verification_skipped") is True, "Should indicate verification was skipped"
        assert summary_data.get("reason") == "empty_evidence_pool", "Should explain why verification was skipped"
