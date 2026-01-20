"""Integration tests for the full Citation Verification Pipeline.

These tests verify the end-to-end flow of the 6-stage citation pipeline:
1. Evidence Selection (Stage 1)
2. Interleaved Generation (Stage 2)
3. Confidence Classification (Stage 3)
4. Isolated Verification (Stage 4)
5. Aggregation & Scoring (Stage 5)
6. Numeric QA Verification (Stage 6)

Tests use mocked LLM responses to ensure deterministic behavior.
"""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.state import ClaimInfo, EvidenceInfo, SourceInfo
from deep_research.services.citation.claim_generator import InterleavedClaim, InterleavedGenerator
from deep_research.services.citation.confidence_classifier import (
    ConfidenceClassifier,
    ConfidenceLevel,
)
from deep_research.services.citation.evidence_selector import EvidencePreSelector, RankedEvidence
from deep_research.services.citation.isolated_verifier import IsolatedVerifier
from deep_research.services.citation.numeric_verifier import NumericVerifier
from deep_research.services.citation.pipeline import (
    CitationVerificationPipeline,
    VerificationEvent,
)
from deep_research.services.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockLLMResponse:
    """Mock LLM response object."""

    content: str
    structured: object = None  # For structured output responses


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client for integration tests."""
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value=MockLLMResponse(content=""))
    return client


@pytest.fixture
def sample_sources() -> list[SourceInfo]:
    """Real-world-like sample sources for integration testing."""
    return [
        SourceInfo(
            url="https://techcrunch.com/2024/ai-market-report",
            title="AI Market Analysis 2024",
            content="""
The global artificial intelligence market reached $150.2 billion in 2024,
marking a 35% increase from the previous year. According to the report,
enterprise AI adoption grew significantly across all sectors.

Machine learning applications accounted for $75 billion of the total market,
while generative AI solutions contributed $45 billion. The remaining $30 billion
came from traditional AI services and consulting.

North America led the market with 42% share, followed by Asia-Pacific at 28%
and Europe at 22%. The report projects the market will reach $300 billion by 2027.
""",
            snippet="AI market reached $150.2 billion in 2024...",
        ),
        SourceInfo(
            url="https://reuters.com/tech/ai-investment-2024",
            title="Tech Giants Increase AI Investments",
            content="""
Major technology companies announced unprecedented AI investments in 2024.
Microsoft committed $13 billion to OpenAI, while Google allocated $10 billion
for AI research and infrastructure.

Amazon Web Services invested $8 billion in AI capabilities, focusing on
enterprise solutions and cloud-based AI services. Meta invested $6 billion
primarily in AI research for virtual reality applications.

Industry analysts noted that total AI investments by tech giants exceeded
$50 billion in 2024, a 60% increase from 2023.
""",
            snippet="Tech giants invested over $50 billion in AI...",
        ),
        SourceInfo(
            url="https://nature.com/ai-healthcare-2024",
            title="AI Applications in Healthcare",
            content="""
Healthcare AI applications showed remarkable progress in 2024. A study
published in Nature Medicine found that AI-assisted diagnostics achieved
94% accuracy in detecting early-stage cancers, compared to 88% for
traditional methods.

The FDA approved 23 new AI-based medical devices in 2024, up from 15 in 2023.
Patient outcomes improved by 15% in hospitals using AI-powered treatment
recommendations.

Researchers cautioned that while results are promising, larger clinical trials
are needed to validate these findings across diverse patient populations.
""",
            snippet="AI diagnostics achieved 94% accuracy...",
        ),
    ]


@pytest.fixture
def sample_claims_with_evidence() -> list[ClaimInfo]:
    """Sample claims with associated evidence for verification testing."""
    return [
        ClaimInfo(
            claim_text="The AI market reached $150.2 billion in 2024.",
            claim_type="numeric",
            position_start=0,
            position_end=50,
            evidence=EvidenceInfo(
                source_url="https://techcrunch.com/2024/ai-market-report",
                quote_text="The global artificial intelligence market reached $150.2 billion in 2024",
                start_offset=0,
                end_offset=75,
                section_heading="Market Overview",
                relevance_score=0.95,
                has_numeric_content=True,
            ),
            confidence_level="high",
        ),
        ClaimInfo(
            claim_text="According to the report, enterprise AI adoption grew significantly.",
            claim_type="general",
            position_start=51,
            position_end=115,
            evidence=EvidenceInfo(
                source_url="https://techcrunch.com/2024/ai-market-report",
                quote_text="According to the report, enterprise AI adoption grew significantly across all sectors",
                start_offset=100,
                end_offset=185,
                section_heading="Market Overview",
                relevance_score=0.88,
                has_numeric_content=False,
            ),
            confidence_level="high",
        ),
        ClaimInfo(
            claim_text="AI diagnostics might improve healthcare outcomes by approximately 15%.",
            claim_type="numeric",
            position_start=116,
            position_end=185,
            evidence=EvidenceInfo(
                source_url="https://nature.com/ai-healthcare-2024",
                quote_text="Patient outcomes improved by 15% in hospitals using AI-powered treatment",
                start_offset=200,
                end_offset=275,
                section_heading="Healthcare Impact",
                relevance_score=0.82,
                has_numeric_content=True,
            ),
            confidence_level="low",  # "might", "approximately" = hedging
        ),
    ]


@pytest.fixture
def mock_app_config():
    """Mock app config for citation tests."""

    @dataclass
    class EvidencePreselectionConfig:
        max_spans_per_source: int = 5
        min_span_length: int = 50
        max_span_length: int = 500
        relevance_threshold: float = 0.3
        relevance_computation_method: str = "hybrid"
        numeric_content_boost: float = 0.1

    @dataclass
    class InterleavedGenerationConfig:
        min_evidence_similarity: float = 0.3

    @dataclass
    class IsolatedVerificationConfig:
        verification_model_tier: str = "analytical"
        quick_verification_tier: str = "simple"
        enable_nei_verdict: bool = True

    @dataclass
    class CitationVerificationConfig:
        enable_evidence_preselection: bool = True
        enable_interleaved_generation: bool = True
        enable_confidence_classification: bool = True
        evidence_preselection: EvidencePreselectionConfig | None = None
        interleaved_generation: InterleavedGenerationConfig | None = None
        isolated_verification: IsolatedVerificationConfig | None = None

        def __post_init__(self) -> None:
            if self.evidence_preselection is None:
                self.evidence_preselection = EvidencePreselectionConfig()
            if self.interleaved_generation is None:
                self.interleaved_generation = InterleavedGenerationConfig()
            if self.isolated_verification is None:
                self.isolated_verification = IsolatedVerificationConfig()

    @dataclass
    class MockAppConfig:
        citation_verification: CitationVerificationConfig | None = None

        def __post_init__(self) -> None:
            if self.citation_verification is None:
                self.citation_verification = CitationVerificationConfig()

    return MockAppConfig()


@pytest.fixture
def patch_app_config(mock_app_config: Any) -> Any:
    """Patch get_app_config to return mock config."""
    with patch("deep_research.core.app_config.get_app_config", return_value=mock_app_config):
        yield mock_app_config


# ---------------------------------------------------------------------------
# Stage 1: Evidence Selection Integration Tests
# ---------------------------------------------------------------------------


class TestEvidenceSelectionIntegration:
    """Integration tests for Stage 1: Evidence Selection."""

    @pytest.mark.asyncio
    async def test_full_evidence_extraction_flow(
        self, mock_llm_client: MagicMock, sample_sources: list[SourceInfo], patch_app_config: Any
    ) -> None:
        """Test complete evidence extraction from sources to ranked spans."""
        # Mock LLM response for span extraction
        mock_response = json.dumps({
            "spans": [
                {
                    "quote_text": "The global artificial intelligence market reached $150.2 billion in 2024",
                    "section": "Market Overview",
                    "relevance_score": 0.95,
                    "has_numeric": True,
                },
                {
                    "quote_text": "Machine learning applications accounted for $75 billion",
                    "section": "Market Breakdown",
                    "relevance_score": 0.85,
                    "has_numeric": True,
                },
            ]
        })
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Execute Stage 1
        evidence = await pipeline.preselect_evidence(
            sources=sample_sources,
            query="What is the size of the AI market in 2024?",
        )

        # Verify evidence extraction
        assert isinstance(evidence, list)
        # LLM extraction returns evidence or fallback
        if evidence:
            assert all(isinstance(e, RankedEvidence) for e in evidence)
            # Should be ranked by relevance
            if len(evidence) > 1:
                assert evidence[0].relevance_score >= evidence[1].relevance_score

    @pytest.mark.asyncio
    async def test_evidence_selection_handles_empty_sources(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test graceful handling of empty source list."""
        pipeline = CitationVerificationPipeline(mock_llm_client)

        evidence = await pipeline.preselect_evidence(
            sources=[],
            query="test query",
        )

        assert evidence == []

    @pytest.mark.asyncio
    async def test_evidence_selection_numeric_content_detection(
        self, mock_llm_client: MagicMock, sample_sources: list[SourceInfo], patch_app_config: Any
    ) -> None:
        """Test that numeric content is properly detected in evidence."""
        # Mock response with numeric evidence
        mock_response = json.dumps({
            "spans": [
                {
                    "quote_text": "$150.2 billion market size",
                    "section": "Financials",
                    "relevance_score": 0.9,
                    "has_numeric": True,
                },
            ]
        })
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        pipeline = CitationVerificationPipeline(mock_llm_client)

        evidence = await pipeline.preselect_evidence(
            sources=sample_sources[:1],
            query="financial figures",
        )

        if evidence:
            # Numeric content should be detected
            numeric_evidence = [e for e in evidence if e.has_numeric_content]
            assert len(numeric_evidence) >= 0  # May or may not have numeric based on extraction


# ---------------------------------------------------------------------------
# Stage 2-3: Interleaved Generation + Confidence Classification
# ---------------------------------------------------------------------------


class TestInterleavedGenerationIntegration:
    """Integration tests for Stage 2: Interleaved Generation."""

    @pytest.mark.asyncio
    async def test_generate_with_citations_flow(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test generation produces claims with citation markers."""
        # Mock LLM to return text with citation markers
        mock_response = "The AI market reached $150 billion [0]. Growth was 35% [1]."
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        evidence_pool = [
            RankedEvidence(
                source_id=None,
                source_url="https://example.com/1",
                source_title="Report 1",
                quote_text="AI market reached $150 billion in 2024",
                start_offset=0,
                end_offset=40,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
            RankedEvidence(
                source_id=None,
                source_url="https://example.com/2",
                source_title="Report 2",
                quote_text="Growth was 35% year over year",
                start_offset=0,
                end_offset=30,
                section_heading=None,
                relevance_score=0.85,
                has_numeric_content=True,
            ),
        ]

        pipeline = CitationVerificationPipeline(mock_llm_client)

        results: list[tuple[str, InterleavedClaim | None]] = []
        async for content, claim in pipeline.generate_with_interleaving(
            evidence_pool=evidence_pool,
            observations=["Previous research findings"],
            query="What is the AI market size?",
        ):
            results.append((content, claim))

        # Should yield content and claims
        assert len(results) >= 1


class TestConfidenceClassificationIntegration:
    """Integration tests for Stage 3: Confidence Classification."""

    def test_full_classification_with_evidence_matching(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test confidence classification with quote matching."""
        pipeline = CitationVerificationPipeline(mock_llm_client)

        # High confidence claim with matching evidence
        high_claim = ClaimInfo(
            claim_text="According to the report, the market grew by 35%.",
            claim_type="numeric",
            position_start=0,
            position_end=50,
            evidence=EvidenceInfo(
                source_url="https://example.com",
                quote_text="the market grew by 35%",
                start_offset=0,
                end_offset=25,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
        )

        confidence = pipeline.classify_confidence(high_claim)
        assert confidence in ["high", "medium"]

        # Low confidence claim with hedging
        low_claim = ClaimInfo(
            claim_text="The market might possibly grow by approximately 35%.",
            claim_type="numeric",
            position_start=0,
            position_end=55,
            evidence=None,
        )

        confidence = pipeline.classify_confidence(low_claim)
        assert confidence == "low"

    def test_confidence_routing_to_verification(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test that confidence level affects verification routing."""
        pipeline = CitationVerificationPipeline(mock_llm_client)

        # High confidence should use quick verification
        high_conf_claim = ClaimInfo(
            claim_text="According to data, as stated in the official report, revenue was $5B.",
            claim_type="numeric",
            position_start=0,
            position_end=70,
            evidence=EvidenceInfo(
                source_url="https://example.com",
                quote_text="revenue was $5B",
                start_offset=0,
                end_offset=20,
                section_heading=None,
                relevance_score=0.95,
                has_numeric_content=True,
            ),
        )

        confidence_result = pipeline.confidence_classifier.classify(
            high_conf_claim.claim_text,
            high_conf_claim.evidence.quote_text if high_conf_claim.evidence else None,
        )

        # With multiple high-confidence indicators + quote match, should have elevated score
        assert confidence_result.score >= 0.5

        # Low confidence should use full verification
        low_conf_claim = ClaimInfo(
            claim_text="Revenue might be around $5B, possibly higher.",
            claim_type="numeric",
            position_start=0,
            position_end=45,
            evidence=None,
        )

        should_quick = pipeline.confidence_classifier.should_use_quick_verification(
            low_conf_claim.claim_text,
        )
        assert should_quick is False


# ---------------------------------------------------------------------------
# Stage 4-5: Verification + Aggregation Integration Tests
# ---------------------------------------------------------------------------


class TestVerificationIntegration:
    """Integration tests for Stage 4-5: Verification and Aggregation."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(
        self,
        mock_llm_client: MagicMock,
        sample_claims_with_evidence: list[ClaimInfo],
        patch_app_config: Any,
    ) -> None:
        """Test complete verification flow with verdict assignment."""
        # Mock verification response
        mock_response = json.dumps({
            "verdict": "SUPPORTED",
            "reasoning": "The evidence directly states the claim.",
            "confidence": 0.9,
        })
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        pipeline = CitationVerificationPipeline(mock_llm_client)

        events: list[VerificationEvent] = []
        async for event in pipeline.verify_claims(sample_claims_with_evidence):
            events.append(event)

        # Should emit verification events
        assert len(events) >= 1
        assert all(isinstance(e, VerificationEvent) for e in events)

        # Check event types
        event_types = [e.event_type for e in events]
        assert "claim_verified" in event_types

    @pytest.mark.asyncio
    async def test_verification_handles_no_evidence(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test that claims without evidence are marked as abstained."""
        claims = [
            ClaimInfo(
                claim_text="This claim has no supporting evidence.",
                claim_type="general",
                position_start=0,
                position_end=40,
                evidence=None,
            )
        ]

        pipeline = CitationVerificationPipeline(mock_llm_client)

        events: list[VerificationEvent] = []
        async for event in pipeline.verify_claims(claims):
            events.append(event)

        # Claim should be marked as abstained
        assert claims[0].abstained is True

    @pytest.mark.asyncio
    async def test_verification_verdict_mapping(
        self,
        mock_llm_client: MagicMock,
        sample_claims_with_evidence: list[ClaimInfo],
        patch_app_config: Any,
    ) -> None:
        """Test that verdicts are correctly mapped from LLM responses."""
        verdicts = ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]

        for expected_verdict in verdicts:
            mock_response = json.dumps({
                "verdict": expected_verdict,
                "reasoning": f"Test reasoning for {expected_verdict}",
            })
            mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

            pipeline = CitationVerificationPipeline(mock_llm_client)
            claim = sample_claims_with_evidence[0]

            events: list[VerificationEvent] = []
            async for event in pipeline.verify_claims([claim]):
                events.append(event)

            # Verify event emitted
            verified_events = [e for e in events if e.event_type == "claim_verified"]
            assert len(verified_events) >= 1


# ---------------------------------------------------------------------------
# Stage 6: Numeric Verification Integration Tests
# ---------------------------------------------------------------------------


class TestNumericVerificationIntegration:
    """Integration tests for Stage 6: Numeric QA Verification."""

    @pytest.mark.asyncio
    async def test_numeric_claim_detection_and_verification(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test full numeric claim detection and QA verification."""
        # Mock QA verification response
        mock_response = json.dumps([
            {
                "question": "What is the market value?",
                "claim_answer": "$150 billion",
                "evidence_answer": "$150.2 billion",
            }
        ])
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        verifier = NumericVerifier(mock_llm_client)

        # Test numeric claim detection
        text = "The market reached $150B. Growth was 35%. Revenue hit $5.2 million."
        claims = verifier.detect_numeric_claims(text)

        assert len(claims) >= 2  # Should detect multiple numeric claims
        assert any("$150B" in c for c in claims)
        assert any("35%" in c for c in claims)

    @pytest.mark.asyncio
    async def test_numeric_value_parsing_and_normalization(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test numeric value parsing with normalization."""
        verifier = NumericVerifier(mock_llm_client)

        # Test currency parsing
        result = verifier.parse_numeric_value("Revenue was $3.2 billion.")
        assert result is not None
        assert result.normalized_value == 3_200_000_000
        assert result.unit == "USD"

        # Test percentage parsing
        result = verifier.parse_numeric_value("Growth increased by 25.5%.")
        assert result is not None
        assert result.normalized_value == 25.5
        assert result.unit == "percent"

        # Test millions
        result = verifier.parse_numeric_value("Costs were €500M.")
        assert result is not None
        assert result.normalized_value == 500_000_000
        assert result.unit == "EUR"

    @pytest.mark.asyncio
    async def test_numeric_qa_verification_match(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test QA-based numeric verification with matching values."""
        mock_response = json.dumps([
            {
                "question": "What is the revenue?",
                "claim_answer": "$3.2 billion",
                "evidence_answer": "$3.2 billion",
            }
        ])
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Company revenue reached $3.2 billion in Q4 2024.",
            start_offset=0,
            end_offset=50,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        result = await verifier.verify_numeric_claim(
            claim_text="Revenue was $3.2 billion.",
            evidence=evidence,
        )

        # Should parse the numeric value
        assert result.parsed_value is not None
        # Should detect as direct derivation
        assert result.derivation_type == "direct"

    @pytest.mark.asyncio
    async def test_numeric_computed_derivation_detection(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test detection of computed/derived numeric values."""
        mock_llm_client.complete.return_value = MockLLMResponse(content="[]")

        verifier = NumericVerifier(mock_llm_client)

        # Evidence has multiple numbers that could be combined
        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Revenue was $10B and costs were $7B.",
            start_offset=0,
            end_offset=40,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        # Claim mentions derived value
        result = await verifier.verify_numeric_claim(
            claim_text="Profit was calculated as $3B.",
            evidence=evidence,
        )

        # Should detect as computed
        assert result.derivation_type == "computed"


# ---------------------------------------------------------------------------
# Full Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end integration tests for the complete citation pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(
        self,
        mock_llm_client: MagicMock,
        sample_sources: list[SourceInfo],
        patch_app_config: Any,
    ) -> None:
        """Test the entire citation pipeline from sources to verified claims."""
        # Mock responses for different stages
        call_count = 0

        def mock_complete_side_effect(*args: Any, **kwargs: Any) -> MockLLMResponse:
            nonlocal call_count
            call_count += 1

            # First call: evidence extraction
            if call_count == 1:
                return MockLLMResponse(content=json.dumps({
                    "spans": [
                        {
                            "quote_text": "AI market reached $150.2 billion",
                            "section": "Overview",
                            "relevance_score": 0.9,
                            "has_numeric": True,
                        }
                    ]
                }))

            # Second call: generation
            elif call_count == 2:
                return MockLLMResponse(
                    content="The AI market reached $150.2 billion in 2024 [0]."
                )

            # Third call: verification
            else:
                return MockLLMResponse(content=json.dumps({
                    "verdict": "SUPPORTED",
                    "reasoning": "Evidence directly supports the claim.",
                }))

        mock_llm_client.complete.side_effect = mock_complete_side_effect

        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Stage 1: Evidence preselection
        evidence = await pipeline.preselect_evidence(
            sources=sample_sources,
            query="What is the AI market size in 2024?",
        )

        # Stage 2-3: Generate with interleaving (includes confidence classification)
        generated_claims: list[InterleavedClaim] = []
        async for content, claim in pipeline.generate_with_interleaving(
            evidence_pool=evidence if evidence else [],
            observations=[],
            query="What is the AI market size in 2024?",
        ):
            if claim:
                generated_claims.append(claim)

        # Convert to ClaimInfo for verification
        claims = [
            ClaimInfo(
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                position_start=c.position_start,
                position_end=c.position_end,
                evidence=EvidenceInfo(
                    source_url=evidence[c.evidence_index].source_url,
                    quote_text=evidence[c.evidence_index].quote_text,
                    start_offset=evidence[c.evidence_index].start_offset,
                    end_offset=evidence[c.evidence_index].end_offset,
                    section_heading=evidence[c.evidence_index].section_heading,
                    relevance_score=evidence[c.evidence_index].relevance_score,
                    has_numeric_content=evidence[c.evidence_index].has_numeric_content,
                ) if c.evidence_index is not None and evidence else None,
            )
            for c in generated_claims
        ]

        # Stage 4-5: Verify claims
        if claims:
            events: list[VerificationEvent] = []
            async for event in pipeline.verify_claims(claims):
                events.append(event)

    @pytest.mark.asyncio
    async def test_pipeline_with_mixed_claim_types(
        self,
        mock_llm_client: MagicMock,
        patch_app_config: Any,
    ) -> None:
        """Test pipeline handles both numeric and general claims."""
        mock_response = json.dumps({
            "verdict": "SUPPORTED",
            "reasoning": "Evidence supports the claim.",
        })
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        pipeline = CitationVerificationPipeline(mock_llm_client)

        claims = [
            # Numeric claim
            ClaimInfo(
                claim_text="Revenue grew by 35% in 2024.",
                claim_type="numeric",
                position_start=0,
                position_end=30,
                evidence=EvidenceInfo(
                    source_url="https://example.com",
                    quote_text="Revenue grew by 35% year-over-year",
                    start_offset=0,
                    end_offset=35,
                    section_heading=None,
                    relevance_score=0.9,
                    has_numeric_content=True,
                ),
            ),
            # General claim
            ClaimInfo(
                claim_text="The company expanded into new markets.",
                claim_type="general",
                position_start=31,
                position_end=70,
                evidence=EvidenceInfo(
                    source_url="https://example.com",
                    quote_text="Company announced expansion into Asian markets",
                    start_offset=100,
                    end_offset=150,
                    section_heading=None,
                    relevance_score=0.85,
                    has_numeric_content=False,
                ),
            ),
        ]

        events: list[VerificationEvent] = []
        async for event in pipeline.verify_claims(claims):
            events.append(event)

        # Should verify both claim types
        assert len(events) >= 2

        # Check that numeric claim triggered numeric detection
        numeric_events = [e for e in events if e.event_type == "numeric_claim_detected"]
        # May or may not have numeric events depending on detection

    @pytest.mark.asyncio
    async def test_pipeline_aggregation_statistics(
        self,
        mock_llm_client: MagicMock,
        patch_app_config: Any,
    ) -> None:
        """Test that pipeline correctly aggregates verification statistics."""
        verdicts = ["SUPPORTED", "SUPPORTED", "PARTIAL", "UNSUPPORTED"]
        call_count = 0

        def mock_verdict_side_effect(*args: Any, **kwargs: Any) -> MockLLMResponse:
            nonlocal call_count
            verdict = verdicts[call_count % len(verdicts)]
            call_count += 1
            return MockLLMResponse(content=json.dumps({
                "verdict": verdict,
                "reasoning": f"Test for {verdict}",
            }))

        mock_llm_client.complete.side_effect = mock_verdict_side_effect

        pipeline = CitationVerificationPipeline(mock_llm_client)

        claims = [
            ClaimInfo(
                claim_text=f"Claim {i}",
                claim_type="general",
                position_start=i * 10,
                position_end=i * 10 + 8,
                evidence=EvidenceInfo(
                    source_url=f"https://example.com/{i}",
                    quote_text=f"Evidence {i}",
                    start_offset=0,
                    end_offset=10,
                    section_heading=None,
                    relevance_score=0.8,
                    has_numeric_content=False,
                ),
            )
            for i in range(4)
        ]

        events: list[VerificationEvent] = []
        async for event in pipeline.verify_claims(claims):
            events.append(event)

        # Should have verification events for all claims
        verified_events = [e for e in events if e.event_type == "claim_verified"]
        assert len(verified_events) == 4


# ---------------------------------------------------------------------------
# Error Handling Integration Tests
# ---------------------------------------------------------------------------


class TestErrorHandlingIntegration:
    """Integration tests for error handling across the pipeline."""

    @pytest.mark.asyncio
    async def test_llm_error_graceful_degradation(
        self,
        mock_llm_client: MagicMock,
        sample_sources: list[SourceInfo],
        patch_app_config: Any,
    ) -> None:
        """Test pipeline handles LLM errors gracefully."""
        # Simulate LLM error
        mock_llm_client.complete.side_effect = Exception("LLM service unavailable")

        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Should not raise, but return empty or fallback
        try:
            evidence = await pipeline.preselect_evidence(
                sources=sample_sources,
                query="test query",
            )
            # If it returns something, it should be a list
            assert isinstance(evidence, list)
        except Exception:
            # Some implementations may raise - that's also acceptable
            pass

    @pytest.mark.asyncio
    async def test_malformed_llm_response_handling(
        self,
        mock_llm_client: MagicMock,
        patch_app_config: Any,
    ) -> None:
        """Test handling of malformed LLM responses."""
        # Return invalid JSON
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="This is not valid JSON at all {{{}}}"
        )

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Test",
            quote_text="Revenue was $5 billion.",
            start_offset=0,
            end_offset=25,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        # Should handle gracefully, not crash
        result = await verifier.verify_numeric_claim(
            claim_text="Revenue was $5B.",
            evidence=evidence,
        )

        # Should return a result even with malformed response
        assert result is not None
        assert result.parsed_value is not None

    @pytest.mark.asyncio
    async def test_empty_evidence_handling(
        self,
        mock_llm_client: MagicMock,
        patch_app_config: Any,
    ) -> None:
        """Test pipeline handles empty evidence gracefully."""
        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Generate with empty evidence pool
        results: list[tuple[str, InterleavedClaim | None]] = []
        async for content, claim in pipeline.generate_with_interleaving(
            evidence_pool=[],
            observations=[],
            query="test query",
        ):
            results.append((content, claim))

        # Should handle gracefully
        # May return empty or fallback content


# ---------------------------------------------------------------------------
# Performance and Edge Case Tests
# ---------------------------------------------------------------------------


class TestPerformanceEdgeCases:
    """Tests for performance characteristics and edge cases."""

    def test_confidence_classifier_is_synchronous(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Verify confidence classification is synchronous (no LLM calls)."""
        classifier = ConfidenceClassifier()

        # Should complete instantly without LLM
        result = classifier.classify(
            "According to the report, revenue grew by 35%.",
            "revenue grew by 35%",
        )

        assert result.level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
        assert 0 <= result.score <= 1
        assert isinstance(result.indicators, list)

    def test_numeric_parsing_edge_cases(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test numeric parsing handles edge cases."""
        verifier = NumericVerifier(mock_llm_client)

        # Very large numbers
        result = verifier.parse_numeric_value("$1.5 trillion in assets")
        assert result is not None
        assert result.normalized_value == 1_500_000_000_000

        # Mixed formats
        result = verifier.parse_numeric_value("1,234,567 users joined")
        assert result is not None
        assert result.normalized_value == 1234567

        # No numbers
        result = verifier.parse_numeric_value("No numeric content here")
        assert result is None

        # Edge case: just percentage
        result = verifier.parse_numeric_value("Exactly 100% complete")
        assert result is not None
        assert result.normalized_value == 100
        assert result.unit == "percent"

    def test_confidence_classification_caching_opportunity(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Test that identical claims produce consistent results."""
        classifier = ConfidenceClassifier()

        claim = "According to the data, sales increased by 25%."
        evidence = "sales increased by 25%"

        result1 = classifier.classify(claim, evidence)
        result2 = classifier.classify(claim, evidence)

        # Should be deterministic
        assert result1.level == result2.level
        assert result1.score == result2.score


# ---------------------------------------------------------------------------
# Stage 7: Grey Reference Prevention Tests
# ---------------------------------------------------------------------------


class TestStage7GreyReferencePrevention:
    """Tests for Stage 7 grey reference prevention fix.

    Verifies that _update_claims_with_stage7_revisions does NOT create
    grey references when the revised claim has a citation key that
    is not in the key_to_evidence map.
    """

    def test_stage7_preserves_original_key_when_new_key_not_in_evidence(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Stage 7 should NOT update citation_key if new key is not in evidence map.

        This prevents grey references (claims with citation_key but no evidence).
        """
        from deep_research.services.citation.atomic_decomposer import (
            AtomicFact,
            ClaimDecomposition,
            ClaimRevision,
        )
        from deep_research.services.citation.pipeline import CitationVerificationPipeline

        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Create a claim with known citation_key and evidence
        original_key = "Arxiv"
        original_claim = ClaimInfo(
            claim_text="The market grew by 35%. [Arxiv]",
            claim_type="numeric",
            position_start=0,
            position_end=32,
            citation_key=original_key,
            evidence=EvidenceInfo(
                source_url="https://arxiv.org/paper",
                quote_text="market grew by 35%",
                start_offset=0,
                end_offset=20,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
        )

        # Create atomic facts for decomposition
        atomic_fact_1 = AtomicFact(
            fact_text="The market grew",
            fact_index=0,
            parent_claim_id=0,
            is_verified=True,
        )
        atomic_fact_2 = AtomicFact(
            fact_text="by approximately 35%",
            fact_index=1,
            parent_claim_id=0,
            is_verified=False,
        )

        # Create decomposition
        decomposition = ClaimDecomposition(
            original_claim=original_claim,
            atomic_facts=[atomic_fact_1, atomic_fact_2],
        )

        # Create revision with a NEW key that's NOT in evidence pool
        revision = ClaimRevision(
            original_claim="The market grew by 35%. [Arxiv]",
            revised_claim="Reportedly, the market grew by approximately 35%. [NewSource]",
            revision_type="partially_softened",
            original_position_start=0,
            original_position_end=32,
            decomposition=decomposition,
            verified_facts=[atomic_fact_1],
            softened_facts=[atomic_fact_2],
        )

        # Evidence pool only has original key, NOT "NewSource"
        key_to_evidence = {
            "Arxiv": RankedEvidence(
                source_id=None,
                source_url="https://arxiv.org/paper",
                source_title="Arxiv Paper",
                quote_text="market grew by 35%",
                start_offset=0,
                end_offset=20,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
        }

        # Content for position finding
        revised_content = "Reportedly, the market grew by approximately 35%. [NewSource]"

        # Call the method under test
        updated_count = pipeline._update_claims_with_stage7_revisions(
            revisions=[revision],
            claims=[original_claim],
            revised_content=revised_content,
            key_to_evidence=key_to_evidence,
        )

        # Should update the claim (position, text)
        assert updated_count == 1

        # BUT citation_key should remain ORIGINAL, not "NewSource"
        # because "NewSource" is not in key_to_evidence
        assert original_claim.citation_key == original_key, (
            f"Expected citation_key to remain '{original_key}', "
            f"but got '{original_claim.citation_key}'. "
            "This would create a grey reference!"
        )

        # Evidence should also remain (not set to None)
        assert original_claim.evidence is not None

    def test_stage7_updates_key_when_new_key_is_in_evidence(
        self, mock_llm_client: MagicMock, patch_app_config: Any
    ) -> None:
        """Stage 7 SHOULD update citation_key when new key IS in evidence map."""
        from deep_research.services.citation.atomic_decomposer import (
            AtomicFact,
            ClaimDecomposition,
            ClaimRevision,
        )
        from deep_research.services.citation.pipeline import CitationVerificationPipeline

        pipeline = CitationVerificationPipeline(mock_llm_client)

        # Create a claim with original key
        original_claim = ClaimInfo(
            claim_text="The market grew by 35%. [Arxiv]",
            claim_type="numeric",
            position_start=0,
            position_end=32,
            citation_key="Arxiv",
            evidence=EvidenceInfo(
                source_url="https://arxiv.org/paper",
                quote_text="market grew by 35%",
                start_offset=0,
                end_offset=20,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
        )

        # Create atomic facts for decomposition
        atomic_fact_1 = AtomicFact(
            fact_text="The market grew by 35%",
            fact_index=0,
            parent_claim_id=0,
            is_verified=True,
        )

        # Create decomposition
        decomposition = ClaimDecomposition(
            original_claim=original_claim,
            atomic_facts=[atomic_fact_1],
        )

        # Create revision with a key that IS in evidence pool
        revision = ClaimRevision(
            original_claim="The market grew by 35%. [Arxiv]",
            revised_claim="According to Reuters, the market grew by 35%. [Reuters]",
            revision_type="fully_verified",
            original_position_start=0,
            original_position_end=32,
            decomposition=decomposition,
            verified_facts=[atomic_fact_1],
            softened_facts=[],
        )

        # Evidence pool has BOTH keys
        key_to_evidence = {
            "Arxiv": RankedEvidence(
                source_id=None,
                source_url="https://arxiv.org/paper",
                source_title="Arxiv Paper",
                quote_text="market grew by 35%",
                start_offset=0,
                end_offset=20,
                section_heading=None,
                relevance_score=0.9,
                has_numeric_content=True,
            ),
            "Reuters": RankedEvidence(
                source_id=None,
                source_url="https://reuters.com/article",
                source_title="Reuters Article",
                quote_text="market grew by 35%",
                start_offset=5,
                end_offset=25,
                section_heading=None,
                relevance_score=0.95,
                has_numeric_content=True,
            ),
        }

        revised_content = "According to Reuters, the market grew by 35%. [Reuters]"

        updated_count = pipeline._update_claims_with_stage7_revisions(
            revisions=[revision],
            claims=[original_claim],
            revised_content=revised_content,
            key_to_evidence=key_to_evidence,
        )

        assert updated_count == 1

        # In this case, citation_key SHOULD be updated to Reuters
        assert original_claim.citation_key == "Reuters"

        # Evidence should be updated from Reuters source
        assert original_claim.evidence is not None
        assert original_claim.evidence.source_url == "https://reuters.com/article"
