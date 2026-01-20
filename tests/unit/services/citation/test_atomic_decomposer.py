"""Tests for Stage 7: Atomic Fact Decomposition service."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.services.citation.atomic_decomposer import (
    AtomicDecomposer,
    AtomicFact,
    ClaimDecomposition,
    ClaimRevision,
    DecompositionMetrics,
    EvidenceSource,
)


@dataclass
class MockClaimInfo:
    """Minimal ClaimInfo for testing."""

    claim_text: str
    claim_type: str = "general"
    position_start: int = 0
    position_end: int = 100
    verification_verdict: str | None = "unsupported"


@dataclass
class MockVerificationRetrievalConfig:
    """Mock configuration for verification retrieval."""

    trigger_on_verdicts: list[str] = None
    max_atomic_facts_per_claim: int = 5
    max_searches_per_fact: int = 2
    max_external_urls_per_search: int = 3
    entailment_threshold: float = 0.6
    internal_search_threshold: float = 0.7
    softening_strategy: str = "hedge"
    decomposition_timeout_seconds: float = 10.0
    search_timeout_seconds: float = 10.0
    crawl_timeout_seconds: float = 15.0
    decomposition_tier: str = "simple"
    entailment_tier: str = "analytical"
    reconstruction_tier: str = "analytical"

    def __post_init__(self):
        if self.trigger_on_verdicts is None:
            self.trigger_on_verdicts = ["unsupported", "partial"]


@dataclass
class MockLLMResponse:
    """Mock LLM response object."""

    content: str
    structured: object = None


class MockAtomicDecompositionOutput:
    """Mock structured output for atomic decomposition."""

    def __init__(self, atomic_facts: list[str], reasoning: str = ""):
        self.atomic_facts = atomic_facts
        self.reasoning = reasoning


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client for decomposer tests."""
    client = MagicMock()
    client.complete = AsyncMock(return_value=MockLLMResponse(content=""))
    return client


@pytest.fixture
def mock_config() -> MockVerificationRetrievalConfig:
    """Mock verification retrieval config."""
    return MockVerificationRetrievalConfig()


@pytest.fixture
def decomposer(mock_llm_client, mock_config) -> AtomicDecomposer:
    """Create AtomicDecomposer with mocked dependencies."""
    return AtomicDecomposer(llm=mock_llm_client, config=mock_config)


class TestAtomicFact:
    """Tests for AtomicFact dataclass."""

    def test_fact_creation(self):
        """Test creating an AtomicFact."""
        fact = AtomicFact(
            fact_text="OpenAI released GPT-4",
            fact_index=0,
            parent_claim_id=1,
        )
        assert fact.fact_text == "OpenAI released GPT-4"
        assert fact.fact_index == 0
        assert fact.parent_claim_id == 1
        assert fact.is_verified is False
        assert fact.evidence is None
        assert fact.evidence_source == EvidenceSource.NONE

    def test_fact_to_dict(self):
        """Test AtomicFact serialization."""
        fact = AtomicFact(
            fact_text="OpenAI released GPT-4",
            fact_index=0,
            parent_claim_id=1,
            is_verified=True,
            entailment_score=0.85,
        )
        result = fact.to_dict()
        assert result["fact_text"] == "OpenAI released GPT-4"
        assert result["is_verified"] is True
        assert result["entailment_score"] == 0.85


class TestClaimDecomposition:
    """Tests for ClaimDecomposition dataclass."""

    def test_decomposition_creation(self):
        """Test creating a ClaimDecomposition."""
        claim = MockClaimInfo(claim_text="Test claim")
        facts = [
            AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0),
            AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0),
        ]
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=facts,
        )
        assert decomposition.original_claim == claim
        assert len(decomposition.atomic_facts) == 2

    def test_update_verification_status_all_verified(self):
        """Test verification status when all facts verified."""
        claim = MockClaimInfo(claim_text="Test claim")
        facts = [
            AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=True),
            AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0, is_verified=True),
        ]
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=facts,
        )
        decomposition.update_verification_status()

        assert decomposition.all_verified is True
        assert decomposition.partial_verified is False
        assert decomposition.verified_count == 2
        assert decomposition.total_count == 2

    def test_update_verification_status_partial(self):
        """Test verification status when some facts verified."""
        claim = MockClaimInfo(claim_text="Test claim")
        facts = [
            AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=True),
            AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0, is_verified=False),
        ]
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=facts,
        )
        decomposition.update_verification_status()

        assert decomposition.all_verified is False
        assert decomposition.partial_verified is True
        assert decomposition.verified_count == 1
        assert decomposition.total_count == 2

    def test_update_verification_status_none_verified(self):
        """Test verification status when no facts verified."""
        claim = MockClaimInfo(claim_text="Test claim")
        facts = [
            AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=False),
            AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0, is_verified=False),
        ]
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=facts,
        )
        decomposition.update_verification_status()

        assert decomposition.all_verified is False
        assert decomposition.partial_verified is False
        assert decomposition.verified_count == 0
        assert decomposition.total_count == 2


class TestAtomicDecomposer:
    """Tests for AtomicDecomposer service."""

    @pytest.mark.asyncio
    async def test_decompose_short_claim_skipped(self, decomposer):
        """Test that short claims are not decomposed."""
        claim = MockClaimInfo(claim_text="Short claim test.")  # 3 words
        result = await decomposer.decompose(claim, claim_index=0)

        assert len(result.atomic_facts) == 1
        assert result.atomic_facts[0].fact_text == "Short claim test."
        assert "short enough" in result.decomposition_reasoning.lower() or "atomic" in result.decomposition_reasoning.lower()

    @pytest.mark.asyncio
    async def test_decompose_complex_claim(self, decomposer, mock_llm_client):
        """Test decomposition of a complex claim."""
        claim = MockClaimInfo(
            claim_text="OpenAI released GPT-4 in March 2023, which achieved 90% on the bar exam and is considered the most advanced model at the time."
        )

        # Mock LLM response
        mock_output = MockAtomicDecompositionOutput(
            atomic_facts=[
                "OpenAI released GPT-4.",
                "GPT-4 was released in March 2023.",
                "GPT-4 achieved 90% on the bar exam.",
                "GPT-4 was considered the most advanced model at the time.",
            ],
            reasoning="Decomposed into 4 independent facts.",
        )
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="", structured=mock_output
        )

        result = await decomposer.decompose(claim, claim_index=0)

        assert len(result.atomic_facts) == 4
        assert result.atomic_facts[0].fact_text == "OpenAI released GPT-4."
        assert result.decomposition_reasoning == "Decomposed into 4 independent facts."

    @pytest.mark.asyncio
    async def test_decompose_deduplicates_facts(self, decomposer, mock_llm_client):
        """Test that duplicate facts are deduplicated."""
        claim = MockClaimInfo(
            claim_text="The company grew significantly and the company expanded operations."
        )

        mock_output = MockAtomicDecompositionOutput(
            atomic_facts=[
                "The company grew.",
                "The company grew.",  # Duplicate
                "The company expanded operations.",
            ]
        )
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="", structured=mock_output
        )

        result = await decomposer.decompose(claim, claim_index=0)

        # Should have only 2 unique facts
        assert len(result.atomic_facts) == 2

    @pytest.mark.asyncio
    async def test_decompose_respects_max_facts(self, decomposer, mock_llm_client, mock_config):
        """Test that decomposition respects max_atomic_facts_per_claim config."""
        mock_config.max_atomic_facts_per_claim = 2
        decomposer.config = mock_config

        claim = MockClaimInfo(
            claim_text="A long claim with many facts that should be limited."
        )

        mock_output = MockAtomicDecompositionOutput(
            atomic_facts=[
                "Fact 1",
                "Fact 2",
                "Fact 3",
                "Fact 4",
                "Fact 5",
            ]
        )
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="", structured=mock_output
        )

        result = await decomposer.decompose(claim, claim_index=0)

        assert len(result.atomic_facts) <= 2

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self, decomposer, mock_llm_client):
        """Test fallback decomposition when LLM fails."""
        claim = MockClaimInfo(
            claim_text="This is a test claim that should fall back to single fact."
        )

        mock_llm_client.complete.side_effect = Exception("LLM error")

        result = await decomposer.decompose(claim, claim_index=0)

        # Should fallback to treating claim as single atomic fact
        assert len(result.atomic_facts) == 1
        assert result.atomic_facts[0].fact_text == claim.claim_text
        assert "fallback" in result.decomposition_reasoning.lower()

    @pytest.mark.asyncio
    async def test_decompose_empty_output_fallback(self, decomposer, mock_llm_client):
        """Test fallback when LLM returns empty facts."""
        claim = MockClaimInfo(
            claim_text="This is a test claim that returns empty decomposition."
        )

        mock_output = MockAtomicDecompositionOutput(atomic_facts=[])
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="", structured=mock_output
        )

        result = await decomposer.decompose(claim, claim_index=0)

        # Should fallback to original claim
        assert len(result.atomic_facts) == 1
        assert result.atomic_facts[0].fact_text == claim.claim_text

    @pytest.mark.asyncio
    async def test_decompose_claims_batch(self, decomposer, mock_llm_client):
        """Test batch decomposition of multiple claims."""
        claims = [
            MockClaimInfo(claim_text="First claim about something.", verification_verdict="unsupported"),
            MockClaimInfo(claim_text="Second claim about another thing.", verification_verdict="partial"),
            MockClaimInfo(claim_text="Third claim that is supported.", verification_verdict="supported"),  # Should be filtered
        ]

        # Mock each decomposition
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="",
            structured=MockAtomicDecompositionOutput(atomic_facts=["Single fact"])
        )

        decompositions, metrics = await decomposer.decompose_claims(claims)

        # Should only process unsupported and partial claims
        assert metrics.total_claims_processed == 2
        assert len(decompositions) == 2


class TestClaimRevision:
    """Tests for ClaimRevision dataclass."""

    def test_revision_creation(self):
        """Test creating a ClaimRevision."""
        claim = MockClaimInfo(claim_text="Original claim")
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[],
        )

        revision = ClaimRevision(
            original_claim="Original claim",
            revised_claim="Reportedly, original claim",
            revision_type="fully_softened",
            original_position_start=0,
            original_position_end=14,
            decomposition=decomposition,
        )

        assert revision.revision_type == "fully_softened"
        assert "reportedly" in revision.revised_claim.lower()

    def test_revision_to_dict(self):
        """Test ClaimRevision serialization."""
        claim = MockClaimInfo(claim_text="Original claim")
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[],
        )

        revision = ClaimRevision(
            original_claim="Original claim",
            revised_claim="Reportedly, original claim",
            revision_type="fully_softened",
            original_position_start=0,
            original_position_end=14,
            decomposition=decomposition,
            new_citations=["https://example.com"],
        )

        result = revision.to_dict()
        assert result["revision_type"] == "fully_softened"
        assert len(result["new_citations"]) == 1


class TestDecompositionMetrics:
    """Tests for DecompositionMetrics dataclass."""

    def test_metrics_avg_calculation(self):
        """Test average facts per claim calculation."""
        metrics = DecompositionMetrics(
            total_claims_processed=4,
            total_atomic_facts=12,
            single_fact_claims=1,
            multi_fact_claims=3,
        )
        metrics.compute_avg()

        assert metrics.avg_facts_per_claim == 3.0

    def test_metrics_avg_zero_claims(self):
        """Test average calculation with zero claims."""
        metrics = DecompositionMetrics()
        metrics.compute_avg()

        assert metrics.avg_facts_per_claim == 0.0
