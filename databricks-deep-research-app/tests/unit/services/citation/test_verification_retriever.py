"""Tests for Stage 7: Verification Retrieval service."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.services.citation.atomic_decomposer import (
    AtomicFact,
    ClaimDecomposition,
    ClaimRevision,
    EvidenceSource,
)
from deep_research.services.citation.verification_retriever import (
    InternalPoolSearcher,
    VerificationEvent,
    VerificationRetriever,
    VerificationRetrievalMetrics,
)


@dataclass
class MockClaimInfo:
    """Minimal ClaimInfo for testing."""

    claim_text: str
    claim_type: str = "general"
    position_start: int = 0
    position_end: int = 100
    verification_verdict: str | None = "unsupported"
    abstained: bool = False


@dataclass
class MockRankedEvidence:
    """Minimal RankedEvidence for testing."""

    source_id: str | None = None
    source_url: str = "https://example.com"
    source_title: str | None = "Example"
    quote_text: str = "Example quote text."
    start_offset: int | None = 0
    end_offset: int | None = 50
    section_heading: str | None = None
    relevance_score: float = 0.8
    has_numeric_content: bool = False


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


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client for retriever tests."""
    client = MagicMock()
    client.complete = AsyncMock(return_value=MockLLMResponse(content=""))
    return client


@pytest.fixture
def mock_config() -> MockVerificationRetrievalConfig:
    """Mock verification retrieval config."""
    return MockVerificationRetrievalConfig()


@pytest.fixture
def sample_evidence_pool() -> list[MockRankedEvidence]:
    """Sample evidence pool for testing."""
    return [
        MockRankedEvidence(
            source_url="https://example.com/article1",
            source_title="Article One",
            quote_text="The company released a new product in March 2024.",
            relevance_score=0.9,
        ),
        MockRankedEvidence(
            source_url="https://example.com/article2",
            source_title="Article Two",
            quote_text="Revenue increased by 25% year-over-year.",
            relevance_score=0.85,
            has_numeric_content=True,
        ),
        MockRankedEvidence(
            source_url="https://example.com/article3",
            source_title="Article Three",
            quote_text="Expanding operations into new markets across Asia.",
            relevance_score=0.75,
        ),
    ]


class TestInternalPoolSearcher:
    """Tests for internal evidence pool searcher."""

    def test_empty_pool(self):
        """Test search with empty evidence pool."""
        searcher = InternalPoolSearcher([])
        results = searcher.search("test fact")
        assert len(results) == 0

    def test_search_with_matching_terms(self, sample_evidence_pool):
        """Test search finds matching evidence."""
        searcher = InternalPoolSearcher(sample_evidence_pool)
        results = searcher.search("company released product", threshold=0.1)

        assert len(results) > 0
        # First result should have highest score
        evidence, score = results[0]
        assert "released" in evidence.quote_text.lower() or "product" in evidence.quote_text.lower()

    def test_search_respects_threshold(self, sample_evidence_pool):
        """Test search respects similarity threshold."""
        searcher = InternalPoolSearcher(sample_evidence_pool)

        # High threshold should return fewer results
        high_results = searcher.search("company released product", threshold=0.9)
        low_results = searcher.search("company released product", threshold=0.1)

        assert len(high_results) <= len(low_results)

    def test_search_respects_top_k(self, sample_evidence_pool):
        """Test search respects top_k limit."""
        searcher = InternalPoolSearcher(sample_evidence_pool)
        results = searcher.search("company operations", threshold=0.1, top_k=1)

        assert len(results) <= 1

    def test_tokenize_removes_short_tokens(self, sample_evidence_pool):
        """Test tokenizer removes tokens shorter than 3 chars."""
        searcher = InternalPoolSearcher(sample_evidence_pool)
        tokens = searcher._tokenize("a to the an is")

        # Short tokens should be removed
        assert "a" not in tokens
        assert "to" not in tokens
        assert "an" not in tokens
        assert "is" not in tokens


class TestVerificationRetrievalMetrics:
    """Tests for VerificationRetrievalMetrics."""

    def test_metrics_initialization(self):
        """Test metrics default values."""
        metrics = VerificationRetrievalMetrics()

        assert metrics.total_claims_processed == 0
        assert metrics.facts_verified == 0
        assert metrics.external_searches == 0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = VerificationRetrievalMetrics(
            total_claims_processed=5,
            facts_verified=3,
            facts_softened=2,
            external_searches=10,
        )

        result = metrics.to_dict()
        assert result["total_claims_processed"] == 5
        assert result["facts_verified"] == 3
        assert result["external_searches"] == 10


class TestVerificationEvent:
    """Tests for VerificationEvent."""

    def test_event_creation(self):
        """Test creating a verification event."""
        event = VerificationEvent(
            event_type="claim_decomposed",
            data={"claim_id": 1, "fact_count": 3},
        )

        assert event.event_type == "claim_decomposed"
        assert event.data["claim_id"] == 1
        assert event.data["fact_count"] == 3


class TestVerificationRetriever:
    """Tests for VerificationRetriever service."""

    def test_filter_claims_by_verdict(self, mock_llm_client, mock_config):
        """Test that claims are filtered by verdict."""
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        claims = [
            MockClaimInfo(claim_text="Unsupported claim", verification_verdict="unsupported"),
            MockClaimInfo(claim_text="Partial claim", verification_verdict="partial"),
            MockClaimInfo(claim_text="Supported claim", verification_verdict="supported"),
            MockClaimInfo(claim_text="Contradicted claim", verification_verdict="contradicted"),
        ]

        filtered = retriever._filter_claims(claims)

        # Should only include unsupported and partial
        assert len(filtered) == 2
        verdicts = [c.verification_verdict for _, c in filtered]
        assert "unsupported" in verdicts
        assert "partial" in verdicts
        assert "supported" not in verdicts
        assert "contradicted" not in verdicts

    def test_apply_revision_to_report(self, mock_llm_client, mock_config):
        """Test applying a revision to report content."""
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        report = "This is the original claim that needs revision. More content follows."
        claim = MockClaimInfo(claim_text="original claim that needs revision")
        decomposition = ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[],
        )

        revision = ClaimRevision(
            original_claim="original claim that needs revision",
            revised_claim="reportedly, the original claim that needs revision",
            revision_type="fully_softened",
            original_position_start=12,
            original_position_end=46,
            decomposition=decomposition,
        )

        updated, offset = retriever.apply_revision_to_report(report, revision)

        assert "reportedly" in updated.lower()
        assert offset != 0  # Length changed

    def test_apply_all_revisions_reverse_order(self, mock_llm_client, mock_config):
        """Test that multiple revisions are applied correctly."""
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        report = "Claim one here. Claim two here."
        claim1 = MockClaimInfo(claim_text="Claim one")
        claim2 = MockClaimInfo(claim_text="Claim two")

        revision1 = ClaimRevision(
            original_claim="Claim one",
            revised_claim="Revised one",
            revision_type="partially_softened",
            original_position_start=0,
            original_position_end=9,
            decomposition=ClaimDecomposition(original_claim=claim1, atomic_facts=[]),
        )
        revision2 = ClaimRevision(
            original_claim="Claim two",
            revised_claim="Revised two",
            revision_type="partially_softened",
            original_position_start=16,
            original_position_end=25,
            decomposition=ClaimDecomposition(original_claim=claim2, atomic_facts=[]),
        )

        updated = retriever.apply_all_revisions(report, [revision1, revision2])

        assert "Revised one" in updated
        assert "Revised two" in updated

    def test_extract_domain(self, mock_llm_client, mock_config):
        """Test domain extraction from URL."""
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        assert retriever._extract_domain("https://example.com/article") == "Example"
        assert retriever._extract_domain("https://www.reuters.com/news") == "Reuters"
        assert retriever._extract_domain("https://arxiv.org/paper") == "Arxiv"

    def test_get_model_tier(self, mock_llm_client, mock_config):
        """Test tier name to ModelTier conversion."""
        from deep_research.services.llm.types import ModelTier

        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        assert retriever._get_model_tier("simple") == ModelTier.SIMPLE
        assert retriever._get_model_tier("analytical") == ModelTier.ANALYTICAL
        assert retriever._get_model_tier("complex") == ModelTier.COMPLEX
        assert retriever._get_model_tier("unknown") == ModelTier.SIMPLE  # Default fallback


class TestVerificationRetrieverAsync:
    """Async tests for VerificationRetriever."""

    @pytest.mark.asyncio
    async def test_retrieve_and_revise_no_claims(self, mock_llm_client, mock_config):
        """Test retrieval with no claims to process."""
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        claims = [
            MockClaimInfo(claim_text="Supported", verification_verdict="supported"),
        ]

        events = []
        async for event in retriever.retrieve_and_revise(
            claims=claims,
            evidence_pool=[],
            report_content="Test report",
            research_query="test",
        ):
            events.append(event)

        # Should emit skipped event
        assert len(events) == 1
        assert events[0].event_type == "stage_7_skipped"

    @pytest.mark.asyncio
    async def test_check_entailment_returns_false_on_error(self, mock_llm_client, mock_config):
        """Test entailment check returns False on error."""
        mock_llm_client.complete.side_effect = Exception("LLM error")
        retriever = VerificationRetriever(llm=mock_llm_client, config=mock_config)

        fact = AtomicFact(fact_text="Test fact", fact_index=0, parent_claim_id=0)
        evidence = MockRankedEvidence()

        entails, score = await retriever._check_entailment(fact, evidence, claim_index=0)

        assert entails is False
        assert score == 0.0
