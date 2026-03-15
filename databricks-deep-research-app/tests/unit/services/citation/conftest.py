"""Test fixtures for citation service tests."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import directly from modules to avoid circular import via __init__.py
# RankedEvidence is defined in evidence_selector, InterleavedClaim in claim_generator


@dataclass
class RankedEvidence:
    """Minimal RankedEvidence for testing (avoids circular import)."""
    source_id: str | None
    source_url: str
    source_title: str
    quote_text: str
    start_offset: int
    end_offset: int
    section_heading: str | None
    relevance_score: float
    has_numeric_content: bool


@dataclass
class InterleavedClaim:
    """Minimal InterleavedClaim for testing (avoids circular import)."""
    claim_text: str
    claim_type: str
    position_start: int
    position_end: int
    evidence: RankedEvidence | None
    evidence_index: int | None
    confidence_score: float | None


@dataclass
class MockLLMResponse:
    """Mock LLM response object."""

    content: str
    structured: object = None  # For structured output responses


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client for citation tests."""
    client = MagicMock()  # No spec to avoid circular import
    client.complete = AsyncMock(return_value=MockLLMResponse(content=""))
    return client


@pytest.fixture
def sample_evidence() -> RankedEvidence:
    """Sample RankedEvidence for testing."""
    return RankedEvidence(
        source_id=None,
        source_url="https://example.com/article",
        source_title="Example Article",
        quote_text="The market grew by 25% in 2024.",
        start_offset=100,
        end_offset=140,
        section_heading="Market Analysis",
        relevance_score=0.85,
        has_numeric_content=True,
    )


@pytest.fixture
def sample_evidence_list() -> list[RankedEvidence]:
    """Multiple evidence items for testing."""
    return [
        RankedEvidence(
            source_id=None,
            source_url="https://example.com/article1",
            source_title="Article One",
            quote_text="Revenue increased by $5 billion in 2024.",
            start_offset=0,
            end_offset=45,
            section_heading="Financial Results",
            relevance_score=0.9,
            has_numeric_content=True,
        ),
        RankedEvidence(
            source_id=None,
            source_url="https://example.com/article2",
            source_title="Article Two",
            quote_text="The company expanded into new markets.",
            start_offset=0,
            end_offset=40,
            section_heading="Business Strategy",
            relevance_score=0.75,
            has_numeric_content=False,
        ),
        RankedEvidence(
            source_id=None,
            source_url="https://example.com/article3",
            source_title="Article Three",
            quote_text="Customer satisfaction improved to 95%.",
            start_offset=0,
            end_offset=42,
            section_heading="Customer Metrics",
            relevance_score=0.8,
            has_numeric_content=True,
        ),
    ]


@pytest.fixture
def sample_claim() -> InterleavedClaim:
    """Sample InterleavedClaim for testing."""
    return InterleavedClaim(
        claim_text="The market experienced significant growth.",
        claim_type="general",
        position_start=0,
        position_end=45,
        evidence=None,
        evidence_index=None,
        confidence_score=0.8,
    )


@pytest.fixture
def sample_source_content() -> str:
    """Sample source content for testing span extraction."""
    return """
The global technology market experienced remarkable growth in 2024.

Revenue from cloud services increased by 35% year-over-year, reaching $500 billion.
This growth was primarily driven by enterprise adoption of AI solutions.

The Asia-Pacific region showed the strongest performance, with a 42% increase in
technology spending. Major companies invested heavily in infrastructure upgrades.

Looking ahead, analysts predict continued expansion in 2025, with particular
emphasis on generative AI applications and sustainable computing initiatives.
"""


@pytest.fixture
def mock_citation_config():
    """Mock citation verification config."""

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

    class AnswerComparisonMethod:
        """Enum-like for answer comparison method."""
        value: str = "f1"

    @dataclass
    class NumericQAVerificationConfig:
        answer_comparison_method: AnswerComparisonMethod = None
        rounding_tolerance: float = 0.05
        require_unit_match: bool = False
        require_entity_match: bool = False

        def __post_init__(self):
            if self.answer_comparison_method is None:
                self.answer_comparison_method = AnswerComparisonMethod()

    class ConfidenceEstimationMethod:
        """Enum-like for confidence estimation method."""
        value: str = "linguistic"

    @dataclass
    class ConfidenceClassificationConfig:
        high_threshold: float = 0.85
        low_threshold: float = 0.50
        quote_match_bonus: float = 0.3
        hedging_word_penalty: float = 0.2
        estimation_method: ConfidenceEstimationMethod = None

        def __post_init__(self):
            if self.estimation_method is None:
                self.estimation_method = ConfidenceEstimationMethod()

    class CorrectionMethod:
        """Enum-like for correction method."""
        value: str = "keyword_semantic_hybrid"

    @dataclass
    class CitationCorrectionConfig:
        correction_method: CorrectionMethod = None
        lambda_weight: float = 0.8
        correction_threshold: float = 0.6
        allow_alternate_citations: bool = True

        def __post_init__(self):
            if self.correction_method is None:
                self.correction_method = CorrectionMethod()

    # Import the real GenerationMode enum to ensure proper comparison
    from deep_research.core.app_config import GenerationMode as RealGenerationMode

    @dataclass
    class CitationVerificationConfig:
        enabled: bool = True
        # Use real enum with STRICT as default for backwards compatibility
        generation_mode: RealGenerationMode = RealGenerationMode.STRICT
        enable_evidence_preselection: bool = True
        enable_interleaved_generation: bool = True
        enable_confidence_classification: bool = True
        enable_citation_correction: bool = True
        enable_numeric_qa_verification: bool = True
        enable_verification_retrieval: bool = False
        unsupported_claim_warning_threshold: float = 0.20
        evidence_preselection: EvidencePreselectionConfig = None
        interleaved_generation: InterleavedGenerationConfig = None
        confidence_classification: ConfidenceClassificationConfig = None
        isolated_verification: IsolatedVerificationConfig = None
        citation_correction: CitationCorrectionConfig = None
        numeric_qa_verification: NumericQAVerificationConfig = None

        def __post_init__(self):
            if self.evidence_preselection is None:
                self.evidence_preselection = EvidencePreselectionConfig()
            if self.interleaved_generation is None:
                self.interleaved_generation = InterleavedGenerationConfig()
            if self.confidence_classification is None:
                self.confidence_classification = ConfidenceClassificationConfig()
            if self.isolated_verification is None:
                self.isolated_verification = IsolatedVerificationConfig()
            if self.citation_correction is None:
                self.citation_correction = CitationCorrectionConfig()
            if self.numeric_qa_verification is None:
                self.numeric_qa_verification = NumericQAVerificationConfig()

    @dataclass
    class MockAppConfig:
        citation_verification: CitationVerificationConfig = None

        def __post_init__(self):
            if self.citation_verification is None:
                self.citation_verification = CitationVerificationConfig()

    return MockAppConfig()


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset config cache before and after each test."""
    from deep_research.core.app_config import clear_config_cache
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def patch_app_config(mock_citation_config):
    """Patch get_app_config to return mock config in all citation modules."""
    with (
        patch("deep_research.core.app_config.get_app_config", return_value=mock_citation_config),
        patch("deep_research.core.app_config.load_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.pipeline.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.evidence_selector.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.claim_generator.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.isolated_verifier.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.numeric_verifier.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.confidence_classifier.get_app_config", return_value=mock_citation_config),
        patch("deep_research.services.citation.citation_corrector.get_app_config", return_value=mock_citation_config),
    ):
        yield mock_citation_config
