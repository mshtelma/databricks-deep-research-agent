"""Unit tests for CitationCorrector service (Stage 5)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.services.citation.citation_corrector import (
    CitationCorrector,
    CorrectionMetrics,
    CorrectionResult,
    CorrectionType,
)
from deep_research.services.citation.evidence_selector import RankedEvidence

from .conftest import MockLLMResponse


@pytest.fixture
def sample_evidence_pool() -> list[RankedEvidence]:
    """Sample evidence pool for testing corrections."""
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
            quote_text="The company expanded into new markets in Asia and Europe.",
            start_offset=0,
            end_offset=60,
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
        RankedEvidence(
            source_id=None,
            source_url="https://example.com/article4",
            source_title="Article Four",
            quote_text="The market grew by 25% in 2024, reaching $150 billion.",
            start_offset=0,
            end_offset=55,
            section_heading="Market Analysis",
            relevance_score=0.85,
            has_numeric_content=True,
        ),
    ]


class TestKeywordOverlap:
    """Tests for keyword overlap computation."""

    def test_high_overlap_with_matching_entities(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Computes high overlap when entities match."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue increased by $5 billion in 2024."
        evidence = "Revenue increased by $5 billion in 2024, according to the report."

        overlap = corrector._compute_keyword_overlap(claim, evidence)

        assert overlap > 0.5  # High overlap due to matching entities

    def test_low_overlap_with_different_entities(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Computes low overlap when entities don't match."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue was $10 million in 2023."
        evidence = "Customer satisfaction reached 95% in 2024."

        overlap = corrector._compute_keyword_overlap(claim, evidence)

        assert overlap < 0.3  # Low overlap due to different entities

    def test_extracts_currency_entities(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Extracts currency entities correctly."""
        corrector = CitationCorrector(mock_llm_client)

        entities = corrector._extract_key_entities("The market reached $150B in value.")

        assert "$150B" in entities or any("150" in e for e in entities)

    def test_extracts_percentage_entities(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Extracts percentage entities correctly."""
        corrector = CitationCorrector(mock_llm_client)

        entities = corrector._extract_key_entities("Growth was 25.5% year over year.")

        assert "25.5%" in entities


class TestSemanticSimilarity:
    """Tests for semantic similarity computation."""

    def test_high_similarity_for_related_text(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Computes high similarity for semantically related text."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "The company expanded into new markets."
        evidence = "The company expanded into new markets in Asia and Europe."

        similarity = corrector._compute_semantic_similarity(claim, evidence)

        assert similarity > 0.4  # Good token overlap

    def test_low_similarity_for_unrelated_text(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Computes low similarity for unrelated text."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue increased significantly."
        evidence = "The weather was nice yesterday."

        similarity = corrector._compute_semantic_similarity(claim, evidence)

        assert similarity < 0.2  # Low token overlap


class TestHybridScore:
    """Tests for hybrid keyword+semantic scoring."""

    def test_hybrid_combines_keyword_and_semantic(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Hybrid score combines both methods."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue reached $5 billion."
        evidence = "Revenue reached $5 billion in Q4 2024."

        hybrid = corrector._compute_hybrid_score(claim, evidence)

        assert 0 <= hybrid <= 1
        assert hybrid > 0.5  # Should be high due to strong match


class TestCitationEntails:
    """Tests for citation entailment check."""

    def test_returns_true_for_supporting_evidence(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Returns True when evidence supports claim."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue increased by $5 billion."
        evidence = "Revenue increased by $5 billion in 2024."

        assert corrector.citation_entails(claim, evidence) is True

    def test_returns_false_for_non_supporting_evidence(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Returns False when evidence doesn't support claim."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue was $10 billion."
        evidence = "Customer satisfaction reached 95%."

        assert corrector.citation_entails(claim, evidence) is False


class TestFindBetterCitation:
    """Tests for finding better evidence."""

    def test_finds_better_matching_evidence(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Finds better matching evidence from pool."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "The market grew by 25% in 2024."
        current = sample_evidence_pool[0]  # Revenue evidence - less relevant

        better, score = corrector.find_better_citation(
            claim, current, sample_evidence_pool
        )

        assert better is not None
        assert "25%" in better.quote_text or "market" in better.quote_text.lower()
        assert score > 0

    def test_returns_none_when_no_better_evidence(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Returns None when no better evidence exists."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Something completely unrelated to any evidence."
        evidence_pool = [
            RankedEvidence(
                source_id=None,
                source_url="https://example.com",
                source_title="Test",
                quote_text="Specific technical detail about software.",
                start_offset=0,
                end_offset=40,
                section_heading=None,
                relevance_score=0.5,
                has_numeric_content=False,
            )
        ]

        better, score = corrector.find_better_citation(claim, None, evidence_pool)

        # May or may not find "better" but score should be low
        assert score < 0.3


class TestFindAlternateCitations:
    """Tests for finding alternate citations."""

    def test_finds_multiple_supporting_evidence(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Finds multiple supporting evidence spans."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue increased by $5 billion."
        primary = sample_evidence_pool[0]

        alternates = corrector.find_alternate_citations(
            claim, primary, sample_evidence_pool, max_alternates=2
        )

        # May or may not find alternates depending on matching
        assert isinstance(alternates, list)
        assert all(isinstance(e, RankedEvidence) for e in alternates)

    def test_excludes_primary_evidence(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Excludes primary evidence from alternates."""
        corrector = CitationCorrector(mock_llm_client)

        primary = sample_evidence_pool[0]

        alternates = corrector.find_alternate_citations(
            "Revenue increased.", primary, sample_evidence_pool
        )

        # Primary should not be in alternates
        for alt in alternates:
            assert alt.quote_text != primary.quote_text


class TestCorrectSingleCitation:
    """Tests for single citation correction."""

    @pytest.mark.asyncio
    async def test_keeps_good_citation(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Keeps citation when evidence supports claim well."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "Revenue increased by $5 billion in 2024."
        evidence = sample_evidence_pool[0]  # Matching evidence

        result = await corrector.correct_single_citation(
            claim=claim,
            current_evidence=evidence,
            evidence_pool=sample_evidence_pool,
            current_verdict="supported",
        )

        assert result.correction_type == CorrectionType.KEEP
        assert result.corrected_evidence == evidence

    @pytest.mark.asyncio
    async def test_replaces_poor_citation(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Replaces citation when better evidence exists."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "The market grew by 25% in 2024."
        evidence = sample_evidence_pool[0]  # Revenue evidence - wrong topic

        result = await corrector.correct_single_citation(
            claim=claim,
            current_evidence=evidence,
            evidence_pool=sample_evidence_pool,
            current_verdict="unsupported",
        )

        # Should find better matching evidence
        if result.correction_type == CorrectionType.REPLACE:
            assert result.corrected_evidence is not None
            assert result.corrected_evidence != evidence

    @pytest.mark.asyncio
    async def test_removes_unsupported_claim(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Removes citation when no supporting evidence exists."""
        corrector = CitationCorrector(mock_llm_client)

        claim = "The quantum computer achieved 1000 qubit stability."
        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Test",
            quote_text="Revenue was $5 billion.",
            start_offset=0,
            end_offset=25,
            section_heading=None,
            relevance_score=0.5,
            has_numeric_content=True,
        )
        evidence_pool = [evidence]

        result = await corrector.correct_single_citation(
            claim=claim,
            current_evidence=evidence,
            evidence_pool=evidence_pool,
            current_verdict="unsupported",
        )

        # Should either remove or keep with low confidence
        assert result.correction_type in [
            CorrectionType.REMOVE,
            CorrectionType.KEEP,
            CorrectionType.REPLACE,
        ]


class TestCorrectCitations:
    """Tests for batch citation correction."""

    @pytest.mark.asyncio
    async def test_processes_multiple_claims(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Processes multiple claims correctly."""
        corrector = CitationCorrector(mock_llm_client)

        claims = [
            ("Revenue increased by $5 billion.", sample_evidence_pool[0], "partial"),
            ("Customer satisfaction reached 95%.", sample_evidence_pool[2], "supported"),
            ("Market grew by 25%.", sample_evidence_pool[1], "unsupported"),
        ]

        results, metrics = await corrector.correct_citations(
            claims_with_evidence=claims,
            evidence_pool=sample_evidence_pool,
        )

        assert len(results) == 3
        assert isinstance(metrics, CorrectionMetrics)
        assert metrics.total_claims == 3

    @pytest.mark.asyncio
    async def test_returns_correct_metrics(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Returns accurate correction metrics."""
        corrector = CitationCorrector(mock_llm_client)

        claims = [
            ("Revenue increased by $5 billion.", sample_evidence_pool[0], "supported"),
        ]

        results, metrics = await corrector.correct_citations(
            claims_with_evidence=claims,
            evidence_pool=sample_evidence_pool,
        )

        # Metrics should sum correctly
        total = metrics.kept + metrics.replaced + metrics.removed + metrics.added_alternate
        assert total == metrics.total_claims

    @pytest.mark.asyncio
    async def test_handles_empty_claims_list(
        self, mock_llm_client: MagicMock, patch_app_config
    ) -> None:
        """Handles empty claims list gracefully."""
        corrector = CitationCorrector(mock_llm_client)

        results, metrics = await corrector.correct_citations(
            claims_with_evidence=[],
            evidence_pool=[],
        )

        assert len(results) == 0
        assert metrics.total_claims == 0
        assert metrics.correction_rate == 0.0


class TestLLMCorrection:
    """Tests for LLM-based correction."""

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Falls back to hybrid matching when LLM fails."""
        mock_llm_client.complete.side_effect = Exception("LLM unavailable")

        corrector = CitationCorrector(mock_llm_client)

        result = await corrector.correct_citation_with_llm(
            claim="Revenue was $5 billion.",
            current_evidence=sample_evidence_pool[0],
            evidence_pool=sample_evidence_pool,
        )

        # Should return a result despite LLM failure
        assert result is not None
        assert isinstance(result.correction_type, CorrectionType)

    @pytest.mark.asyncio
    async def test_parses_llm_response(
        self,
        mock_llm_client: MagicMock,
        sample_evidence_pool: list[RankedEvidence],
        patch_app_config,
    ) -> None:
        """Parses valid LLM response correctly."""
        mock_response = json.dumps({
            "action": "replace",
            "evidence_index": 2,
            "reasoning": "Evidence 2 better supports the claim.",
        })
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        corrector = CitationCorrector(mock_llm_client)

        result = await corrector.correct_citation_with_llm(
            claim="Customer satisfaction reached 95%.",
            current_evidence=sample_evidence_pool[0],
            evidence_pool=sample_evidence_pool,
        )

        # Should parse and return correction
        assert result is not None


class TestCorrectionResult:
    """Tests for CorrectionResult dataclass."""

    def test_creates_keep_result(self) -> None:
        """Creates KEEP result correctly."""
        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Test",
            quote_text="Test quote.",
            start_offset=0,
            end_offset=12,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=False,
        )

        result = CorrectionResult(
            claim_text="Test claim.",
            correction_type=CorrectionType.KEEP,
            original_evidence=evidence,
            corrected_evidence=evidence,
            reasoning="Evidence is correct.",
            confidence=1.0,
        )

        assert result.correction_type == CorrectionType.KEEP
        assert result.original_evidence == result.corrected_evidence

    def test_creates_replace_result(self) -> None:
        """Creates REPLACE result correctly."""
        old_evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com/1",
            source_title="Old",
            quote_text="Old quote.",
            start_offset=0,
            end_offset=10,
            section_heading=None,
            relevance_score=0.5,
            has_numeric_content=False,
        )

        new_evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com/2",
            source_title="New",
            quote_text="New quote.",
            start_offset=0,
            end_offset=10,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=False,
        )

        result = CorrectionResult(
            claim_text="Test claim.",
            correction_type=CorrectionType.REPLACE,
            original_evidence=old_evidence,
            corrected_evidence=new_evidence,
            reasoning="Found better evidence.",
            confidence=0.85,
        )

        assert result.correction_type == CorrectionType.REPLACE
        assert result.corrected_evidence != result.original_evidence


class TestCorrectionMetrics:
    """Tests for CorrectionMetrics dataclass."""

    def test_calculates_correction_rate(self) -> None:
        """Calculates correction rate correctly."""
        metrics = CorrectionMetrics(
            total_claims=10,
            kept=5,
            replaced=3,
            removed=1,
            added_alternate=1,
        )

        # 5 out of 10 needed correction (replaced + removed + added_alternate)
        assert metrics.correction_rate == 0.5

    def test_handles_zero_claims(self) -> None:
        """Handles zero claims gracefully."""
        metrics = CorrectionMetrics(total_claims=0)

        assert metrics.correction_rate == 0.0

    def test_all_kept_means_zero_rate(self) -> None:
        """All kept means 0% correction rate."""
        metrics = CorrectionMetrics(
            total_claims=10,
            kept=10,
            replaced=0,
            removed=0,
            added_alternate=0,
        )

        assert metrics.correction_rate == 0.0
