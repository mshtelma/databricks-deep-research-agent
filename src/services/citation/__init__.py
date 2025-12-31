"""Citation verification services for claim-level attribution.

This module implements the 6-stage citation verification pipeline:
1. Evidence Pre-Selection - Extract relevant spans from sources
2. Interleaved Generation - Generate claims constrained by evidence
3. Confidence Classification - Route claims by verification complexity
4. Isolated Verification - Verify claims without generation context bias
5. Citation Correction - Fix incorrect citations post-hoc
6. Numeric QA Verification - QA-based verification for numeric claims
"""

from src.services.citation.claim_generator import InterleavedClaim, InterleavedGenerator
from src.services.citation.confidence_classifier import (
    ConfidenceClassifier,
    ConfidenceLevel,
    ConfidenceResult,
)
from src.services.citation.content_evaluator import (
    ContentQuality,
    evaluate_content_quality,
    filter_high_quality_sources,
)
from src.services.citation.evidence_selector import EvidencePreSelector, RankedEvidence
from src.services.citation.isolated_verifier import (
    IsolatedVerifier,
    Verdict,
    VerificationResult,
)
from src.services.citation.numeric_verifier import (
    NumericValue,
    NumericVerificationResult,
    NumericVerifier,
    QAVerificationResult,
)
from src.services.citation.citation_corrector import (
    CitationCorrector,
    CorrectionType,
    CorrectionResult,
    CorrectionMetrics,
)
from src.services.citation.pipeline import CitationVerificationPipeline, VerificationEvent

__all__ = [
    # Full pipeline orchestrator
    "CitationVerificationPipeline",
    "VerificationEvent",
    # Content quality filtering (pre-stage 1)
    "ContentQuality",
    "evaluate_content_quality",
    "filter_high_quality_sources",
    # Stage 1: Evidence Pre-Selection
    "EvidencePreSelector",
    "RankedEvidence",
    # Stage 2: Interleaved Generation
    "InterleavedGenerator",
    "InterleavedClaim",
    # Stage 3: Confidence Classification
    "ConfidenceClassifier",
    "ConfidenceLevel",
    "ConfidenceResult",
    # Stage 4: Isolated Verification
    "IsolatedVerifier",
    "Verdict",
    "VerificationResult",
    # Stage 5: Citation Correction
    "CitationCorrector",
    "CorrectionType",
    "CorrectionResult",
    "CorrectionMetrics",
    # Stage 6: Numeric Verification
    "NumericVerifier",
    "NumericValue",
    "NumericVerificationResult",
    "QAVerificationResult",
]
