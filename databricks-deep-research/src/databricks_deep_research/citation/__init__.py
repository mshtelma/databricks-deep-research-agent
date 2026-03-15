"""Citation verification pipeline, types, and configuration.

This package provides the 7-stage citation verification pipeline orchestrator
and all associated data types.  NO app-level dependencies.

Public API:
    Pipeline:
        ``CitationVerificationPipeline`` -- orchestrator
        ``VerificationEvent`` -- streaming event
    Config:
        ``CitationConfig`` and stage sub-configs
    Types:
        All data types from ``citation.types``
"""

from databricks_deep_research.citation.analysis_grounding import AnalysisGroundingVerifier
from databricks_deep_research.citation.config import (
    AnswerComparisonMethod,
    CitationConfig,
    CitationCorrectionConfig,
    ConfidenceClassificationConfig,
    ConfidenceEstimationMethod,
    CorrectionMethod,
    EvidencePreselectionConfig,
    GenerationMode,
    GroundingValidationConfig,
    InterleavedGenerationConfig,
    IsolatedVerificationConfig,
    NumericQAVerificationConfig,
    PostVerificationConfig,
    ReactSynthesisConfig,
    RelevanceMethod,
    SofteningStrategy,
    SynthesisMode,
    VerificationRetrievalConfig,
)
from databricks_deep_research.citation.pipeline import (
    CitationVerificationPipeline,
    VerificationEvent,
)
from databricks_deep_research.citation.types import (
    AnalysisSummaryInfo,
    ClaimInfo,
    ClaimRole,
    ConfidenceLevel,
    ConfidenceResult,
    ContentQuality,
    CorrectionAction,
    CorrectionMetrics,
    CorrectionResult,
    EvidenceInfo,
    EvidenceSpanOutput,
    InterleavedClaim,
    NumericValue,
    NumericVerificationResult,
    QAVerificationResult,
    RankedEvidence,
    VerificationMethod,
    VerificationOutput,
    VerificationResult,
    VerificationSummaryInfo,
    VerificationVerdict,
)

__all__ = [
    # --- Pipeline (pipeline.py) ---
    "CitationVerificationPipeline",
    "VerificationEvent",
    # --- Types (types.py) ---
    # Enums
    "VerificationVerdict",
    "CorrectionAction",
    "ConfidenceLevel",
    "ClaimRole",
    "VerificationMethod",
    # Evidence (Stage 1)
    "EvidenceInfo",
    "RankedEvidence",
    # Claims (Stage 2)
    "InterleavedClaim",
    "ClaimInfo",
    # Confidence (Stage 3)
    "ConfidenceResult",
    # Verification (Stage 4)
    "VerificationResult",
    # Correction (Stage 5)
    "CorrectionResult",
    "CorrectionMetrics",
    # Numeric (Stage 6)
    "NumericValue",
    "QAVerificationResult",
    "NumericVerificationResult",
    # Summary
    "VerificationSummaryInfo",
    "AnalysisSummaryInfo",
    # Content quality
    "ContentQuality",
    # Structured LLM output schemas
    "EvidenceSpanOutput",
    "VerificationOutput",
    "AnalysisGroundingVerifier",
    # --- Config (config.py) ---
    "CitationConfig",
    # Strategy / method enums
    "RelevanceMethod",
    "AnswerComparisonMethod",
    "ConfidenceEstimationMethod",
    "CorrectionMethod",
    "SofteningStrategy",
    "GenerationMode",
    "SynthesisMode",
    # Stage configs
    "EvidencePreselectionConfig",
    "InterleavedGenerationConfig",
    "ConfidenceClassificationConfig",
    "IsolatedVerificationConfig",
    "CitationCorrectionConfig",
    "NumericQAVerificationConfig",
    "VerificationRetrievalConfig",
    "GroundingValidationConfig",
    "PostVerificationConfig",
    "ReactSynthesisConfig",
]
