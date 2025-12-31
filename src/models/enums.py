"""Enums for citation verification models.

This module defines all enums used in the claim-level citation feature:
- ClaimType: General vs. numeric claims
- VerificationVerdict: Four-tier verification outcomes
- ConfidenceLevel: HaluGate-style routing classification
- CorrectionType: Citation correction outcomes
- DerivationType: How numeric values were obtained
"""

from enum import Enum


class ClaimType(str, Enum):
    """Type of claim for specialized handling."""

    GENERAL = "general"  # Standard factual claim
    NUMERIC = "numeric"  # Claim containing numeric values requiring QA verification


class VerificationVerdict(str, Enum):
    """Four-tier verification verdict for claims.

    Based on research showing the need to distinguish between:
    - Full support (claim is completely entailed)
    - Partial support (some aspects supported, others unstated)
    - No support (no evidence basis, but not contradicted)
    - Contradiction (evidence directly opposes the claim)
    """

    SUPPORTED = "supported"  # Claim FULLY entailed by evidence
    PARTIAL = "partial"  # Some aspects supported, others unstated
    UNSUPPORTED = "unsupported"  # No evidence basis (not contradicted)
    CONTRADICTED = "contradicted"  # Evidence DIRECTLY opposes claim


class ConfidenceLevel(str, Enum):
    """HaluGate-style confidence level for verification routing.

    Used to route claims to appropriate verification paths:
    - HIGH: Quick verification (direct quotes, exact matches)
    - MEDIUM: Standard verification (paraphrased facts)
    - LOW: Full verification (hedged, comparative, synthetic claims)
    """

    HIGH = "high"  # >0.85 confidence - direct quotes, exact matches
    MEDIUM = "medium"  # 0.50-0.85 - paraphrased facts
    LOW = "low"  # <0.50 - hedged, comparative, synthetic


class CorrectionType(str, Enum):
    """Type of citation correction applied during post-processing.

    From CiteFix research:
    - KEEP: Original citation is correct (~60% of cases)
    - REPLACE: Found better citation from pool (~25% of cases)
    - REMOVE: No valid citation exists (~10% of cases)
    - ADD_ALTERNATE: Multiple valid citations available (~5% of cases)
    """

    KEEP = "keep"  # Citation is correct, no change needed
    REPLACE = "replace"  # Found better citation from evidence pool
    REMOVE = "remove"  # No valid citation exists
    ADD_ALTERNATE = "add_alternate"  # Multiple valid citations, added alternate


class DerivationType(str, Enum):
    """How a numeric value was obtained.

    Important for provenance tracking:
    - DIRECT: Value quoted directly from source
    - COMPUTED: Value calculated from source values
    """

    DIRECT = "direct"  # Quoted directly from source
    COMPUTED = "computed"  # Calculated from source values
