"""Claim role classification -- fact, analysis, or free heuristics.

Extracted from pipeline.py to reduce god-object complexity.
Pure functions with no I/O or LLM calls.
"""

from __future__ import annotations

import re

from databricks_deep_research.citation.types import ClaimInfo, ClaimRole
from databricks_deep_research.citation.utils import NUMERIC_PATTERN, TEMPORAL_PATTERN

# ---------------------------------------------------------------------------
# Constants (moved from pipeline.py)
# ---------------------------------------------------------------------------

ANALYSIS_ROLE_CUES: tuple[str, ...] = (
    "suggests",
    "may indicate",
    "appears consistent with",
    "appears to",
    "indicates",
    "reflects",
    "demonstrates",
    "shows that",
    "implies",
    "signals",
    "points to",
    "distorted",
    "obscured",
    "momentum",
    "trajectory",
    "resilience",
    "headwind",
    "tailwind",
    "positioned",
    "strong foundation",
    "healthy performance",
    "strong performance",
    "positive momentum",
    "complex earnings picture",
    "earnings picture",
    "growth driver",
    "bright spot",
    "current earnings trajectory",
    "essential context",
    "comparable store sales momentum",
)

STRUCTURAL_TEXT_PATTERNS: tuple[str, ...] = (
    "introduction",
    "conclusion",
    "overview",
    "summary",
    "in summary",
    "overall",
    "the following sections",
    "this report examines",
    "this analysis examines",
)

FACTUAL_PAYLOAD_PATTERNS = re.compile(
    r"\b("
    r"reported|increased|decreased|declined|reached|totaled|includes|announced|"
    r"delivered|generated|recorded|operating profit|operating loss|eps|sales|guidance|"
    r"quarter|full-year|fiscal|digital growth|ecommerce"
    r")\b",
    re.IGNORECASE,
)

ANALYSIS_SPLIT_MARKERS: tuple[str, ...] = (
    " but still indicating ",
    " but still suggesting ",
    " but still reflecting ",
    " indicating ",
    " suggesting ",
    " reflecting ",
    " demonstrating ",
    " showing ",
    " highlighting ",
    " marking ",
    " continuing ",
    " which suggests ",
    " which indicates ",
    " which reflects ",
    " which demonstrates ",
)

ANALYSIS_TAIL_PATTERN = re.compile(
    r",\s*(?:"
    r"(?:(?:but|and)\s+)?(?:still\s+)?"
    r"marking|continuing|demonstrating|highlighting|suggesting|indicating|"
    r"reflecting|showing|underscoring"
    r")\b.*$",
    re.IGNORECASE,
)

LEADING_CONCESSIVE_PATTERN = re.compile(
    r"^(?:(?:some sources indicate that|according to available information,|reportedly,)\s+)?"
    r"(?:(?:despite|while|although|though|however)\b[^,]*,\s*)+",
    re.IGNORECASE,
)

MATERIAL_ANALYSIS_MARKERS: tuple[str, ...] = (
    "because",
    "due to",
    "driven by",
    "reflects",
    "indicates",
    "suggests",
    "demonstrates",
    "strongest",
    "weakest",
    "accelerating",
    "momentum",
    "trajectory",
    "healthy",
    "robust",
    "resilience",
    "outperformed",
    "exceeded expectations",
    "non-recurring",
)


# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------


def claim_has_citation_keys(claim: ClaimInfo) -> bool:
    """Return True if the claim has any citation keys attached."""
    return bool(claim.citation_keys or claim.citation_key)


def looks_structural(text: str) -> bool:
    """Return True if *text* appears to be structural (heading, label, etc.)."""
    stripped = text.strip().lower()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if re.match(r"^\*\*[^*]+\*\*:\s*", text.strip()):
        return True
    if stripped.endswith(":") and not FACTUAL_PAYLOAD_PATTERNS.search(stripped):
        return True
    return any(pattern in stripped for pattern in STRUCTURAL_TEXT_PATTERNS)


def contains_metric_payload(text: str) -> bool:
    """Return True if *text* contains a meaningful numeric metric."""
    for match in NUMERIC_PATTERN.finditer(text):
        raw = match.group(0).strip().lower()
        if re.fullmatch(r"20\d{2}", raw):
            continue
        if re.fullmatch(r"\(?\d+\)?", raw):
            continue
        return True
    return False


def contains_numeric_or_date_payload(text: str) -> bool:
    """Return True if *text* contains numeric metrics or date+factual content."""
    if contains_metric_payload(text):
        return True
    return bool(
        TEMPORAL_PATTERN.search(text)
        and FACTUAL_PAYLOAD_PATTERNS.search(text)
    )


def contains_analysis_cues(text: str) -> bool:
    """Return True if *text* contains interpretive / analysis language."""
    lowered = text.lower()
    return any(cue in lowered for cue in ANALYSIS_ROLE_CUES)


def contains_material_analysis(text: str) -> bool:
    """Return True if *text* contains material analysis markers."""
    lowered = text.lower()
    return any(marker in lowered for marker in MATERIAL_ANALYSIS_MARKERS)


def contains_factual_payload(claim: ClaimInfo) -> bool:
    """Return True if *claim* contains factual content worth verifying."""
    text = claim.claim_text.strip()
    return bool(
        (claim.claim_type == "numeric" and contains_metric_payload(text))
        or claim_has_citation_keys(claim)
        or contains_numeric_or_date_payload(text)
        or FACTUAL_PAYLOAD_PATTERNS.search(text)
    )


def extract_factual_core(text: str) -> str | None:
    """Extract the factual core from a claim that mixes fact + analysis."""
    stripped = text.strip()
    stripped = re.sub(LEADING_CONCESSIVE_PATTERN, "", stripped).strip()
    lowered = stripped.lower()

    trimmed = re.sub(ANALYSIS_TAIL_PATTERN, "", stripped).rstrip(" ,;:")
    if trimmed and trimmed != stripped:
        return trimmed

    for marker in ANALYSIS_SPLIT_MARKERS:
        index = lowered.find(marker)
        if index <= 0:
            continue
        core = stripped[:index].rstrip(" ,;:")
        if core:
            core = re.sub(r"^(?:while|although|though)\s+", "", core, flags=re.IGNORECASE)
            return core
    return None


def classify_claim_role(claim: ClaimInfo) -> str:
    """Classify a claim as fact, analysis, or free after generation."""
    text = claim.claim_text.strip()
    explicit_role = claim.claim_role or ClaimRole.FACT.value

    if explicit_role == ClaimRole.FREE.value:
        if looks_structural(text) and not FACTUAL_PAYLOAD_PATTERNS.search(text):
            return ClaimRole.FREE.value
        if (
            not claim_has_citation_keys(claim)
            and not FACTUAL_PAYLOAD_PATTERNS.search(text)
            and not contains_numeric_or_date_payload(text)
        ):
            return ClaimRole.FREE.value
        if claim.claim_type == "numeric" and contains_metric_payload(text):
            return ClaimRole.FACT.value
        if contains_analysis_cues(text):
            return ClaimRole.ANALYSIS.value
        return ClaimRole.FACT.value

    if explicit_role == ClaimRole.ANALYSIS.value:
        if not contains_analysis_cues(text) and contains_factual_payload(claim):
            return ClaimRole.FACT.value
        return ClaimRole.ANALYSIS.value

    if looks_structural(text) and not contains_factual_payload(claim):
        return ClaimRole.FREE.value

    if contains_analysis_cues(text):
        if claim.claim_type == "numeric" and extract_factual_core(text):
            return ClaimRole.FACT.value
        return ClaimRole.ANALYSIS.value

    if claim.claim_type == "numeric":
        if not contains_metric_payload(text) and looks_structural(text):
            return ClaimRole.FREE.value
        return ClaimRole.FACT.value

    return ClaimRole.FACT.value
