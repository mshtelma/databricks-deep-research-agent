"""Claim role classification -- fact, analysis, or free heuristics.

Extracted from pipeline.py to reduce god-object complexity.
Pure functions with no I/O or LLM calls.

PR3-E R2.2 also adds ``classify_negative_existence`` here — an
async function that runs ONLY on non-fully-supported claims to flag
negative-existence assertions for disposition force-REMOVE.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from databricks_deep_research.citation.types import ClaimInfo, ClaimRole
from databricks_deep_research.citation.utils import NUMERIC_PATTERN, TEMPORAL_PATTERN

logger = logging.getLogger(__name__)


# PR3-E R2.2: the verdict set on which the is_negative_existence classifier
# runs. The plan calls for {abstained, unsupported, contradicted}; the R0
# ADR finding (hypothesis a — contradicted verdicts get normalized to
# partial/unsupported/abstained before reaching Stage 8) broadened this to
# include "partial" so the classifier catches normalized contradicted
# claims. Keep ``supported`` out — those are pre-validated true claims.
_NON_FULLY_SUPPORTED_VERDICTS: frozenset[str] = frozenset(
    {"abstained", "unsupported", "contradicted", "partial"}
)


_NEGATIVE_EXISTENCE_PROMPT = (
    "Does this claim assert that a specific fact, entity, period, or value "
    "is NOT present, NOT available, or DOES NOT exist in the source corpus?\n\n"
    "Claim: {claim_text}\n\n"
    "Respond with a single JSON object: "
    '{{"is_negative_existence": <true|false>, "reasoning": "<one sentence>"}}'
)


def _negative_existence_eligible(claim: ClaimInfo) -> bool:
    """Return True if the claim is in scope for the classifier.

    PR3-E R2.2: only verdicts that are not fully supported get classified
    (the classifier never fires on ``supported`` claims).
    """
    if claim.abstained:
        return True
    verdict = (claim.verification_verdict or "").lower()
    return verdict in _NON_FULLY_SUPPORTED_VERDICTS


async def classify_negative_existence(
    claim: ClaimInfo,
    llm_client: Any,
    *,
    model_tier: str = "fast",
) -> tuple[bool, str | None]:
    """Classify whether *claim* is a negative-existence assertion.

    Args:
        claim: The claim to classify. Returns (False, None) immediately
            when the claim's verdict is not in the eligible set.
        llm_client: An object exposing the FrameworkLLMClient API
            ``await complete(messages, tier, ...)`` OR — for tests — a
            duck-typed object with a ``complete`` method that returns a
            dict-shaped JSON response.
        model_tier: The tier to call. Defaults to ``"fast"`` per plan
            latency budget.

    Returns:
        ``(is_negative_existence, reasoning_or_None)``. Defaults to
        ``(False, None)`` on any classifier error so the disposition
        pipeline's behaviour is preserved when the model call fails.
    """
    if not _negative_existence_eligible(claim):
        return False, None
    prompt = _NEGATIVE_EXISTENCE_PROMPT.format(claim_text=claim.claim_text)
    try:
        raw = await _call_classifier_llm(llm_client, prompt, model_tier)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(
            "NEGATIVE_EXISTENCE_CLASSIFIER_ERROR claim_head=%r error=%s",
            claim.claim_text[:60],
            exc,
        )
        return False, None
    return _coerce_negative_existence_response(raw)


async def _call_classifier_llm(
    llm_client: Any, prompt: str, model_tier: str
) -> Any:
    """Call the LLM via the FrameworkLLMClient API OR the duck-typed test stub.

    The framework's ``FrameworkLLMClient.complete`` takes
    ``(messages, tier, ...)``; test stubs may take
    ``(prompt=..., model_tier=..., response_format=...)``. This helper
    tries the framework signature first and falls back to the stub
    signature on TypeError. Returns the raw response (either an
    ``LLMResponse`` whose ``content`` field is parsed, or the stub's
    dict directly).
    """
    try:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        resp = await llm_client.complete(messages, tier=model_tier)
        # Framework response: LLMResponse(content=str, ...)
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            return content
        return resp
    except TypeError:
        return await llm_client.complete(
            prompt=prompt,
            model_tier=model_tier,
            response_format="json",
        )


def _coerce_negative_existence_response(raw: Any) -> tuple[bool, str | None]:
    """Best-effort parse of the classifier's response into (bool, reasoning)."""
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return False, None
        if not isinstance(payload, dict):
            return False, None
    else:
        return False, None
    flag = payload.get("is_negative_existence")
    if not isinstance(flag, bool):
        return False, None
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = None
    return flag, reasoning

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
