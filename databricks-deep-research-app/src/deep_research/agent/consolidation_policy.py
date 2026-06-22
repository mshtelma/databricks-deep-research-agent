"""Trust policy: which verified claims may become durable chat knowledge.

The citation pipeline persists ``verification_data`` per research session.
Only claims it judged ``supported`` or ``partial`` (and did not abstain on) are
trustworthy enough to carry forward as durable findings — persisting an
``unsupported``/``contradicted``/``abstained`` claim would let a refuted
statement resurface as "knowledge" in a later turn. This module is the single
gate enforcing that.

Pure functions only — no I/O.
"""

from __future__ import annotations

from typing import Any

# Verdicts trustworthy enough to persist, mapped to the confidence tier they
# carry into memory. Anything not in this map is dropped.
_VERDICT_TO_CONFIDENCE: dict[str, str] = {
    "supported": "high",
    "partial": "medium",
}


def extract_consolidatable_claims(
    verification_data: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Return the claims worth persisting as durable findings.

    Args:
        verification_data: the persisted dict
            ``{"claims": [claim_dict, ...], "summary": {...}}`` (or ``None``).

    Returns:
        A list of ``{"claim_text", "confidence"}`` for ``supported``/``partial``
        non-abstained claims with non-empty text. All other claims are dropped.
    """
    if not verification_data:
        return []
    claims = verification_data.get("claims")
    if not isinstance(claims, list):
        return []

    result: list[dict[str, str]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if claim.get("abstained"):
            continue
        verdict = str(claim.get("verification_verdict") or "")
        confidence = _VERDICT_TO_CONFIDENCE.get(verdict)
        if confidence is None:
            continue
        text = str(claim.get("claim_text") or "").strip()
        if not text:
            continue
        result.append({"claim_text": text, "confidence": confidence})
    return result
