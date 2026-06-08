"""Framework-side verification extraction — relocated from the app.

Centralises the logic that converts a :class:`WorkflowState` (with
framework-native ``claims``, ``verification_summary`` artifacts) plus a
sources list into a structured summary suitable for downstream consumers.

Two entry points:

- :func:`extract_verification` reads the framework-native ``claims`` and
  ``verification_summary`` keys from a :class:`WorkflowState` (set by the
  7-stage citation pipeline). Preferred when the workflow uses the
  framework's verification stages.
- :func:`extract_verification_from_report` parses ``[N]`` numeric citation
  markers out of a synthesizer's final report. Used as a fallback when
  framework-native artifacts are absent.

The original implementations lived in the app's ``framework_orchestrator``
module (``_extract_verification_from_framework_state`` and
``_extract_verification_from_report``). Relocating here lets external
Python API consumers receive structured verification metadata without
depending on the app package.

Contract for callers:

- Both functions return a :class:`VerificationSummary` with
  ``.claims`` and ``.summary`` attributes.
- ``.claims`` is a list of :class:`Claim` instances.
- ``.summary`` is a :class:`SummaryInfo` or ``None`` (if no verification
  artifacts could be derived).
- The app's backward-compat shim converts ``Claim`` / ``SummaryInfo`` into
  its own ``ClaimInfo`` / ``VerificationSummaryInfo`` dataclasses by
  field-by-field copy — the field names match exactly.

The 7-stage citation pipeline at ``citation/pipeline.py`` is NOT touched by
this module; we only consume its output artifacts.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_NUMERIC_CITATION_RE = re.compile(r"\[(\d+)\]")


class Evidence(BaseModel):
    """Pre-selected evidence span — mirrors the app's ``EvidenceInfo``."""

    model_config = ConfigDict(extra="forbid")

    source_url: str
    quote_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None
    has_numeric_content: bool = False


class Claim(BaseModel):
    """Atomic claim — mirrors the app's ``ClaimInfo``."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str
    claim_type: str = "general"
    position_start: int = 0
    position_end: int = 0
    evidence: Evidence | None = None
    confidence_level: str | None = None
    verification_verdict: str | None = None
    verification_reasoning: str | None = None
    abstained: bool = False
    citation_key: str | None = None
    citation_keys: list[str] | None = None
    from_free_block: bool = False


class SummaryInfo(BaseModel):
    """Verification summary stats — mirrors the app's ``VerificationSummaryInfo``."""

    model_config = ConfigDict(extra="forbid")

    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    abstained_count: int = 0
    unsupported_rate: float = 0.0
    contradicted_rate: float = 0.0
    warning: bool = False
    citation_corrections: int = 0
    # Stage 7 metrics
    claim_revisions: int = 0
    atomic_facts_total: int = 0
    atomic_facts_verified: int = 0
    atomic_facts_softened: int = 0
    claims_fully_verified: int = 0
    claims_partially_softened: int = 0
    claims_fully_softened: int = 0
    external_searches: int = 0
    new_sources_added: int = 0


class VerificationSummary(BaseModel):
    """Top-level wrapper exposing ``claims`` and ``summary`` accessors.

    Provides the structured representation a Python API user (or the app's
    shim) can consume directly. Field names match the
    ``VerificationSummaryEvent`` payload (``events/types.py:510-521``) where
    overlapping; additional fields capture the underlying claim list.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(default_factory=list)
    summary: SummaryInfo | None = None

    # -- Convenience accessors mirroring the VerificationSummaryEvent shape ----

    @property
    def total_claims(self) -> int:
        return self.summary.total_claims if self.summary else len(self.claims)

    @property
    def verified_claims(self) -> int:
        return self.summary.supported_count if self.summary else 0

    @property
    def corrected_citations(self) -> int:
        return self.summary.citation_corrections if self.summary else 0

    @property
    def removed_claims(self) -> int:
        return self.summary.contradicted_count if self.summary else 0

    @property
    def softened_claims(self) -> int:
        return self.summary.unsupported_count if self.summary else 0

    @property
    def overall_confidence(self) -> float:
        if self.summary is None or self.summary.total_claims == 0:
            return 0.0
        return self.summary.supported_count / self.summary.total_claims


def _state_get(wf_state: Any, key: str) -> Any:
    """Read ``key`` from a :class:`WorkflowState` if it exposes ``.get(...)``."""
    if wf_state is None:
        return None
    getter = getattr(wf_state, "get", None)
    if getter is None:
        return None
    try:
        return getter(key)
    except Exception:
        return None


def _build_evidence_from_dict(evidence_raw: dict[str, Any]) -> Evidence | None:
    if not evidence_raw.get("source_url"):
        return None
    return Evidence(
        source_url=str(evidence_raw.get("source_url", "") or ""),
        quote_text=str(evidence_raw.get("quote_text", "") or ""),
        start_offset=evidence_raw.get("start_offset"),
        end_offset=evidence_raw.get("end_offset"),
        section_heading=evidence_raw.get("section_heading"),
        relevance_score=evidence_raw.get("relevance_score"),
        has_numeric_content=bool(evidence_raw.get("has_numeric_content", False)),
    )


def _evidence_from_source(source: Any) -> Evidence | None:
    """Build an Evidence pointing at a source object's url/snippet."""
    if isinstance(source, dict):
        url = str(source.get("url", "") or "")
        snippet = str(source.get("snippet", "") or "")
    else:
        url = str(getattr(source, "url", "") or "")
        snippet = str(getattr(source, "snippet", "") or "")
    if not url:
        return None
    return Evidence(source_url=url, quote_text=snippet)


def extract_verification(
    wf_state: Any,
    sources: list[Any],
) -> VerificationSummary:
    """Extract structured verification data from a framework :class:`WorkflowState`.

    Reads framework-native ``claims`` and ``verification_summary`` artifacts
    written by the 7-stage citation pipeline. When neither key is present,
    returns an empty :class:`VerificationSummary`.

    Args:
        wf_state: The :class:`WorkflowState` (or any object exposing ``.get``).
        sources: Ordered list of source objects (dicts or AppSourceInfo-like)
            used to fall back when a claim references a source by index but
            does not carry a full ``evidence`` dict.

    Returns:
        :class:`VerificationSummary` with ``.claims`` and ``.summary``
        populated as best-effort.
    """
    raw_claims = _state_get(wf_state, "claims")
    raw_summary = _state_get(wf_state, "verification_summary")

    if not raw_claims and not raw_summary:
        return VerificationSummary()

    claims: list[Claim] = []
    if isinstance(raw_claims, list):
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                continue

            evidence: Evidence | None = None
            evidence_raw = raw_claim.get("evidence")
            if isinstance(evidence_raw, dict):
                evidence = _build_evidence_from_dict(evidence_raw)
            if evidence is None:
                citation_keys = raw_claim.get("citation_keys") or []
                citation_key = (
                    raw_claim.get("citation_key")
                    or (citation_keys[0] if citation_keys else None)
                )
                if isinstance(citation_key, str) and citation_key.isdigit():
                    source_index = int(citation_key)
                    if 0 <= source_index < len(sources):
                        evidence = _evidence_from_source(sources[source_index])

            claims.append(
                Claim(
                    claim_text=str(raw_claim.get("claim_text", "") or ""),
                    claim_type=str(
                        raw_claim.get("claim_type", "general") or "general"
                    ),
                    position_start=int(raw_claim.get("position_start", 0) or 0),
                    position_end=int(raw_claim.get("position_end", 0) or 0),
                    evidence=evidence,
                    confidence_level=raw_claim.get("confidence_level"),
                    verification_verdict=raw_claim.get("verification_verdict"),
                    verification_reasoning=raw_claim.get("verification_reasoning"),
                    abstained=bool(raw_claim.get("abstained", False)),
                    citation_key=raw_claim.get("citation_key"),
                    citation_keys=raw_claim.get("citation_keys"),
                    from_free_block=bool(raw_claim.get("from_free_block", False)),
                )
            )

    summary: SummaryInfo | None = None
    if isinstance(raw_summary, dict):
        summary = SummaryInfo(
            total_claims=int(raw_summary.get("total_claims", 0) or 0),
            supported_count=int(
                raw_summary.get(
                    "verified_claims",
                    raw_summary.get("supported_count", 0),
                )
                or 0
            ),
            partial_count=int(raw_summary.get("partial_count", 0) or 0),
            unsupported_count=int(
                raw_summary.get(
                    "softened_claims",
                    raw_summary.get("unsupported_count", 0),
                )
                or 0
            ),
            contradicted_count=int(
                raw_summary.get(
                    "removed_claims",
                    raw_summary.get("contradicted_count", 0),
                )
                or 0
            ),
            abstained_count=int(raw_summary.get("abstained_count", 0) or 0),
            unsupported_rate=float(raw_summary.get("unsupported_rate", 0.0) or 0.0),
            contradicted_rate=float(raw_summary.get("contradicted_rate", 0.0) or 0.0),
            warning=bool(raw_summary.get("warning", False)),
            citation_corrections=int(
                raw_summary.get(
                    "corrected_citations",
                    raw_summary.get("citation_corrections", 0),
                )
                or 0
            ),
            claim_revisions=int(raw_summary.get("claim_revisions", 0) or 0),
            atomic_facts_total=int(raw_summary.get("atomic_facts_total", 0) or 0),
            atomic_facts_verified=int(
                raw_summary.get("atomic_facts_verified", 0) or 0
            ),
            atomic_facts_softened=int(
                raw_summary.get("atomic_facts_softened", 0) or 0
            ),
            claims_fully_verified=int(
                raw_summary.get("claims_fully_verified", 0) or 0
            ),
            claims_partially_softened=int(
                raw_summary.get("claims_partially_softened", 0) or 0
            ),
            claims_fully_softened=int(
                raw_summary.get("claims_fully_softened", 0) or 0
            ),
            external_searches=int(raw_summary.get("external_searches", 0) or 0),
            new_sources_added=int(raw_summary.get("new_sources_added", 0) or 0),
        )

    return VerificationSummary(claims=claims, summary=summary)


def extract_verification_from_report(
    final_report: str,
    sources: list[Any],
) -> VerificationSummary:
    """Parse ``[N]`` markers in a synthesizer report into structured claims.

    Used when framework-native ``claims`` artifacts are absent (e.g. the
    workflow used a custom synthesizer that did not populate
    ``wf_state["claims"]``). Splits the report into sentences, extracts
    citation markers, and binds each cited sentence to a source.

    Args:
        final_report: Markdown / text final report from a synthesizer.
        sources: Ordered list of source objects matching the indices in the
            ``[N]`` markers. Indexing convention is auto-detected (0- vs
            1-based).

    Returns:
        :class:`VerificationSummary` whose ``.claims`` are sentence-level
        atomic claims linked to source URLs.  ``.summary`` is populated only
        when at least one claim was extracted.
    """
    if not final_report or not sources:
        return VerificationSummary()

    sentences = re.split(r"(?<=[.!?])\s+|\n+", final_report)

    all_marker_indices: set[int] = set()
    for sentence in sentences:
        text = sentence.strip()
        if text:
            all_marker_indices.update(int(m) for m in _NUMERIC_CITATION_RE.findall(text))

    index_offset = 0 if 0 in all_marker_indices else 1

    logger.info(
        "FWK_VERIFICATION_INDEX_DETECT markers=%s offset=%d sources=%d",
        sorted(all_marker_indices)[:10],
        index_offset,
        len(sources),
    )

    claims: list[Claim] = []
    position = 0

    for sentence in sentences:
        text = sentence.strip()
        if not text:
            position += 1
            continue

        markers = _NUMERIC_CITATION_RE.findall(text)
        if not markers:
            position += len(text) + 1
            continue

        claim_text = _NUMERIC_CITATION_RE.sub("", text).strip()
        if len(claim_text) < 10:
            position += len(text) + 1
            continue

        cited_indices = sorted({int(m) for m in markers})
        citation_keys: list[str] = [str(idx) for idx in cited_indices]

        evidence: Evidence | None = None
        for idx in cited_indices:
            pool_idx = idx - index_offset
            if 0 <= pool_idx < len(sources):
                source = sources[pool_idx]
                if isinstance(source, dict):
                    source_url = source.get("url", "") or ""
                    source_snippet = source.get("snippet", "") or ""
                else:
                    source_url = getattr(source, "url", "") or ""
                    source_snippet = getattr(source, "snippet", "") or ""
                if evidence is None and source_url:
                    evidence = Evidence(
                        source_url=source_url,
                        quote_text=source_snippet[:300] if source_snippet else "",
                    )

        has_numbers = bool(re.search(r"\$[\d,.]+|\d+\.\d+%|\d{2,}", claim_text))

        claims.append(
            Claim(
                claim_text=claim_text,
                claim_type="numeric" if has_numbers else "general",
                position_start=position,
                position_end=position + len(text),
                evidence=evidence,
                confidence_level="high",
                verification_verdict="supported",
                verification_reasoning="Cited by synthesizer with source reference",
                abstained=False,
                citation_key=citation_keys[0] if citation_keys else None,
                citation_keys=citation_keys if len(citation_keys) > 1 else None,
            )
        )
        position += len(text) + 1

    if not claims:
        return VerificationSummary()

    summary = SummaryInfo(
        total_claims=len(claims),
        supported_count=len(claims),
        partial_count=0,
        unsupported_count=0,
        contradicted_count=0,
        abstained_count=0,
        unsupported_rate=0.0,
        contradicted_rate=0.0,
        warning=False,
        citation_corrections=0,
    )

    logger.info(
        "FWK_VERIFICATION_EXTRACTED claims=%d sources_cited=%d",
        len(claims),
        len({c.evidence.source_url for c in claims if c.evidence}),
    )

    return VerificationSummary(claims=claims, summary=summary)


__all__ = [
    "Claim",
    "Evidence",
    "SummaryInfo",
    "VerificationSummary",
    "extract_verification",
    "extract_verification_from_report",
]
