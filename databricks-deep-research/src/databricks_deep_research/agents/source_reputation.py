"""Source-quality reputation scorer.

A stateless, dependency-free helper that turns a per-agent pair of
wildcard pattern lists (preferred / deprecated) into a signed delta to
apply to a candidate source's admission score. Framework code stays
domain-agnostic — the scorer holds no hardcoded list of "good" or "bad"
domains. All data flows in from the caller (typically constructed at the
boundary from a ``DomainFilterConfig`` populated by the designer UI or
chat).

Used by ``agents.source_aware._score_source_relevance`` to apply a soft
ranking adjustment on top of keyword + enterprise-vector scoring. See
``DomainFilterConfig`` in ``deep_research.core.app_config`` for the
user-editable shape, and the per-agent designer UI in
``frontend/src/components/agents/DomainFilterSection.tsx`` for how users
populate it.

Example
-------
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov", "investors.*"],
        deprecated_patterns=["ainvest.com", "*.spam"],
    )
    adj = scorer.score("https://www.cdc.gov/foo")
    # → ReputationAdjustment(delta=+2, matched_pattern="preferred:*.gov", reason="...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Scoring magnitudes — part of the *scoring grammar* (boost vs penalty),
# not domain-specific data. They live as module constants because the
# only sensible knob is "should we tune the magnitudes once across all
# users". A future request for per-pattern weights would change the
# constructor signature, not these defaults.
PREFERRED_DELTA: int = 2
DEPRECATED_DELTA: int = -2
IR_SUBDOMAIN_DELTA: int = 1


@dataclass(frozen=True)
class ReputationAdjustment:
    """The result of scoring one source for reputation.

    Attributes
    ----------
    delta : int
        Signed adjustment to add to the source's admission score. Positive
        values push the source up the ranking; negative values push down.
        Zero means no patterns matched (and no IR-subdomain bonus applied).
    matched_pattern : str | None
        Human-readable identifier of the first matched pattern, with a
        prefix indicating which list it came from (``preferred:``,
        ``deprecated:``, ``ir_subdomain:``). ``None`` when no match.
    reason : str
        Comma-separated string describing every match contribution; ready
        to append to the existing ``ADMISSION_SOURCE_SCORE`` log line.
    """

    delta: int
    matched_pattern: str | None
    reason: str


def _match_one(host: str, patterns: list[str]) -> str | None:
    """Return the first pattern in ``patterns`` that matches ``host``, else None.

    Uses the same wildcard semantics as the existing app-side
    ``DomainFilter.match_domain_pattern`` (suffix ``*.tld``, prefix
    ``name.*``, exact-match, and fallback ``fnmatch``). We inline the
    matcher here rather than importing from the app layer so the framework
    has no app dependency — the rules are simple and stable.
    """
    if not host:
        return None
    for raw in patterns:
        pattern = raw.lower().strip()
        if not pattern:
            continue
        if host == pattern:
            return pattern
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if host == suffix or host.endswith("." + suffix):
                return pattern
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if host == prefix or host.startswith(prefix + "."):
                return pattern
        # General fnmatch is the fallback — matches things like "*.example.*"
        import fnmatch
        if fnmatch.fnmatch(host, pattern):
            return pattern
    return None


class SourceReputationScorer:
    """Wraps preferred/deprecated wildcard patterns and produces a signed delta.

    Construction is cheap (lower-cased once); call :meth:`score` per source.
    Both lists may be empty: with empty lists and ``ir_subdomain_boost=False``
    the scorer is a no-op and ``is_active`` returns False.

    Parameters
    ----------
    preferred_patterns : list[str]
        Wildcard patterns whose URLs receive a positive delta.
    deprecated_patterns : list[str]
        Wildcard patterns whose URLs receive a negative delta.
    ir_subdomain_boost : bool, optional
        When True (default), URLs whose host starts with ``investors.`` or
        ``ir.`` receive a small additional bonus. The heuristic is
        domain-agnostic (any company's IR pages are higher-signal than the
        marketing root). Set False when the caller wants strict pattern-only
        behaviour.
    """

    def __init__(
        self,
        *,
        preferred_patterns: list[str],
        deprecated_patterns: list[str],
        ir_subdomain_boost: bool = True,
    ) -> None:
        self._preferred: list[str] = [p.lower().strip() for p in preferred_patterns if p]
        self._deprecated: list[str] = [p.lower().strip() for p in deprecated_patterns if p]
        self._ir_subdomain_boost: bool = ir_subdomain_boost

    @property
    def is_active(self) -> bool:
        """True if at least one signal source (lists or IR bonus) is configured."""
        return bool(self._preferred) or bool(self._deprecated) or self._ir_subdomain_boost

    def score(self, url: str) -> ReputationAdjustment:
        """Return the signed reputation delta + matched-pattern reason for ``url``.

        Algorithm (applied additively in this order):
          1. Match against deprecated patterns → penalty (-2 default).
          2. Match against preferred patterns → boost (+2 default).
          3. If host starts with ``investors.`` or ``ir.`` → IR bonus (+1).

        At most one match per list contributes — duplicate patterns within
        a list do not stack.
        """
        if not url:
            return ReputationAdjustment(0, None, "")
        try:
            host = (urlparse(url).hostname or "").lower()
        except Exception:  # noqa: BLE001 — defensive against malformed URLs
            return ReputationAdjustment(0, None, "")
        if not host:
            return ReputationAdjustment(0, None, "")
        host = host.removeprefix("www.")

        delta: int = 0
        matched_parts: list[str] = []
        first_match: str | None = None

        deprecated_hit = _match_one(host, self._deprecated)
        if deprecated_hit is not None:
            delta += DEPRECATED_DELTA
            label = f"deprecated:{deprecated_hit}"
            matched_parts.append(label)
            first_match = first_match or label

        preferred_hit = _match_one(host, self._preferred)
        if preferred_hit is not None:
            delta += PREFERRED_DELTA
            label = f"preferred:{preferred_hit}"
            matched_parts.append(label)
            first_match = first_match or label

        if self._ir_subdomain_boost and (
            host.startswith("investors.") or host.startswith("ir.")
        ):
            delta += IR_SUBDOMAIN_DELTA
            label = f"ir_subdomain:{host}"
            matched_parts.append(label)
            first_match = first_match or label

        # Emit a conflict signal when a pattern matched in BOTH lists; the
        # net delta is zero (or small) so the source effectively keeps its
        # baseline score, but operators may want to know their config has a
        # contradictory pattern.
        if deprecated_hit and preferred_hit:
            logger.info(
                "REPUTATION_CONFLICT host=%s preferred=%s deprecated=%s",
                host, preferred_hit, deprecated_hit,
            )

        reason = ", ".join(matched_parts)
        if delta != 0:
            logger.debug(
                "REPUTATION_DELTA url=%s delta=%+d reason=%s",
                url[:120], delta, reason,
            )
        return ReputationAdjustment(delta=delta, matched_pattern=first_match, reason=reason)


__all__ = [
    "PREFERRED_DELTA",
    "DEPRECATED_DELTA",
    "IR_SUBDOMAIN_DELTA",
    "ReputationAdjustment",
    "SourceReputationScorer",
]
