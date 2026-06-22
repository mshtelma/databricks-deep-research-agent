"""Unit tests for the ``HIGH_ABSTAIN_RATE`` observability warning floor.

The warning is a batch-level signal: a high *ratio* of abstentions only means
something across a batch large enough for the ratio to be meaningful. Without a
minimum-sample floor, a single-claim batch that legitimately abstains (e.g. the
capability validator, which verifies one synthetic claim per model-surface)
trips it at ``1/1 = 100%`` — a false alarm. ``_warn_if_high_abstain`` suppresses
the warning below ``_HIGH_ABSTAIN_MIN_CLAIMS`` and keeps it for real batches.
"""

import logging

from databricks_deep_research.citation.pipeline import (
    _HIGH_ABSTAIN_MIN_CLAIMS,
    _warn_if_high_abstain,
)
from databricks_deep_research.citation.types import ClaimInfo, ClaimRole

_LOGGER_NAME = "databricks_deep_research.citation.pipeline"


def _claim(*, abstained: bool, idx: int = 0) -> ClaimInfo:
    text = "Some assertion text."
    return ClaimInfo(
        claim_text=text,
        claim_type="general",
        position_start=idx,
        position_end=idx + len(text),
        abstained=abstained,
        claim_role=ClaimRole.FACT.value,
    )


def _warned(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "HIGH_ABSTAIN_RATE" in r.getMessage()]


def test_single_claim_abstain_does_not_warn(caplog) -> None:
    """The capability-validator case: 1 claim, abstained → 1/1 = 100% but silent."""
    claims = [_claim(abstained=True)]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_if_high_abstain(claims)
    assert _warned(caplog) == []


def test_below_floor_all_abstain_does_not_warn(caplog) -> None:
    """A sub-floor batch (all abstaining) is still too small to be a signal."""
    claims = [_claim(abstained=True, idx=i) for i in range(_HIGH_ABSTAIN_MIN_CLAIMS - 1)]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_if_high_abstain(claims)
    assert _warned(caplog) == []


def test_at_floor_high_abstain_warns(caplog) -> None:
    """A batch at the floor with a >10% abstain rate warns, with correct counts."""
    n = _HIGH_ABSTAIN_MIN_CLAIMS
    claims = [_claim(abstained=True, idx=i) for i in range(n)]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_if_high_abstain(claims)
    messages = _warned(caplog)
    assert len(messages) == 1
    assert f"abstained={n}" in messages[0]
    assert f"total={n}" in messages[0]


def test_at_floor_low_abstain_does_not_warn(caplog) -> None:
    """At/above the floor but at-or-below the 10% threshold → no warning."""
    # 1 of 10 = 10.0%, which is not strictly greater than the 0.10 threshold.
    claims = [_claim(abstained=(i == 0), idx=i) for i in range(10)]
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_if_high_abstain(claims)
    assert _warned(caplog) == []


def test_empty_batch_does_not_warn(caplog) -> None:
    """No claims → no division, no warning."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _warn_if_high_abstain([])
    assert _warned(caplog) == []
