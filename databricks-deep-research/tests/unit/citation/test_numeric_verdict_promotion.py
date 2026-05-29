"""Defect E / Part 1: deterministic numeric corroboration must override a
non-supported NLI verdict so grounded numbers are KEPT, not hedged."""
from __future__ import annotations

import pytest

from databricks_deep_research.citation.pipeline import (
    _NUMERIC_PROMOTION_MIN_CONFIDENCE,
    _should_promote_numeric_verdict,
)


def _match(conf: float) -> dict:
    return {"overall_match": True, "confidence": conf}


@pytest.mark.parametrize("verdict", ["partial", "unsupported"])
def test_promotes_nonsupported_on_strong_numeric_match(verdict: str) -> None:
    assert _should_promote_numeric_verdict(verdict, _match(0.95), enabled=True) is True


def test_never_promotes_contradicted() -> None:
    assert _should_promote_numeric_verdict("contradicted", _match(1.0), enabled=True) is False


def test_supported_is_noop() -> None:
    assert _should_promote_numeric_verdict("supported", _match(1.0), enabled=True) is False


def test_requires_overall_match() -> None:
    nr = {"overall_match": False, "confidence": 1.0}
    assert _should_promote_numeric_verdict("partial", nr, enabled=True) is False


def test_respects_min_confidence() -> None:
    assert _NUMERIC_PROMOTION_MIN_CONFIDENCE == 0.9
    assert _should_promote_numeric_verdict("partial", _match(0.67), enabled=True) is False


def test_flag_off_disables_promotion() -> None:
    assert _should_promote_numeric_verdict("partial", _match(0.95), enabled=False) is False


def test_empty_numeric_result_is_safe() -> None:
    assert _should_promote_numeric_verdict("partial", {}, enabled=True) is False
    assert _should_promote_numeric_verdict(None, {}, enabled=True) is False
