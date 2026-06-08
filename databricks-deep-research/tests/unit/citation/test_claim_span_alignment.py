"""Span-alignment guard for Stage-8 claim modifications.

Regression for the scaffold-output corruption where a cited claim was spliced
into the middle of a word — ``"...total addressable mark" + "Return on equity
... [46]." + "et targeted..."`` — caused by a drifted ``position_start``/
``position_end`` being trusted blindly by ``_replace_claim_span`` / ``_remove_claim``.
"""
from __future__ import annotations

from databricks_deep_research.citation.pipeline import (
    _remove_claim,
    _replace_claim_span,
)
from databricks_deep_research.citation.types import ClaimInfo


def _claim(text: str, start: int, end: int) -> ClaimInfo:
    return ClaimInfo(
        claim_text=text, claim_type="general", position_start=start, position_end=end
    )


def test_replace_with_aligned_offsets_behaves_normally() -> None:
    content = "Alpha sentence. Beta claim here. Gamma sentence."
    start = content.index("Beta claim here.")
    claim = _claim("Beta claim here.", start, start + len("Beta claim here."))
    out = _replace_claim_span(content, claim, "Beta claim revised [1].")
    assert out == "Alpha sentence. Beta claim revised [1]. Gamma sentence."


def test_replace_with_drifted_offsets_relocates_not_mid_word() -> None:
    content = "A total addressable market grows fast. Return on equity fell sharply."
    bad_start = content.index("market") + 4  # mid-word inside 'market'
    bad_end = bad_start + 6
    claim = _claim("Return on equity fell sharply.", bad_start, bad_end)
    out = _replace_claim_span(content, claim, "Return on equity declined [1].")
    assert "market grows fast" in out  # 'market' stays intact — no mid-word splice
    assert "markReturn" not in out
    assert "Return on equity declined [1]." in out  # relocated to the real sentence
    assert "Return on equity fell sharply." not in out


def test_replace_skips_when_claim_text_absent() -> None:
    content = "Some unrelated prose without the claim."
    claim = _claim("This sentence is nowhere in content.", 3, 9)
    out = _replace_claim_span(content, claim, "REPLACEMENT")
    assert out == content  # unfindable + drifted -> skip, never corrupt


def test_remove_with_drifted_offsets_does_not_corrupt() -> None:
    content = "A total addressable market grows fast. Return on equity fell sharply."
    bad_start = content.index("market") + 4
    bad_end = bad_start + 6
    claim = _claim("Return on equity fell sharply.", bad_start, bad_end)
    out = _remove_claim(content, claim, context=None)
    assert "market grows fast" in out
    assert "Return on equity fell sharply." not in out  # removed via relocation
