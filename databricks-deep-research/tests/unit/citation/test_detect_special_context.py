"""Regression for Defect D: row-start table claims must be detected as
``context == "table"`` so Stage-8 softening appends ``[unverified]`` instead
of prepending a hedge before the leading ``|`` (which corrupts the row).
"""
from __future__ import annotations

from databricks_deep_research.citation.pipeline import (
    _build_softened_fact_text,
    _detect_special_context,
)

_TABLE = (
    "Intro paragraph.\n\n"
    "| Total Type | Column | Value |\n"
    "|---|---|---|\n"
    "| Total Fiscal Year | col_15 | 50,337 |\n"
)


def test_row_start_position_detected_as_table() -> None:
    row_start = _TABLE.index("| Total Fiscal Year")
    assert _detect_special_context(_TABLE, row_start) == "table"


def test_midrow_position_still_table() -> None:
    midrow = _TABLE.index("col_15")
    assert _detect_special_context(_TABLE, midrow) == "table"


def test_prose_position_is_not_table() -> None:
    content = "This is an ordinary sentence with no pipes at all.\n"
    assert _detect_special_context(content, 10) is None


def test_soften_table_row_claim_does_not_prepend_prose() -> None:
    row_start = _TABLE.index("| Total Fiscal Year")
    ctx = _detect_special_context(_TABLE, row_start)
    claim_text = "| Total Fiscal Year | col_15 | 50,337 |"
    softened = _build_softened_fact_text(claim_text, ctx)
    assert not softened.lstrip().startswith(
        (
            "Some sources indicate",
            "It has been suggested",
            "According to available",
            "Reportedly",
        )
    )
    assert softened.startswith("|")  # row structure preserved
