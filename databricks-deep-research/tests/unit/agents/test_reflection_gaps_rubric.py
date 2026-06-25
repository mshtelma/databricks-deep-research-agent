"""Unit tests for the additive ReflectionOutput fields (feature 3.5).

Covers:
  * Round-trip with AND without the new ``knowledge_gaps`` / ``rubric`` fields
    (back-compat: an omitted payload validates exactly as before).
  * Junk rubric / gaps coerce gracefully and NEVER raise (mirrors the
    existing ``_normalize_*`` validators in ``output_models``).
  * The ``max_length=10`` cap and 1-10 clamps behave as declared.

All pure / mocked — no LLM, no I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from databricks_deep_research.agents.output_models import (
    ReflectionOutput,
    ReflectionRubric,
)

# ---------------------------------------------------------------------------
# Back-compat: omitting the new fields == today's behavior
# ---------------------------------------------------------------------------


def test_reflection_output_without_new_fields_roundtrips() -> None:
    """A pre-3.5 reflection payload (no gaps/rubric) validates with defaults."""
    out = ReflectionOutput(decision="continue", reasoning="more to do")
    assert out.knowledge_gaps == []
    assert out.rubric is None
    # Round-trips through dump/validate unchanged.
    again = ReflectionOutput.model_validate(out.model_dump())
    assert again.knowledge_gaps == []
    assert again.rubric is None


def test_reflection_output_missing_keys_treated_as_empty() -> None:
    """JSON missing the gaps/rubric keys entirely is accepted (defaults)."""
    out = ReflectionOutput.model_validate(
        {"decision": "complete", "reasoning": "done", "directives": []}
    )
    assert out.knowledge_gaps == []
    assert out.rubric is None


# ---------------------------------------------------------------------------
# With the new fields populated
# ---------------------------------------------------------------------------


def test_reflection_output_with_gaps_and_rubric_roundtrips() -> None:
    out = ReflectionOutput.model_validate(
        {
            "decision": "continue",
            "reasoning": "gaps remain",
            "knowledge_gaps": ["pricing not found", "no 2025 figures"],
            "rubric": {
                "completeness": 6,
                "depth": 7,
                "reliability": 8,
                "recency": 5,
                "overall": 6.5,
            },
        }
    )
    assert out.knowledge_gaps == ["pricing not found", "no 2025 figures"]
    assert isinstance(out.rubric, ReflectionRubric)
    assert out.rubric.completeness == 6
    assert out.rubric.overall == pytest.approx(6.5)
    # Survives a dump/validate cycle.
    again = ReflectionOutput.model_validate(out.model_dump())
    assert again.knowledge_gaps == out.knowledge_gaps
    assert again.rubric is not None
    assert again.rubric.depth == 7


# ---------------------------------------------------------------------------
# Junk coercion: never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "junk_gaps",
    [
        None,
        "not a list",
        123,
        {"a": "b"},
        [None, "", "  ", "real gap"],  # blanks/None dropped, real kept
    ],
)
def test_junk_knowledge_gaps_never_raises(junk_gaps: object) -> None:
    out = ReflectionOutput.model_validate(
        {"decision": "continue", "reasoning": "x", "knowledge_gaps": junk_gaps}
    )
    assert isinstance(out.knowledge_gaps, list)
    # No falsy / blank entries ever survive.
    assert all(g and g.strip() for g in out.knowledge_gaps)


@pytest.mark.parametrize("junk_rubric", [None, "string", 5, [1, 2, 3], True])
def test_junk_rubric_coerces_to_none_never_raises(junk_rubric: object) -> None:
    out = ReflectionOutput.model_validate(
        {"decision": "continue", "reasoning": "x", "rubric": junk_rubric}
    )
    assert out.rubric is None


def test_rubric_dimension_clamps_out_of_range_values() -> None:
    """Out-of-range / non-numeric dimensions clamp to [1,10] (junk -> 5)."""
    rubric = ReflectionRubric.model_validate(
        {
            "completeness": 99,  # clamps to 10
            "depth": -4,  # clamps to 1
            "reliability": "not a number",  # -> 5
            "recency": "8",  # numeric string -> 8
            "overall": 42.0,  # clamps to 10.0
        }
    )
    assert rubric.completeness == 10
    assert rubric.depth == 1
    assert rubric.reliability == 5
    assert rubric.recency == 8
    assert rubric.overall == pytest.approx(10.0)


def test_knowledge_gaps_over_cap_rejected() -> None:
    """More than 10 gaps trips the declared ``max_length`` cap."""
    with pytest.raises(ValidationError):
        ReflectionOutput.model_validate(
            {
                "decision": "continue",
                "reasoning": "x",
                "knowledge_gaps": [f"gap-{i}" for i in range(11)],
            }
        )
