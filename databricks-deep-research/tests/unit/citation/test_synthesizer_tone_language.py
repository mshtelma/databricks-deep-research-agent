"""Per-run TONE + OUTPUT-LANGUAGE knobs for synthesis (spec §4.4).

Covers:
  * The self-describing ``Tone`` enum (each value carries its own definition).
  * ``_build_reclaim_generation_instructions`` injects the tone/language clause
    when set AND the default (no knobs) path is byte-identical.
  * A REGRESSION GUARD that the hard numeric/unit citation rules in the actual
    generation prompts are NOT weakened or removed by this feature
    (project-scaffold-million-unit-gap).
  * A multi-language smoke: ``output_language`` produces a language directive.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.agents.builtins.synthesizer import (
    _build_reclaim_generation_instructions,
    _build_tone_language_clause,
)
from databricks_deep_research.agents.config import AgentNodeConfig, Tone
from databricks_deep_research.citation.claim_generator import (
    _NATURAL_GENERATION_PROMPT,
    _STRICT_GENERATION_PROMPT,
    InterleavedGenerator,
)
from databricks_deep_research.citation.types import RankedEvidence
from databricks_deep_research.llm.client import LLMResponse

# The canonical hard numeric/unit rules that MUST survive (verbatim) regardless
# of tone/language. These are the "state units / no bare numbers / no raw column
# ids" guardrails from project-scaffold-million-unit-gap. If a future edit
# rewords them, update BOTH this list and the prompt deliberately — never let the
# tone/language feature silently drop them.
_HARD_NUMERIC_UNIT_RULES = [
    "never a bare number",
    "state the value's unit of measure or currency exactly as the source expresses it",
    "internal field or column identifier",
    "NEVER synthesize numbers not in the evidence",
]


def _synth_config(
    *,
    tone: Tone | None = None,
    output_language: str | None = None,
    system_prompt: str = "",
    output_schema: dict | None = None,
) -> AgentNodeConfig:
    return AgentNodeConfig(
        subtype="synthesizer",
        system_prompt=system_prompt,
        tone=tone,
        output_language=output_language,
        output_schema=output_schema,
    )


# ---------------------------------------------------------------------------
# Tone enum — self-describing
# ---------------------------------------------------------------------------


def test_tone_values_are_self_describing() -> None:
    """Each Tone value carries its own parenthetical definition (the glossary IS
    the enum), so the directive needs no external lookup table."""
    for member in Tone:
        assert "(" in member.value and ")" in member.value, member
        assert member.directive() == member.value


def test_tone_from_name_round_trips_and_degrades() -> None:
    assert Tone.from_name("objective") is Tone.OBJECTIVE
    assert Tone.from_name("FORMAL") is Tone.FORMAL
    assert Tone.from_name("  analytical  ") is Tone.ANALYTICAL
    # Unknown / empty => None (degrade to unchanged synthesis, never raise).
    assert Tone.from_name("nonsense") is None
    assert Tone.from_name("") is None
    assert Tone.from_name(None) is None


def test_tone_names_are_lowercase_member_names() -> None:
    names = Tone.names()
    assert "objective" in names
    assert "formal" in names
    assert all(n == n.lower() for n in names)


# ---------------------------------------------------------------------------
# _build_tone_language_clause
# ---------------------------------------------------------------------------


def test_tone_language_clause_empty_when_unset() -> None:
    assert _build_tone_language_clause(_synth_config()) == ""


def test_tone_language_clause_contains_tone_directive() -> None:
    clause = _build_tone_language_clause(_synth_config(tone=Tone.ANALYTICAL))
    assert Tone.ANALYTICAL.value in clause
    # The clause explicitly refuses to relax citation/numeric/unit rules.
    assert "WITHOUT relaxing" in clause


def test_tone_language_clause_contains_language_directive() -> None:
    clause = _build_tone_language_clause(_synth_config(output_language="Spanish"))
    assert "Spanish" in clause
    # Numeric/unit invariants must be carried through the language directive too.
    assert "numeric values, units, currencies" in clause


# ---------------------------------------------------------------------------
# _build_reclaim_generation_instructions — injection + default parity
# ---------------------------------------------------------------------------


def test_default_path_is_byte_identical_without_knobs() -> None:
    """No tone, no language, no contract => empty string (unchanged today)."""
    assert _build_reclaim_generation_instructions(_synth_config()) == ""


def test_default_path_byte_identical_with_contract_only() -> None:
    """With a report contract but NO tone/language, the output must equal the
    output produced before the feature (i.e. no style clause appended)."""
    schema = {"report_contract": ["Section A", "Section B"]}
    with_knobs_absent = _build_reclaim_generation_instructions(
        _synth_config(output_schema=schema)
    )
    assert "Report Style" not in with_knobs_absent
    assert "Output Contract" in with_knobs_absent


def test_instructions_contain_tone_clause_when_set() -> None:
    out = _build_reclaim_generation_instructions(_synth_config(tone=Tone.FORMAL))
    assert Tone.FORMAL.value in out
    assert "Report Style" in out


def test_instructions_contain_language_clause_when_set() -> None:
    out = _build_reclaim_generation_instructions(
        _synth_config(output_language="Japanese")
    )
    assert "Japanese" in out


def test_tone_and_language_appended_after_designer_contract() -> None:
    """The style clause must come LAST — after the Designer contract — so it
    constrains style only and never displaces the report contract."""
    schema = {"report_contract": ["Executive Summary", "Findings"]}
    out = _build_reclaim_generation_instructions(
        _synth_config(
            tone=Tone.PERSUASIVE,
            output_language="German",
            output_schema=schema,
        )
    )
    contract_idx = out.index("Output Contract")
    style_idx = out.index("Report Style")
    assert contract_idx < style_idx, "style clause must follow the contract"
    assert Tone.PERSUASIVE.value in out
    assert "German" in out


# ---------------------------------------------------------------------------
# REGRESSION GUARD — hard numeric/unit rules must persist in the real prompts
# ---------------------------------------------------------------------------


def test_hard_numeric_unit_rules_present_in_strict_prompt() -> None:
    """The tone/language feature touches generation INSTRUCTIONS only. The hard
    numeric/unit rules live in the strict generation prompt template and MUST
    remain present and unweakened (project-scaffold-million-unit-gap)."""
    for rule in _HARD_NUMERIC_UNIT_RULES:
        assert rule in _STRICT_GENERATION_PROMPT, f"missing hard rule: {rule!r}"


def test_hard_numeric_rule_present_in_natural_prompt() -> None:
    assert "Don't synthesize or invent numbers not in the evidence" in (
        _NATURAL_GENERATION_PROMPT
    )


def test_tone_language_clause_does_not_weaken_numeric_rules() -> None:
    """A combined assertion: when tone+language are set, the produced generation
    instructions STILL coexist with (do not override) the hard numeric/unit
    rules. The clause text itself reinforces the invariants."""
    out = _build_reclaim_generation_instructions(
        _synth_config(tone=Tone.CASUAL, output_language="French")
    )
    # The style clause explicitly preserves the downstream hard rules.
    assert "WITHOUT relaxing any citation, numeric, or unit rule" in out
    # And the downstream prompt still carries the canonical guardrails verbatim.
    for rule in _HARD_NUMERIC_UNIT_RULES:
        assert rule in _STRICT_GENERATION_PROMPT


# ---------------------------------------------------------------------------
# Language re-force in the drift-prone Stage-2 sub-call (redundancy)
# ---------------------------------------------------------------------------


def _make_ranked_evidence() -> RankedEvidence:
    return RankedEvidence(
        source_id=None,
        source_url="https://example.com/a",
        source_title="Source A",
        quote_text="The system reported revenue of $3.2 billion in Q4.",
        start_offset=0,
        end_offset=50,
        section_heading=None,
        relevance_score=0.9,
        has_numeric_content=True,
        is_snippet_based=False,
    )


def _capturing_llm() -> tuple[MagicMock, list[str]]:
    """Mock FrameworkLLMClient that records each prompt passed to ``complete``."""
    captured: list[str] = []

    async def _complete(*, messages: list[dict[str, Any]], **_kwargs: Any) -> LLMResponse:
        captured.append(messages[0]["content"])
        return LLMResponse(content="Revenue was $3.2 billion [0].", structured=None)

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=_complete)
    return llm, captured


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["Spanish", "Japanese", "German"])
async def test_output_language_reforced_in_generation_prompt(language: str) -> None:
    """Multi-language smoke: setting ``output_language`` re-forces a BINDING
    language directive on the Stage-2 generation prompt (redundant with the
    instructions clause) so the report does not drift back to English."""
    llm, captured = _capturing_llm()
    gen = InterleavedGenerator(llm)

    async for _ in gen.synthesize_with_streaming(
        query="What was Q4 revenue?",
        evidence_pool=[_make_ranked_evidence()],
        output_language=language,
    ):
        pass

    assert captured, "generation prompt was never built"
    prompt = captured[0]
    assert "OUTPUT LANGUAGE (BINDING)" in prompt
    assert language in prompt
    # The re-force must not collide with numeric/unit invariants.
    assert "Do NOT translate or alter numeric values" in prompt


@pytest.mark.asyncio
async def test_no_output_language_leaves_prompt_unchanged() -> None:
    """Default path: no ``output_language`` => no language block appended."""
    llm, captured = _capturing_llm()
    gen = InterleavedGenerator(llm)

    async for _ in gen.synthesize_with_streaming(
        query="What was Q4 revenue?",
        evidence_pool=[_make_ranked_evidence()],
    ):
        pass

    assert captured
    assert "OUTPUT LANGUAGE (BINDING)" not in captured[0]
