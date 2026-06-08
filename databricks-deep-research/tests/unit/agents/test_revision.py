"""Unit tests for the ``RevisionContext`` model and the
``build_revision_block_md`` synthesizer pre-prompt hook.

The tests treat the abstraction as a black box: shape in, shape out. They
do not assume any specific workflow definition.
"""
from __future__ import annotations

import json

import pytest

from databricks_deep_research.agents.output_models import (
    ReflectionDirective,
    ReflectionOutput,
)
from databricks_deep_research.agents.revision import (
    RevisionContext,
    build_revision_block_md,
    parse_reflection_output,
)


# ---------------------------------------------------------------------------
# ReflectionDirective + ReflectionOutput defaulting
# ---------------------------------------------------------------------------


def test_reflection_directive_construction() -> None:
    d = ReflectionDirective(severity="critical", section="X", issue="Y", fix="Z")
    assert d.severity == "critical"
    assert d.section == "X"


def test_reflection_directive_rejects_empty_fields() -> None:
    with pytest.raises(Exception):  # noqa: BLE001 — Pydantic ValidationError
        ReflectionDirective(severity="major", section="", issue="ok", fix="ok")


def test_reflection_output_directives_defaults_empty() -> None:
    """Reflectors that don't emit ``directives`` still validate (backcompat)."""
    out = ReflectionOutput(decision="continue", reasoning="all good")
    assert out.directives == []


def test_reflection_output_directives_accepts_list() -> None:
    out = ReflectionOutput(
        decision="adjust",
        reasoning="x",
        directives=[ReflectionDirective(severity="minor", section="A", issue="b", fix="c")],
    )
    assert len(out.directives) == 1


def test_reflection_output_directives_coerces_none_to_empty() -> None:
    out = ReflectionOutput.model_validate(
        {"decision": "continue", "reasoning": "", "directives": None}
    )
    assert out.directives == []


# ---------------------------------------------------------------------------
# RevisionContext.render_as_markdown
# ---------------------------------------------------------------------------


def _make_directive(**overrides: object) -> ReflectionDirective:
    defaults: dict[str, object] = {
        "severity": "major",
        "section": "Section",
        "issue": "Something is wrong",
        "fix": "Do the fix",
    }
    defaults.update(overrides)
    return ReflectionDirective(**defaults)  # type: ignore[arg-type]


def test_revision_context_renders_when_directives_present() -> None:
    ctx = RevisionContext(
        prior_draft="# Draft\n\nBody",
        directives=[_make_directive()],
        reflector_reasoning="Coverage gap on X",
        passes_remaining=1,
    )
    md = ctx.render_as_markdown()
    assert "REVISION PASS" in md
    assert "passes remaining: 1" in md
    assert "Section: Something is wrong" in md
    assert "FIX: Do the fix" in md
    assert "DIRECTIVE RESPONSES" in md
    # Prior draft is wrapped in a fenced code block (prompt-injection safety).
    assert "```markdown\n# Draft\n\nBody\n```" in md


def test_revision_context_renders_depletion_when_passes_zero() -> None:
    ctx = RevisionContext(
        prior_draft="prior",
        directives=[_make_directive()],
        reflector_reasoning="reason",
        passes_remaining=0,
    )
    md = ctx.render_as_markdown()
    assert "REVISION DEPLETED" in md
    assert "Finalise the report" in md
    # No DIRECTIVE RESPONSES table when depleted — that's the revision flow.
    assert "DIRECTIVE RESPONSES" not in md


def test_revision_context_passes_long_draft_through_verbatim() -> None:
    """No framework-level truncation; length defers to user intent."""
    long_draft = "x" * 50000
    ctx = RevisionContext(prior_draft=long_draft, directives=[_make_directive()])
    md = ctx.render_as_markdown()
    assert long_draft in md
    assert "[truncated" not in md


def test_revision_context_handles_zero_directives() -> None:
    """When directives list is empty, render shows a placeholder rather than crashing."""
    ctx = RevisionContext(
        prior_draft="prior",
        directives=[],
        reflector_reasoning="reason",
        passes_remaining=1,
    )
    md = ctx.render_as_markdown()
    assert "(none — reflector provided no structured directives" in md


# ---------------------------------------------------------------------------
# parse_reflection_output — tolerant coercer
# ---------------------------------------------------------------------------


def test_parse_reflection_output_passthrough() -> None:
    out = ReflectionOutput(decision="adjust", reasoning="r")
    assert parse_reflection_output(out) is out


def test_parse_reflection_output_from_dict() -> None:
    parsed = parse_reflection_output({"decision": "adjust", "reasoning": "r"})
    assert parsed is not None
    assert parsed.decision == "adjust"


def test_parse_reflection_output_from_json_string() -> None:
    raw = json.dumps({"decision": "complete", "reasoning": "done"})
    parsed = parse_reflection_output(raw)
    assert parsed is not None
    assert parsed.decision == "complete"


def test_parse_reflection_output_from_malformed_returns_none() -> None:
    assert parse_reflection_output(None) is None
    assert parse_reflection_output("not json") is None
    assert parse_reflection_output({"decision": "garbage"}) is None
    assert parse_reflection_output(["a", "b"]) is None


# ---------------------------------------------------------------------------
# build_revision_block_md — the public hook
# ---------------------------------------------------------------------------


def test_build_revision_block_md_empty_when_no_draft() -> None:
    assert build_revision_block_md({}) == ""
    assert build_revision_block_md({"coverage_review": {"decision": "adjust"}}) == ""


def test_build_revision_block_md_empty_when_decision_continue() -> None:
    out = build_revision_block_md(
        {
            "draft_report": "draft",
            "coverage_review": ReflectionOutput(decision="continue", reasoning="r"),
        }
    )
    assert out == ""


def test_build_revision_block_md_empty_when_decision_complete() -> None:
    out = build_revision_block_md(
        {
            "draft_report": "draft",
            "coverage_review": {"decision": "complete", "reasoning": "r"},
        }
    )
    assert out == ""


def test_build_revision_block_md_renders_on_adjust_with_directives() -> None:
    out = build_revision_block_md(
        {
            "draft_report": "Draft body",
            "coverage_review": {
                "decision": "adjust",
                "reasoning": "Coverage gaps remain",
                "directives": [
                    {
                        "severity": "critical",
                        "section": "Risk",
                        "issue": "Missing X",
                        "fix": "Add X paragraph",
                    }
                ],
            },
        }
    )
    assert "REVISION PASS" in out
    assert "Risk: Missing X" in out
    assert "Add X paragraph" in out


def test_build_revision_block_md_fallback_extracts_from_reasoning() -> None:
    """When ``directives`` is empty but reasoning has severity-tagged bullets,
    the fallback recovers usable directives."""
    reasoning = (
        "Several defects:\n"
        "- **critical:** Risk Analysis: Missing breach litigation — FIX: add it\n"
        "- [major] Fundamentals: Stale numbers — FIX: refresh from Q4\n"
        "  Some extra prose that's not a directive\n"
    )
    out = build_revision_block_md(
        {
            "draft_report": "draft",
            "coverage_review": {
                "decision": "adjust",
                "reasoning": reasoning,
                "directives": [],
            },
        }
    )
    assert "REVISION PASS" in out
    assert "Missing breach litigation" in out
    assert "Stale numbers" in out


def test_build_revision_block_md_passes_remaining_zero_renders_depleted() -> None:
    out = build_revision_block_md(
        {
            "draft_report": "draft",
            "coverage_review": {
                "decision": "adjust",
                "reasoning": "still issues",
                "directives": [
                    {"severity": "minor", "section": "X", "issue": "y", "fix": "z"}
                ],
            },
            "revision_passes_remaining": 0,
        }
    )
    assert "REVISION DEPLETED" in out


def test_build_revision_block_md_non_string_draft_returns_empty() -> None:
    assert build_revision_block_md({"draft_report": 12345, "coverage_review": {}}) == ""


def test_build_revision_block_md_never_raises_on_garbage_state() -> None:
    """Defensive: any unexpected shape returns empty, never raises."""
    assert build_revision_block_md(None) == ""  # type: ignore[arg-type]
    assert build_revision_block_md({"coverage_review": 42, "draft_report": "x"}) == ""
    assert build_revision_block_md({"coverage_review": {"decision": "adjust"}, "draft_report": ""}) == ""


def test_build_revision_block_md_handles_pydantic_review_object() -> None:
    review = ReflectionOutput(
        decision="adjust",
        reasoning="r",
        directives=[
            ReflectionDirective(severity="major", section="S", issue="i", fix="f")
        ],
    )
    out = build_revision_block_md({"draft_report": "d", "coverage_review": review})
    assert "REVISION PASS" in out
    assert "S: i" in out
