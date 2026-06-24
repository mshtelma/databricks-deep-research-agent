"""Tests for the framework-side cross-session memory READ render policy.

Covers the pure selection policy (confidence threshold + max-cap + ordering),
the spotlighted role=user message builder, and the empty/no-op short-circuit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from databricks_deep_research.memory import (
    CrossSessionFact,
    build_cross_session_memory_message,
    render_cross_session_facts,
    select_facts,
)


def _fact(content: str, confidence: str = "high", days_ago: int = 0) -> CrossSessionFact:
    return CrossSessionFact(
        content=content,
        confidence=confidence,  # type: ignore[arg-type]
        updated_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


class TestSelectFacts:
    def test_confidence_threshold_drops_low(self) -> None:
        facts = [_fact("keep-high", "high"), _fact("drop-low", "low")]
        selected = select_facts(facts, min_confidence="medium")
        contents = [f.content for f in selected]
        assert "keep-high" in contents
        assert "drop-low" not in contents

    def test_low_floor_keeps_low(self) -> None:
        facts = [_fact("low-fact", "low")]
        selected = select_facts(facts, min_confidence="low")
        assert [f.content for f in selected] == ["low-fact"]

    def test_max_cap_honored(self) -> None:
        facts = [_fact(f"f{i}", "high", days_ago=i) for i in range(10)]
        selected = select_facts(facts, max_facts=3)
        assert len(selected) == 3

    def test_max_cap_zero_evicts_all(self) -> None:
        assert select_facts([_fact("x")], max_facts=0) == []

    def test_orders_confidence_then_recency(self) -> None:
        facts = [
            _fact("older-high", "high", days_ago=5),
            _fact("newer-high", "high", days_ago=1),
            _fact("medium", "medium", days_ago=0),
        ]
        selected = select_facts(facts, min_confidence="medium", max_facts=3)
        # High before medium; within high, newer first.
        assert [f.content for f in selected] == [
            "newer-high",
            "older-high",
            "medium",
        ]


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert render_cross_session_facts([]) == ""

    def test_all_below_threshold_returns_empty(self) -> None:
        assert render_cross_session_facts([_fact("x", "low")], min_confidence="high") == ""

    def test_max_chars_truncates(self) -> None:
        rendered = render_cross_session_facts(
            [_fact("A" * 500, "high")], max_chars=50
        )
        assert len(rendered) <= 50
        assert rendered.endswith("…")

    def test_includes_confidence_label(self) -> None:
        rendered = render_cross_session_facts([_fact("the fact", "high")])
        assert "[high]" in rendered
        assert "the fact" in rendered


class TestBuildMessage:
    def test_none_when_no_facts(self) -> None:
        assert build_cross_session_memory_message([]) is None

    def test_none_when_all_below_threshold(self) -> None:
        msg = build_cross_session_memory_message(
            [_fact("x", "low")], min_confidence="high"
        )
        assert msg is None

    def test_role_is_user(self) -> None:
        msg = build_cross_session_memory_message([_fact("remember this", "high")])
        assert msg is not None
        assert msg["role"] == "user"

    def test_content_is_spotlight_wrapped(self) -> None:
        msg = build_cross_session_memory_message([_fact("remember this", "high")])
        assert msg is not None
        # Spotlighting marks untrusted content as DATA via the attached_context
        # sentinel (OWASP defense-in-depth).
        assert "<attached_context" in msg["content"]
        assert "remember this" in msg["content"]
