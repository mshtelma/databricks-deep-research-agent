"""Unit tests for source quality tracking feature.

Tests ResearchState.record_source_quality() and the reflector's
_format_source_quality() helper.
"""

import pytest

from deep_research.agent.nodes.reflector import _format_source_quality
from deep_research.agent.state import ResearchState


class TestRecordSourceQuality:
    """Tests for ResearchState.record_source_quality()."""

    def test_quality_signal_recorded(self) -> None:
        """Record a signal, verify it appears in source_quality_history."""
        state = ResearchState(query="test query")

        state.record_source_quality("genie_sales", "good")

        assert "genie_sales" in state.source_quality_history
        assert state.source_quality_history["genie_sales"] == ["good"]

    def test_multiple_signals_same_source(self) -> None:
        """Record multiple signals for same source, verify list grows."""
        state = ResearchState(query="test query")

        state.record_source_quality("vs_docs", "good")
        state.record_source_quality("vs_docs", "low_content")
        state.record_source_quality("vs_docs", "empty")

        assert state.source_quality_history["vs_docs"] == [
            "good",
            "low_content",
            "empty",
        ]

    def test_signals_different_sources(self) -> None:
        """Record signals for different sources, verify isolation."""
        state = ResearchState(query="test query")

        state.record_source_quality("genie_sales", "good")
        state.record_source_quality("vs_docs", "empty")
        state.record_source_quality("genie_sales", "low_content")

        assert state.source_quality_history["genie_sales"] == ["good", "low_content"]
        assert state.source_quality_history["vs_docs"] == ["empty"]

    def test_empty_history_initially(self) -> None:
        """New state has empty source_quality_history."""
        state = ResearchState(query="test query")

        assert state.source_quality_history == {}
        assert len(state.source_quality_history) == 0


class TestFormatSourceQuality:
    """Tests for _format_source_quality() reflector helper."""

    def test_format_empty_history(self) -> None:
        """Empty history returns the placeholder string."""
        state = ResearchState(query="test query")

        result = _format_source_quality(state)

        assert result == "(No enterprise source data yet)"

    def test_format_with_signals(self) -> None:
        """History with signals returns formatted string with source names and signal lists."""
        state = ResearchState(query="test query")
        state.record_source_quality("genie_sales", "good")
        state.record_source_quality("genie_sales", "low_content")
        state.record_source_quality("vs_docs", "empty")

        result = _format_source_quality(state)

        # Each source should appear as a bullet line
        assert "- genie_sales:" in result
        assert "- vs_docs:" in result

        # Signals should be comma-separated within brackets
        assert "good, low_content" in result
        assert "empty" in result

    def test_format_limits_to_last_5(self) -> None:
        """More than 5 signals: only last 5 shown per source."""
        state = ResearchState(query="test query")

        signals = ["good", "empty", "low_content", "good", "good", "empty", "low_content"]
        for sig in signals:
            state.record_source_quality("vs_docs", sig)

        # Sanity check: all 7 signals are recorded in state
        assert len(state.source_quality_history["vs_docs"]) == 7

        result = _format_source_quality(state)

        # The format uses signals[-5:], so we expect only the last 5
        # Last 5 of the list: ["low_content", "good", "good", "empty", "low_content"]
        expected_last_5 = signals[-5:]
        expected_summary = ", ".join(expected_last_5)
        assert expected_summary in result

        # The first two signals ("good", "empty") should NOT appear as a prefix
        # Verify the full line contains exactly the last-5 summary
        for line in result.split("\n"):
            if "vs_docs" in line:
                assert f"[{expected_summary}]" in line
                break
        else:
            pytest.fail("vs_docs line not found in formatted output")

    def test_format_single_signal(self) -> None:
        """A single signal formats correctly without trailing comma."""
        state = ResearchState(query="test query")
        state.record_source_quality("ka_helper", "good")

        result = _format_source_quality(state)

        assert "- ka_helper: [good]" in result

    def test_format_exactly_5_signals(self) -> None:
        """Exactly 5 signals: all shown (boundary condition for the [-5:] slice)."""
        state = ResearchState(query="test query")
        signals = ["good", "empty", "low_content", "good", "empty"]
        for sig in signals:
            state.record_source_quality("genie_hr", sig)

        result = _format_source_quality(state)

        expected_summary = ", ".join(signals)
        assert f"[{expected_summary}]" in result

    def test_format_multiple_sources_each_line(self) -> None:
        """Each source gets its own bullet line."""
        state = ResearchState(query="test query")
        state.record_source_quality("source_a", "good")
        state.record_source_quality("source_b", "empty")
        state.record_source_quality("source_c", "low_content")

        result = _format_source_quality(state)

        lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
        assert len(lines) == 3
        source_names = {line.split(":")[0].strip("- ") for line in lines}
        assert source_names == {"source_a", "source_b", "source_c"}
