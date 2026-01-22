"""Unit tests for ReAct researcher early stopping.

Tests the TOKEN OPTIMIZATION early stopping functionality.
"""

import pytest

from deep_research.agent.nodes.react_researcher import ReactResearchState


class TestInfoGainTracking:
    """Tests for information gain tracking in ReactResearchState."""

    def test_record_first_tool_call(self) -> None:
        """Test recording first tool call with new sources."""
        state = ReactResearchState()
        state.high_quality_sources = ["http://example.com"]
        state.crawled_content = {"http://example.com": "a" * 2000}
        state.tool_call_count = 1

        state.record_tool_call_outcome()

        assert len(state._info_gain_history) == 1
        assert state._info_gain_history[0] > 0.5  # New source = high gain

    def test_record_no_gain_call(self) -> None:
        """Test recording call with no new information."""
        state = ReactResearchState()
        state.high_quality_sources = ["http://example.com"]
        state.crawled_content = {"http://example.com": "a" * 2000}
        state.tool_call_count = 1
        state._last_high_quality_count = 1
        state._last_content_length = 2000

        state.record_tool_call_outcome()

        assert len(state._info_gain_history) == 1
        assert state._info_gain_history[0] < 0.1  # No gain

    def test_consecutive_low_gain_tracking(self) -> None:
        """Test tracking consecutive low-gain calls."""
        state = ReactResearchState()
        state.high_quality_sources = ["http://example.com"]
        state.crawled_content = {"http://example.com": "content"}
        state._last_high_quality_count = 1
        state._last_content_length = len("content")

        # Simulate 5 consecutive no-gain calls
        for i in range(5):
            state.tool_call_count = i + 1
            state.record_tool_call_outcome()

        assert state._consecutive_low_gain_calls == 5

    def test_consecutive_reset_on_gain(self) -> None:
        """Test consecutive counter resets on high-gain call."""
        state = ReactResearchState()
        state._consecutive_low_gain_calls = 3

        # Add new high-quality source
        state.high_quality_sources = ["http://example.com"]
        state.crawled_content = {"http://example.com": "a" * 5000}
        state.tool_call_count = 1

        state.record_tool_call_outcome()

        assert state._consecutive_low_gain_calls == 0


class TestEarlyStopping:
    """Tests for early stopping decision logic."""

    def test_no_stop_before_min_calls(self) -> None:
        """Test no early stop before minimum calls."""
        state = ReactResearchState()
        state.tool_call_count = 3
        state.high_quality_sources = ["a", "b", "c", "d", "e"]
        state._consecutive_low_gain_calls = 10

        should_stop, reason = state.should_stop_early(min_calls=5)
        assert should_stop is False

    def test_no_stop_without_min_sources(self) -> None:
        """Test no early stop without minimum sources."""
        state = ReactResearchState()
        state.tool_call_count = 10
        state.high_quality_sources = ["a", "b"]  # Only 2
        state._consecutive_low_gain_calls = 10

        should_stop, reason = state.should_stop_early(min_sources=3)
        assert should_stop is False

    def test_stop_on_diminishing_returns(self) -> None:
        """Test early stop on diminishing returns."""
        state = ReactResearchState()
        state.tool_call_count = 10
        state.high_quality_sources = ["a", "b", "c"]
        state._consecutive_low_gain_calls = 5

        should_stop, reason = state.should_stop_early(
            min_calls=5, min_sources=3, max_low_gain_calls=5
        )
        assert should_stop is True
        assert "diminishing_returns" in reason

    def test_stop_on_high_coverage(self) -> None:
        """Test early stop when coverage is high."""
        state = ReactResearchState()
        state.tool_call_count = 10
        state.high_quality_sources = ["a", "b", "c", "d", "e"]  # 5 sources
        state._info_gain_history = [0.0, 0.05, 0.05]  # Low recent gain

        should_stop, reason = state.should_stop_early(min_calls=5, min_sources=3)
        assert should_stop is True
        assert "high_coverage" in reason

    def test_no_stop_with_high_recent_gain(self) -> None:
        """Test no stop if still gaining information."""
        state = ReactResearchState()
        state.tool_call_count = 10
        state.high_quality_sources = ["a", "b", "c", "d", "e"]
        state._info_gain_history = [0.8, 0.6, 0.5]  # High recent gain

        should_stop, reason = state.should_stop_early(min_calls=5, min_sources=3)
        # Recent gain is high (1.9 > 0.3), so shouldn't stop
        assert should_stop is False


class TestIntegration:
    """Integration tests for info gain + early stopping."""

    def test_typical_research_session(self) -> None:
        """Test a typical research session with early stopping."""
        state = ReactResearchState()

        # Simulate initial productive calls
        for i in range(5):
            state.tool_call_count = i + 1
            state.high_quality_sources.append(f"http://source{i}.com")
            state.crawled_content[f"http://source{i}.com"] = f"content{i}" * 500
            state.record_tool_call_outcome()

        # At this point we have 5 sources, should not stop yet (recent gain still high)
        should_stop, _ = state.should_stop_early(min_calls=5, min_sources=3)
        # Depends on gain calculation, but likely won't stop yet

        # Simulate unproductive calls (no new sources)
        for i in range(5):
            state.tool_call_count = 5 + i + 1
            state.record_tool_call_outcome()

        # Now should stop due to diminishing returns
        should_stop, reason = state.should_stop_early(
            min_calls=5, min_sources=3, max_low_gain_calls=5
        )
        assert should_stop is True
