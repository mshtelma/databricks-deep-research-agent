"""Unit tests for orchestrator shared helper functions (C1 fix)."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.framework_orchestrator import (
    _extract_verification_from_report,
)
from deep_research.agent.orchestrator import (
    OrchestrationConfig,
    stream_research,
)
from deep_research.schemas.streaming import ResearchCompletedEvent


class TestFrameworkDelegation:
    """Ensure the production orchestrator path delegates to the framework runtime."""

    @pytest.mark.asyncio
    async def test_stream_research_uses_framework_path(self) -> None:
        """stream_research should yield framework events and skip legacy setup."""
        config = OrchestrationConfig()
        terminal_event = ResearchCompletedEvent(
            session_id=uuid4(),
            total_steps_executed=1,
            total_steps_skipped=0,
            plan_iterations=1,
            total_duration_ms=1,
            final_report="done",
        )

        async def _mock_framework_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield terminal_event

        with patch(
            "deep_research.agent.framework_orchestrator.stream_research_via_framework",
            side_effect=_mock_framework_stream,
        ) as framework_stream:
            events = [
                event
                async for event in stream_research(
                    query="framework only",
                    llm=MagicMock(),
                    brave_client=MagicMock(),
                    crawler=MagicMock(),
                    config=config,
                )
            ]

        assert events == [terminal_event]
        framework_stream.assert_called_once()


# =============================================================================
# _extract_verification_from_report tests
# =============================================================================


def _make_source(url: str, title: str) -> object:
    """Create a lightweight source object for testing."""
    from types import SimpleNamespace

    return SimpleNamespace(url=url, title=title, snippet=f"Snippet for {title}")


class TestExtractVerificationFromReport:
    """Tests for _extract_verification_from_report."""

    def test_1indexed_markers_map_to_correct_sources(self) -> None:
        """LLM uses [1], [2] (1-indexed) → sources[0], sources[1]."""
        report = "Revenue grew 2% year over year [1]. EPS guidance was raised for Q4 [2]."
        sources = [
            _make_source("https://a.com", "Source A"),
            _make_source("https://b.com", "Source B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "1"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://a.com"
        assert claims[1].citation_key == "2"
        assert claims[1].evidence is not None
        assert claims[1].evidence.source_url == "https://b.com"

    def test_0indexed_markers_map_to_correct_sources(self) -> None:
        """LLM uses [0], [1] (0-indexed) → sources[0], sources[1]."""
        report = "Revenue grew 2% year over year [0]. EPS guidance was raised for Q4 [1]."
        sources = [
            _make_source("https://a.com", "Source A"),
            _make_source("https://b.com", "Source B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "0"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://a.com"
        assert claims[1].citation_key == "1"
        assert claims[1].evidence is not None
        assert claims[1].evidence.source_url == "https://b.com"

    def test_citation_keys_are_numeric_strings(self) -> None:
        """Citation keys must be numeric strings matching markdown parser."""
        report = "Multi-source claim about earnings [1][3]. Single claim about revenue growth [2]."
        sources = [_make_source(f"https://{i}.com", f"S{i}") for i in range(4)]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "1"
        assert claims[0].citation_keys == ["1", "3"]
        assert claims[1].citation_key == "2"
        assert claims[1].citation_keys is None  # single marker → None

    def test_out_of_bounds_marker_no_crash(self) -> None:
        """Markers beyond source count produce claims without evidence."""
        report = "This claim references a nonexistent source [5]."
        sources = [_make_source("https://a.com", "A")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 1
        assert claims[0].evidence is None
        assert claims[0].citation_key == "5"

    def test_empty_sources_returns_empty(self) -> None:
        """Empty sources list returns no claims."""
        report = "Some text with markers [1]."
        claims, summary = _extract_verification_from_report(report, [])
        assert claims == []
        assert summary is None

    def test_no_markers_returns_empty(self) -> None:
        """Report without markers produces no claims."""
        report = "Just plain text without any citations at all."
        sources = [_make_source("https://a.com", "A")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert claims == []

    def test_none_report_returns_empty(self) -> None:
        """None report returns no claims."""
        claims, summary = _extract_verification_from_report(None, [_make_source("https://a.com", "A")])
        assert claims == []
        assert summary is None

    def test_single_1indexed_marker(self) -> None:
        """Single [1] marker with 1-indexed convention maps to sources[0]."""
        report = "The company reported strong earnings growth [1]."
        sources = [_make_source("https://only.com", "Only Source")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 1
        assert claims[0].citation_key == "1"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://only.com"

    def test_verification_summary_counts(self) -> None:
        """Summary has correct claim counts."""
        report = "First claim here [1]. Second claim here [2]. Third claim here [1][2]."
        sources = [
            _make_source("https://a.com", "A"),
            _make_source("https://b.com", "B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 3
        assert summary is not None
        assert summary.total_claims == 3
        assert summary.supported_count == 3
        assert summary.unsupported_count == 0
