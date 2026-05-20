"""Regression test for the prompt_context.py rewrite (+435 lines).

Guards the basic shape of ``compile_synthesis_context`` output so the
rewrite cannot silently regress: observations are rendered as bullets,
sources are rendered with their URL preserved (citations require URLs),
and the resulting ``CompiledSynthesisContext`` contains both halves
together with stable stats. The test uses fakes for the LLM client and
pool state so it runs offline under ``make test``.
"""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.agents.prompt_context import (
    CompiledSynthesisContext,
    compile_synthesis_context,
    default_synthesis_context,
)


class _FakePool:
    """Minimal pool stub exposing ``snapshot()`` like the real PoolState."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def snapshot(self) -> list[Any]:
        return list(self._items)


class _FakeLLMClient:
    """Stand-in for FrameworkLLMClient — never called in this test.

    The default config does not invoke summarisation for inputs under the
    compaction threshold, so tests that stay below that threshold can use
    a fake that raises if the LLM is touched.
    """

    async def complete(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError(
            "LLM was called for a compile that should not require summarisation"
        )


@pytest.mark.asyncio
async def test_compile_synthesis_context_renders_observations_and_sources() -> None:
    obs_pool = _FakePool(
        items=[
            {"text": "Q3 revenue grew 12% year-over-year."},
            {"text": "Operating margin held at 28%."},
        ]
    )
    src_pool = _FakePool(
        items=[
            {
                "title": "ACME Q3 2026 earnings call",
                "url": "https://example.com/q3-2026",
                "snippet": "Revenue +12% YoY; margin steady at 28%.",
                "content": "Full transcript discussing the quarterly results.",
            },
        ]
    )

    result = await compile_synthesis_context(
        query="What were ACME Q3 2026 earnings highlights?",
        pools={"observations": obs_pool, "sources": src_pool},
        llm_client=_FakeLLMClient(),
        config=default_synthesis_context(),
    )

    assert isinstance(result, CompiledSynthesisContext)
    # Observations land as bullet lines.
    assert "Q3 revenue grew 12%" in result.all_observations
    assert "Operating margin held at 28%" in result.all_observations
    # Sources keep title + URL so the synthesizer can cite them.
    assert "ACME Q3 2026 earnings call" in result.sources_list
    assert "https://example.com/q3-2026" in result.sources_list
    # Stats track what came in vs out so we can detect a silent rewrite drop.
    assert result.stats.observation_items_in == 2
    assert result.stats.observation_items_out == 2
    assert result.stats.source_items_in == 1
    assert result.stats.source_clusters_out >= 1


@pytest.mark.asyncio
async def test_compile_synthesis_context_handles_empty_pools() -> None:
    result = await compile_synthesis_context(
        query="empty",
        pools={},
        llm_client=_FakeLLMClient(),
        config=default_synthesis_context(),
    )
    assert result.all_observations == ""
    assert result.sources_list == ""
    assert result.fallback_discovery_sources == ""
    assert result.stats.observation_items_in == 0
    assert result.stats.source_items_in == 0


@pytest.mark.asyncio
async def test_compile_synthesis_context_dedupes_sources_by_url() -> None:
    """Two source entries with the same URL must collapse — guards against the
    URL registry losing dedup behaviour after the rewrite."""
    src_pool = _FakePool(
        items=[
            {
                "title": "ACME 10-K filing",
                "url": "https://example.com/10k",
                "snippet": "Full annual report.",
            },
            {
                "title": "ACME 10-K filing (mirror)",
                "url": "https://example.com/10k",
                "snippet": "Mirror of the same report.",
            },
        ]
    )
    result = await compile_synthesis_context(
        query="annual report",
        pools={"sources": src_pool},
        llm_client=_FakeLLMClient(),
        config=default_synthesis_context(),
    )
    # The URL should appear at most once even though there are two raw entries.
    assert result.sources_list.count("https://example.com/10k") <= 1


@pytest.mark.asyncio
async def test_compile_synthesis_context_filters_metadata_only_sources() -> None:
    src_pool = _FakePool(
        items=[
            {
                "title": "Search result title only",
                "url": "https://example.com/title-only",
                "evidence_quality": "metadata_only",
                "admission_status": "accepted_low_value",
            },
            {
                "title": "Crawled source",
                "url": "https://example.com/crawled",
                "snippet": "Extracted source text.",
                "evidence_quality": "snippet_only",
                "admission_status": "accepted",
            },
        ]
    )
    result = await compile_synthesis_context(
        query="evidence filter",
        pools={"sources": src_pool},
        llm_client=_FakeLLMClient(),
        config=default_synthesis_context(),
    )

    assert "title-only" not in result.sources_list
    assert "https://example.com/crawled" in result.sources_list
    assert result.stats.source_items_in == 1
