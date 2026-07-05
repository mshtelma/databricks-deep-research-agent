"""Tests for evidence-list construction (agent/structured_evidence.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from deep_research.agent.structured_evidence import (
    EvidenceItem,
    build_evidence,
    build_legend,
    render_evidence_block,
)

pytestmark = pytest.mark.unit


def _pool(url: str, relevance: float = 0.0, **extra: Any) -> dict[str, Any]:
    return {"url": url, "title": f"T {url}", "relevance_score": relevance, **extra}


# ---------------------------------------------------------------------------
# build_evidence — shapes, dedupe, ranking, refs
# ---------------------------------------------------------------------------


def test_build_evidence_dedupes_by_url() -> None:
    items = build_evidence([_pool("https://a"), _pool("https://a")], [])
    assert len(items) == 1


def test_build_evidence_ranks_cited_first_then_relevance() -> None:
    sources = [
        _pool("https://low", relevance=0.1),
        _pool("https://high", relevance=0.9),
        _pool("https://cited", relevance=0.0),
    ]
    claims = [{"evidence": {"source_url": "https://cited"}}]
    items = build_evidence(sources, claims)
    assert [i.url for i in items] == ["https://cited", "https://high", "https://low"]
    assert [i.ref for i in items] == ["1", "2", "3"]


def test_build_evidence_object_shaped_claims() -> None:
    claims = [SimpleNamespace(evidence=SimpleNamespace(source_url="https://x"))]
    items = build_evidence([_pool("https://x"), _pool("https://y", 0.5)], claims)
    assert items[0].url == "https://x"


def test_build_evidence_insertion_order_tiebreak() -> None:
    items = build_evidence([_pool("https://a"), _pool("https://b")], [])
    assert [i.url for i in items] == ["https://a", "https://b"]


def test_build_evidence_caps_items() -> None:
    sources = [_pool(f"https://s{i}") for i in range(40)]
    items = build_evidence(sources, [], max_items=24)
    assert len(items) == 24
    assert items[-1].ref == "24"


def test_build_evidence_pool_dict_fields() -> None:
    items = build_evidence(
        [{"url": "https://a", "filename": "doc.pdf", "snippet": "s", "content": "c"}],
        [],
    )
    assert items[0].title == "doc.pdf"
    assert items[0].snippet == "s"
    assert items[0].content == "c"


def test_build_evidence_orm_like_object() -> None:
    row = SimpleNamespace(
        url="https://a", title="Row", snippet="snip", content=None,
        relevance_score=0.4,
    )
    items = build_evidence([row], [])
    assert items[0].title == "Row"
    assert items[0].snippet == "snip"


def test_build_evidence_cached_docsource_metadata() -> None:
    doc = SimpleNamespace(
        url="https://a",
        title="Doc",
        metadata={"snippet": "ms", "content": "mc", "relevance_score": 0.7},
    )
    items = build_evidence([doc], [])
    assert items[0].snippet == "ms"
    assert items[0].content == "mc"


def test_build_evidence_skips_urlless_sources() -> None:
    assert build_evidence([{"title": "no url"}, SimpleNamespace(title="x")], []) == []


# ---------------------------------------------------------------------------
# render_evidence_block
# ---------------------------------------------------------------------------


def _items(n: int, content: str | None = None) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            ref=str(i + 1),
            url=f"https://s{i}",
            title=f"Source {i}",
            snippet=f"snippet {i}",
            content=content,
        )
        for i in range(n)
    ]


def test_render_headers_always_present_even_over_budget() -> None:
    block = render_evidence_block(_items(5, content="c " * 400), budget_chars=0)
    for i in range(5):
        assert f"[{i + 1}] Source {i} — https://s{i}" in block
    assert "Snippet:" not in block
    assert "Content:" not in block


def test_render_top_k_gets_longer_content() -> None:
    long_content = "word " * 1000
    block = render_evidence_block(
        _items(3, content=long_content), budget_chars=100_000, full_top_k=1
    )
    lines = [ln for ln in block.splitlines() if ln.strip().startswith("Content:")]
    assert len(lines) == 3
    assert len(lines[0]) > len(lines[1])
    assert len(lines[1]) == len(lines[2])


def test_render_respects_budget_for_optional_parts() -> None:
    block = render_evidence_block(_items(10, content="c" * 2000), budget_chars=3000)
    # Mandatory headers plus at most ~budget of optional detail.
    optional = sum(
        len(ln)
        for ln in block.splitlines()
        if ln.strip().startswith(("Snippet:", "Content:"))
    )
    assert optional <= 3000


# ---------------------------------------------------------------------------
# build_legend
# ---------------------------------------------------------------------------


def test_build_legend_only_used_refs_in_item_order() -> None:
    items = _items(4)
    legend = build_legend(items, {"3", "1"})
    assert legend == [
        {"ref": "1", "url": "https://s0", "title": "Source 0"},
        {"ref": "3", "url": "https://s2", "title": "Source 2"},
    ]


def test_build_legend_empty_when_nothing_used() -> None:
    assert build_legend(_items(2), set()) == []
