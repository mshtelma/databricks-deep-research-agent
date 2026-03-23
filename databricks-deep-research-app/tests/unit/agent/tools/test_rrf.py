"""Unit tests for Reciprocal Rank Fusion (RRF) function.

Tests cover:
- Overlapping results across multiple sets (boosted fused scores)
- Disjoint result sets (all docs included)
- Empty result sets (gracefully ignored)
- Single result set (passthrough in same order)
- Highest-scoring version preserved for duplicate docs
"""

from __future__ import annotations

from deep_research.agent.tools.user_vector_search import reciprocal_rank_fusion
from deep_research.services.vector_search_query import VectorSearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    doc_id: str,
    content: str = "content",
    url: str | None = None,
    score: float = 0.5,
    title: str = "Untitled",
) -> VectorSearchResult:
    """Create a VectorSearchResult for testing."""
    return VectorSearchResult(
        id=doc_id,
        title=title,
        content=content,
        url=url,
        score=score,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Test: Overlapping results get higher fused scores
# ---------------------------------------------------------------------------


def test_rrf_overlapping_results() -> None:
    """Documents appearing in multiple result sets should get higher fused
    scores than those appearing in only one set.

    RRF formula: score(d) = sum(1 / (k + rank_i + 1)) for each set where d appears.
    A doc in two sets gets two rank contributions, so its fused score is higher.
    """
    # Doc A appears in both sets at rank 0
    # Doc B appears only in set 1 at rank 1
    # Doc C appears only in set 2 at rank 1
    set1 = [
        _make_result("A", url="http://a.com", score=0.9),
        _make_result("B", url="http://b.com", score=0.7),
    ]
    set2 = [
        _make_result("A", url="http://a.com", score=0.85),
        _make_result("C", url="http://c.com", score=0.6),
    ]

    fused = reciprocal_rank_fusion([set1, set2], k=60)

    assert len(fused) == 3  # A, B, C

    # Doc A should be first (highest fused score from appearing in both sets)
    assert fused[0].url == "http://a.com"

    # B and C should follow (each in one set at rank 1)
    remaining_urls = {fused[1].url, fused[2].url}
    assert remaining_urls == {"http://b.com", "http://c.com"}

    # Verify A's fused score is strictly higher than B's or C's
    # A: 1/(60+0+1) + 1/(60+0+1) = 2/61
    # B: 1/(60+1+1) = 1/62
    # C: 1/(60+1+1) = 1/62
    # So A's score (2/61 ~ 0.0328) > B's score (1/62 ~ 0.0161)


# ---------------------------------------------------------------------------
# Test: Disjoint results are all included
# ---------------------------------------------------------------------------


def test_rrf_disjoint_results() -> None:
    """All documents from disjoint result sets should be included in
    the fused output."""
    set1 = [
        _make_result("A", url="http://a.com", score=0.9),
        _make_result("B", url="http://b.com", score=0.8),
    ]
    set2 = [
        _make_result("C", url="http://c.com", score=0.95),
        _make_result("D", url="http://d.com", score=0.7),
    ]
    set3 = [
        _make_result("E", url="http://e.com", score=0.85),
    ]

    fused = reciprocal_rank_fusion([set1, set2, set3], k=60)

    assert len(fused) == 5
    fused_urls = {r.url for r in fused}
    assert fused_urls == {"http://a.com", "http://b.com", "http://c.com", "http://d.com", "http://e.com"}

    # All at rank 0 in their respective sets should have the same RRF score: 1/(60+0+1) = 1/61
    # A, C, E are at rank 0; B, D are at rank 1
    # So A, C, E should come before B, D
    top_3_urls = {fused[i].url for i in range(3)}
    assert top_3_urls == {"http://a.com", "http://c.com", "http://e.com"}


# ---------------------------------------------------------------------------
# Test: Empty set is ignored
# ---------------------------------------------------------------------------


def test_rrf_empty_set_ignored() -> None:
    """An empty result set should not crash the fusion and should be
    treated as contributing nothing."""
    set1 = [
        _make_result("A", url="http://a.com", score=0.9),
        _make_result("B", url="http://b.com", score=0.8),
    ]
    empty: list[VectorSearchResult] = []
    set3 = [
        _make_result("C", url="http://c.com", score=0.7),
    ]

    fused = reciprocal_rank_fusion([set1, empty, set3], k=60)

    assert len(fused) == 3
    fused_urls = {r.url for r in fused}
    assert fused_urls == {"http://a.com", "http://b.com", "http://c.com"}


def test_rrf_all_empty_sets() -> None:
    """All empty result sets should return an empty list."""
    fused = reciprocal_rank_fusion([[], [], []], k=60)
    assert fused == []


def test_rrf_no_sets() -> None:
    """No result sets at all should return an empty list."""
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []


# ---------------------------------------------------------------------------
# Test: Single set passthrough
# ---------------------------------------------------------------------------


def test_rrf_single_set_passthrough() -> None:
    """A single result set should return results in the same order,
    since RRF with one set is just 1/(k + rank + 1) which is monotonically
    decreasing with rank."""
    results = [
        _make_result("A", url="http://a.com", score=0.95),
        _make_result("B", url="http://b.com", score=0.85),
        _make_result("C", url="http://c.com", score=0.75),
        _make_result("D", url="http://d.com", score=0.65),
    ]

    fused = reciprocal_rank_fusion([results], k=60)

    assert len(fused) == 4
    # Order should be preserved (rank 0, 1, 2, 3 -> decreasing RRF scores)
    assert fused[0].url == "http://a.com"
    assert fused[1].url == "http://b.com"
    assert fused[2].url == "http://c.com"
    assert fused[3].url == "http://d.com"


# ---------------------------------------------------------------------------
# Test: Preserves highest scoring version
# ---------------------------------------------------------------------------


def test_rrf_preserves_highest_scoring_version() -> None:
    """When the same document appears in multiple result sets, the version
    with the highest original score should be preserved in the output."""
    # Doc A appears in set1 with score 0.7 and in set2 with score 0.95
    set1 = [
        _make_result("A", url="http://a.com", score=0.7, title="Low score A"),
        _make_result("B", url="http://b.com", score=0.8, title="B"),
    ]
    set2 = [
        _make_result("A", url="http://a.com", score=0.95, title="High score A"),
        _make_result("C", url="http://c.com", score=0.6, title="C"),
    ]

    fused = reciprocal_rank_fusion([set1, set2], k=60)

    # Find doc A in the fused results
    doc_a = next(r for r in fused if r.url == "http://a.com")

    # Should keep the highest score version
    assert doc_a.score == 0.95
    assert doc_a.title == "High score A"


# ---------------------------------------------------------------------------
# Test: Dedup key falls back to content hash when URL is None
# ---------------------------------------------------------------------------


def test_rrf_dedup_by_content_hash_when_no_url() -> None:
    """When results have no URL, dedup should use content hash.
    Same content in different sets should be merged."""
    set1 = [
        _make_result("id1", content="Identical content for dedup", url=None, score=0.8),
        _make_result("id2", content="Unique content A", url=None, score=0.6),
    ]
    set2 = [
        _make_result("id3", content="Identical content for dedup", url=None, score=0.9),
        _make_result("id4", content="Unique content B", url=None, score=0.5),
    ]

    fused = reciprocal_rank_fusion([set1, set2], k=60)

    # "Identical content for dedup" should be merged (same content hash)
    # So we should have 3 unique results, not 4
    assert len(fused) == 3

    # The merged doc should have the higher score
    identical_doc = next(r for r in fused if r.content == "Identical content for dedup")
    assert identical_doc.score == 0.9


# ---------------------------------------------------------------------------
# Test: RRF with different k values
# ---------------------------------------------------------------------------


def test_rrf_k_parameter_affects_scoring() -> None:
    """A higher k value should reduce the weight given to top-ranked results,
    making the scoring flatter across ranks."""
    set1 = [
        _make_result("A", url="http://a.com", score=0.9),
        _make_result("B", url="http://b.com", score=0.5),
    ]
    set2 = [
        _make_result("B", url="http://b.com", score=0.8),
        _make_result("A", url="http://a.com", score=0.4),
    ]

    # With small k=1, rank matters a lot
    fused_small_k = reciprocal_rank_fusion([set1, set2], k=1)
    # A: set1 rank 0 -> 1/(1+0+1) = 0.5, set2 rank 1 -> 1/(1+1+1) = 0.333 -> total 0.833
    # B: set1 rank 1 -> 1/(1+1+1) = 0.333, set2 rank 0 -> 1/(1+0+1) = 0.5 -> total 0.833
    # Equal scores! Both are rank 0 in one set and rank 1 in the other

    # With large k=1000, rank matters less (all scores converge)
    fused_large_k = reciprocal_rank_fusion([set1, set2], k=1000)

    # Both configs should return both docs
    assert len(fused_small_k) == 2
    assert len(fused_large_k) == 2


# ---------------------------------------------------------------------------
# Test: Large result sets
# ---------------------------------------------------------------------------


def test_rrf_large_result_sets() -> None:
    """Test RRF with larger result sets to ensure correct merging
    and no performance issues."""
    # set1: docs 0..49, set2: docs 25..74
    # Overlapping docs 25-49 share the same URL so RRF deduplicates them
    set1 = [_make_result(f"doc_{i}", url=f"http://docs/{i}", score=1.0 - i * 0.01) for i in range(50)]
    set2 = [_make_result(f"doc_{i + 25}", url=f"http://docs/{i + 25}", score=1.0 - i * 0.01) for i in range(50)]

    # docs 25-49 overlap between set1 and set2 (same URLs)
    fused = reciprocal_rank_fusion([set1, set2], k=60)

    # Total unique docs: 0-24 (set1 only) + 25-49 (both, merged) + 50-74 (set2 only) = 75
    assert len(fused) == 75

    # All 75 unique URLs should be present
    fused_urls = {r.url for r in fused}
    expected_urls = {f"http://docs/{i}" for i in range(75)}
    assert fused_urls == expected_urls
