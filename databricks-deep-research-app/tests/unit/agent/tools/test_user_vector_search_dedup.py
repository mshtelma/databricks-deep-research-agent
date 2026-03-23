from unittest.mock import MagicMock

from deep_research.agent.tools.user_vector_search import UserVectorSearchTool


def _make_result(content: str, doc_id: str = "1") -> MagicMock:
    """Create a mock VectorSearchResult."""
    result = MagicMock()
    result.content = content
    result.id = doc_id
    result.url = f"https://example.com/{doc_id}"
    result.title = f"Document {doc_id}"
    result.score = 0.85
    return result


def test_deduplicate_results_per_call_isolation():
    """Two separate calls return the same results — no cross-call filtering."""
    results = [_make_result("Kroger Q3 2025 revenue $35.1B", "doc1")]

    # Call 1
    unique1 = UserVectorSearchTool._deduplicate_results(results)
    assert len(unique1) == 1

    # Call 2 — same results, should NOT be filtered
    unique2 = UserVectorSearchTool._deduplicate_results(results)
    assert len(unique2) == 1


def test_deduplicate_results_within_call():
    """Duplicate results within a single call are deduplicated."""
    results = [
        _make_result("Kroger Q3 2025 revenue $35.1B", "doc1"),
        _make_result("Kroger Q3 2025 revenue $35.1B", "doc2"),  # same content, diff ID
    ]
    unique = UserVectorSearchTool._deduplicate_results(results)
    assert len(unique) == 1


def test_deduplicate_results_different_content_preserved():
    """Results with different content are all preserved."""
    results = [
        _make_result("Kroger Q3 revenue $35.1B", "doc1"),
        _make_result("Kroger Q4 revenue $36.2B", "doc2"),
        _make_result("Kroger guidance update", "doc3"),
    ]
    unique = UserVectorSearchTool._deduplicate_results(results)
    assert len(unique) == 3


def test_no_seen_hashes_attribute():
    """UserVectorSearchTool must NOT have _seen_hashes instance attribute."""
    assert not hasattr(UserVectorSearchTool, '_seen_hashes')
    # The staticmethod should not reference self
    import inspect
    source = inspect.getsource(UserVectorSearchTool._deduplicate_results)
    assert "self._seen_hashes" not in source
    assert "self." not in source
