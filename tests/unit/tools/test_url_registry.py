"""Unit tests for URL registry."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.agent.tools.url_registry import IndexedUrl, UrlRegistry


class TestUrlRegistry:
    """Tests for UrlRegistry class."""

    def test_register_and_retrieve(self) -> None:
        registry = UrlRegistry()

        index = registry.register(
            url="https://example.com",
            title="Example",
            snippet="Test snippet",
        )

        assert index == 0
        assert registry.get_url(index) == "https://example.com"

    def test_register_returns_entry(self) -> None:
        registry = UrlRegistry()

        index = registry.register(
            url="https://example.com",
            title="Example Title",
            snippet="Example snippet",
            relevance_score=0.9,
        )

        entry = registry.get_entry(index)
        assert entry is not None
        assert entry.url == "https://example.com"
        assert entry.title == "Example Title"
        assert entry.snippet == "Example snippet"
        assert entry.relevance_score == 0.9

    def test_duplicate_returns_same_index(self) -> None:
        registry = UrlRegistry()

        idx1 = registry.register(url="https://example.com", title="First")
        idx2 = registry.register(url="https://example.com", title="Second")

        assert idx1 == idx2
        assert len(registry) == 1

    def test_incremental_indices(self) -> None:
        registry = UrlRegistry()

        idx1 = registry.register(url="https://one.com", title="One")
        idx2 = registry.register(url="https://two.com", title="Two")
        idx3 = registry.register(url="https://three.com", title="Three")

        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 2
        assert len(registry) == 3

    def test_get_nonexistent_index(self) -> None:
        registry = UrlRegistry()
        assert registry.get_url(999) is None
        assert registry.get_entry(999) is None

    def test_reverse_lookup(self) -> None:
        registry = UrlRegistry()
        registry.register(url="https://example.com", title="Test")

        assert registry.get_index("https://example.com") == 0
        assert registry.get_index("https://nonexistent.com") is None

    def test_clear(self) -> None:
        registry = UrlRegistry()
        registry.register(url="https://example.com", title="Test")

        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0
        assert registry.get_url(0) is None

    def test_list_all_sorted(self) -> None:
        registry = UrlRegistry()
        registry.register(url="https://three.com", title="Three")
        registry.register(url="https://one.com", title="One")
        registry.register(url="https://two.com", title="Two")

        all_entries = registry.list_all()
        assert len(all_entries) == 3
        assert [e.index for e in all_entries] == [0, 1, 2]
        assert [e.url for e in all_entries] == [
            "https://three.com",
            "https://one.com",
            "https://two.com",
        ]

    def test_contains_check(self) -> None:
        registry = UrlRegistry()
        registry.register(url="https://example.com", title="Test")

        assert 0 in registry
        assert 1 not in registry
        assert 999 not in registry

    def test_get_count(self) -> None:
        registry = UrlRegistry()

        assert registry.get_count() == 0

        registry.register(url="https://one.com", title="One")
        assert registry.get_count() == 1

        registry.register(url="https://two.com", title="Two")
        assert registry.get_count() == 2

    def test_metadata_storage(self) -> None:
        registry = UrlRegistry()

        index = registry.register(
            url="https://example.com",
            title="Test",
            snippet="Snippet",
            metadata={"source": "brave", "crawled": True},
        )

        entry = registry.get_entry(index)
        assert entry is not None
        assert entry.metadata == {"source": "brave", "crawled": True}

    def test_default_values(self) -> None:
        registry = UrlRegistry()

        index = registry.register(url="https://example.com")

        entry = registry.get_entry(index)
        assert entry is not None
        assert entry.title == ""
        assert entry.snippet == ""
        assert entry.relevance_score == 0.5
        assert entry.metadata == {}


class TestUrlRegistryThreadSafety:
    """Tests for thread-safety of UrlRegistry."""

    def test_concurrent_registration(self) -> None:
        """Test that concurrent registrations don't lose data."""
        registry = UrlRegistry()
        num_threads = 10
        urls_per_thread = 100

        def register_urls(thread_id: int) -> list[int]:
            indices = []
            for i in range(urls_per_thread):
                url = f"https://thread{thread_id}-url{i}.com"
                index = registry.register(url=url, title=f"Thread {thread_id} URL {i}")
                indices.append(index)
            return indices

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_urls, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        # Should have all unique URLs registered
        total_urls = num_threads * urls_per_thread
        assert len(registry) == total_urls

        # All indices should be unique across threads
        all_indices = [idx for thread_indices in results for idx in thread_indices]
        assert len(set(all_indices)) == total_urls

    def test_concurrent_read_write(self) -> None:
        """Test concurrent reads and writes don't cause issues."""
        registry = UrlRegistry()
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(50):
                    registry.register(
                        url=f"https://writer{thread_id}-{i}.com",
                        title=f"Title {i}",
                    )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(100):
                    registry.list_all()
                    for i in range(10):
                        registry.get_url(i)
                        registry.get_entry(i)
            except Exception as e:
                errors.append(e)

        threads = []

        # Start writer threads
        for i in range(3):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        # Start reader threads
        for _ in range(3):
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0


class TestIndexedUrl:
    """Tests for IndexedUrl dataclass."""

    def test_create_indexed_url(self) -> None:
        entry = IndexedUrl(
            index=0,
            url="https://example.com",
            title="Example",
            snippet="A test snippet",
            relevance_score=0.8,
        )

        assert entry.index == 0
        assert entry.url == "https://example.com"
        assert entry.title == "Example"
        assert entry.snippet == "A test snippet"
        assert entry.relevance_score == 0.8
        assert entry.metadata == {}

    def test_indexed_url_with_metadata(self) -> None:
        entry = IndexedUrl(
            index=1,
            url="https://example.com",
            title="Test",
            snippet="Snippet",
            metadata={"key": "value"},
        )

        assert entry.metadata == {"key": "value"}
