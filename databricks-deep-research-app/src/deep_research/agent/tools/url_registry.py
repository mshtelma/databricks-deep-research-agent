"""URL Registry for index-based URL handling.

Provides a mapping layer between numeric indices and actual URLs,
preventing the LLM from seeing or manipulating real URLs directly.

This is a security feature that ensures:
1. LLM only sees numeric indices like [0], [1], [2]
2. Actual URLs are hidden from the LLM
3. URL lookup happens internally when executing tools
4. Invalid indices are rejected (can't hallucinate URLs)
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class IndexedUrl:
    """A URL entry in the registry."""

    index: int
    url: str
    title: str
    snippet: str
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class UrlRegistry:
    """Thread-safe registry mapping indices to URLs.

    Per-session registry that hides actual URLs from the LLM.
    The LLM only sees numeric indices; URL lookup happens internally.

    Example usage:
        registry = UrlRegistry()

        # During search - register URLs and get indices
        for result in search_results:
            index = registry.register(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
            )
            # Format for LLM: "[{index}] {title}\n    {snippet}"

        # During crawl - resolve index to URL
        url = registry.get_url(index=0)
        if url is None:
            return "Error: Invalid index"
        content = await crawler.fetch(url)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._urls: dict[int, IndexedUrl] = {}
        self._url_to_index: dict[str, int] = {}  # Reverse lookup for deduplication
        self._next_index: int = 0
        self._lock = Lock()

    def register(
        self,
        url: str,
        title: str = "",
        snippet: str = "",
        relevance_score: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Register a URL and return its index.

        If URL already registered, returns existing index (deduplication).

        Args:
            url: The actual URL
            title: Page title from search results
            snippet: Search snippet/description
            relevance_score: Relevance score from search (0.0-1.0)
            metadata: Optional additional metadata

        Returns:
            Integer index for this URL
        """
        with self._lock:
            # Return existing index if URL already registered
            if url in self._url_to_index:
                return self._url_to_index[url]

            # Assign new index
            index = self._next_index
            self._next_index += 1

            entry = IndexedUrl(
                index=index,
                url=url,
                title=title,
                snippet=snippet,
                relevance_score=relevance_score,
                metadata=metadata or {},
            )

            self._urls[index] = entry
            self._url_to_index[url] = index

            return index

    def get_url(self, index: int) -> str | None:
        """Get URL by index.

        Args:
            index: Numeric index

        Returns:
            URL string or None if not found
        """
        with self._lock:
            entry = self._urls.get(index)
            return entry.url if entry else None

    def get_entry(self, index: int) -> IndexedUrl | None:
        """Get full entry by index.

        Args:
            index: Numeric index

        Returns:
            IndexedUrl or None if not found
        """
        with self._lock:
            return self._urls.get(index)

    def get_index(self, url: str) -> int | None:
        """Get index for a URL (reverse lookup).

        Args:
            url: URL to look up

        Returns:
            Index or None if URL not registered
        """
        with self._lock:
            return self._url_to_index.get(url)

    def list_all(self) -> list[IndexedUrl]:
        """Get all registered URLs.

        Returns:
            List of all IndexedUrl entries, sorted by index
        """
        with self._lock:
            return sorted(self._urls.values(), key=lambda x: x.index)

    def get_count(self) -> int:
        """Get number of registered URLs.

        Returns:
            Number of URLs in registry
        """
        with self._lock:
            return len(self._urls)

    def clear(self) -> None:
        """Clear all registered URLs."""
        with self._lock:
            self._urls.clear()
            self._url_to_index.clear()
            self._next_index = 0

    def __len__(self) -> int:
        """Return number of registered URLs."""
        return self.get_count()

    def __contains__(self, index: int) -> bool:
        """Check if index exists in registry."""
        with self._lock:
            return index in self._urls
