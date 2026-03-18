"""Unit tests for verification cache in ResearchState.

Tests the TOKEN OPTIMIZATION verification cache functionality.
"""

import pytest

from deep_research.agent.state import ResearchState


class TestVerificationCache:
    """Tests for the verification cache in ResearchState."""

    def test_cache_starts_empty(self) -> None:
        """Test cache is empty on new state."""
        state = ResearchState(query="test query")
        assert state.get_verification_cache() == {}

    def test_cache_verification(self) -> None:
        """Test caching a verification result."""
        state = ResearchState(query="test query")
        fingerprint = "abc123def456gh78"
        result = {"verdict": "SUPPORTED", "reasoning": "Match"}

        state.cache_verification(fingerprint, result)

        assert state.get_cached_verification(fingerprint) == result

    def test_get_cached_missing(self) -> None:
        """Test getting a missing cached result returns None."""
        state = ResearchState(query="test query")
        result = state.get_cached_verification("nonexistent")
        assert result is None

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        state = ResearchState(query="test query")
        state.cache_verification("key1", {"verdict": "SUPPORTED"})
        state.cache_verification("key2", {"verdict": "PARTIAL"})

        assert len(state.get_verification_cache()) == 2

        state.clear_verification_cache()

        assert len(state.get_verification_cache()) == 0

    def test_get_cache_stats(self) -> None:
        """Test getting cache statistics."""
        state = ResearchState(query="test query")
        state.cache_verification("key1", {"verdict": "SUPPORTED"})
        state.cache_verification("key2", {"verdict": "PARTIAL"})
        state.cache_verification("key3", {"verdict": "UNSUPPORTED"})

        stats = state.get_verification_cache_stats()

        assert stats["cache_size"] == 3

    def test_cache_overwrites_same_key(self) -> None:
        """Test caching with same key overwrites."""
        state = ResearchState(query="test query")
        state.cache_verification("key1", {"verdict": "SUPPORTED"})
        state.cache_verification("key1", {"verdict": "PARTIAL"})

        result = state.get_cached_verification("key1")
        assert result == {"verdict": "PARTIAL"}

    def test_cache_isolate_between_states(self) -> None:
        """Test caches are isolated between different states."""
        state1 = ResearchState(query="query 1")
        state2 = ResearchState(query="query 2")

        state1.cache_verification("key1", {"verdict": "SUPPORTED"})

        # state2 should not have state1's cached result
        assert state2.get_cached_verification("key1") is None

    def test_get_verification_cache_returns_dict(self) -> None:
        """Test get_verification_cache returns the actual dict for batch use."""
        state = ResearchState(query="test query")
        state.cache_verification("key1", {"verdict": "SUPPORTED"})

        cache = state.get_verification_cache()

        # Should be the same dict instance
        cache["key2"] = {"verdict": "PARTIAL"}
        assert state.get_cached_verification("key2") == {"verdict": "PARTIAL"}
