"""Unit tests for DiscoveryCache.

Tests for:
- Cache key generation per user (T010p)
- TTL expiration (T010p)
- Concurrent access (T010p)
- Invalidation behavior
- Statistics tracking
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from deep_research.schemas.data_source import DataSourceType
from deep_research.schemas.discovery import DiscoveredSource, DiscoveryStatus
from deep_research.services.discovery_cache import (
    CacheEntry,
    DEFAULT_CACHE_TTL,
    DiscoveryCache,
    REFRESH_BUFFER,
    get_discovery_cache,
    reset_discovery_cache,
)


@pytest.fixture
def cache() -> DiscoveryCache:
    """Create a fresh cache instance for testing."""
    return DiscoveryCache()


@pytest.fixture
def sample_source() -> DiscoveredSource:
    """Create a sample discovered source for testing."""
    return DiscoveredSource(
        source_id="vs:catalog.schema.test_index",
        source_type=DataSourceType.VECTOR_SEARCH,
        name="test_index",
        endpoint_name="test_endpoint",
        description="Test Vector Search index",
        status=DiscoveryStatus.READY,
        capabilities=["ann", "hybrid"],
        metadata={"index_name": "test_index"},
        discovered_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_sources(sample_source: DiscoveredSource) -> list[DiscoveredSource]:
    """Create multiple sample sources."""
    genie_source = DiscoveredSource(
        source_id="genie:space123",
        source_type=DataSourceType.GENIE,
        name="Test Genie Space",
        endpoint_name="space123",
        description="Test Genie space",
        status=DiscoveryStatus.READY,
        capabilities=["sql", "conversation"],
        metadata={"space_id": "space123"},
        discovered_at=datetime.now(UTC),
    )
    return [sample_source, genie_source]


class TestCacheKeyGeneration:
    """Tests for cache key generation per user."""

    def test_hash_token_consistent(self, cache: DiscoveryCache) -> None:
        """Test that hashing is consistent for same token."""
        token = "test-token-12345"
        hash1 = cache._hash_token(token)
        hash2 = cache._hash_token(token)
        assert hash1 == hash2
        assert len(hash1) == 16  # 16 char prefix

    def test_hash_token_different_for_different_tokens(self, cache: DiscoveryCache) -> None:
        """Test that different tokens produce different hashes."""
        hash1 = cache._hash_token("token-1")
        hash2 = cache._hash_token("token-2")
        assert hash1 != hash2

    def test_make_key_with_type(self, cache: DiscoveryCache) -> None:
        """Test key generation with source type."""
        token = "test-token"
        key = cache._make_key(user_token=token, source_type=DataSourceType.VECTOR_SEARCH)
        assert ":vector_search" in key
        assert "discovery:" in key

    def test_make_key_without_type(self, cache: DiscoveryCache) -> None:
        """Test key generation without source type."""
        token = "test-token"
        key = cache._make_key(user_token=token, source_type=None)
        assert ":all" in key

    def test_make_key_with_user_id(self, cache: DiscoveryCache) -> None:
        """Test key generation with user_id (preferred over token)."""
        user_id = "user-12345"
        key = cache._make_key(user_id=user_id)
        assert "discovery:" in key
        assert ":all" in key

    def test_make_key_user_id_takes_precedence(self, cache: DiscoveryCache) -> None:
        """Test that user_id takes precedence over user_token."""
        user_id = "user-12345"
        token = "test-token"
        key_with_both = cache._make_key(user_id=user_id, user_token=token)
        key_with_id_only = cache._make_key(user_id=user_id)
        assert key_with_both == key_with_id_only

    def test_make_key_requires_at_least_one_identifier(self, cache: DiscoveryCache) -> None:
        """Test that _make_key raises ValueError without user_id or user_token."""
        with pytest.raises(ValueError, match="Either user_id or user_token must be provided"):
            cache._make_key()

    def test_different_users_different_keys(self, cache: DiscoveryCache) -> None:
        """Test that different users get different cache keys."""
        key1 = cache._make_key(user_token="user1-token")
        key2 = cache._make_key(user_token="user2-token")
        assert key1 != key2

    def test_different_user_ids_different_keys(self, cache: DiscoveryCache) -> None:
        """Test that different user_ids get different cache keys."""
        key1 = cache._make_key(user_id="user1")
        key2 = cache._make_key(user_id="user2")
        assert key1 != key2


class TestCacheExpiration:
    """Tests for TTL expiration behavior."""

    @pytest.mark.asyncio
    async def test_entry_not_expired_initially(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test that a new entry is not expired."""
        user_id = "test-user-id"
        await cache.set(sources=sample_sources, user_id=user_id)

        cached = await cache.get(user_id=user_id)
        assert cached is not None
        assert len(cached) == len(sample_sources)

    @pytest.mark.asyncio
    async def test_entry_expires_after_ttl(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test that entry expires after TTL."""
        user_id = "test-user-id"

        # Set with very short TTL
        await cache.set(sources=sample_sources, user_id=user_id, ttl=timedelta(milliseconds=1))

        # Wait for expiration
        await asyncio.sleep(0.01)

        cached = await cache.get(user_id=user_id)
        assert cached is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test setting custom TTL."""
        user_id = "test-user-id"
        custom_ttl = timedelta(seconds=1)

        await cache.set(sources=sample_sources, user_id=user_id, ttl=custom_ttl)

        # Should be cached
        cached = await cache.get(user_id=user_id)
        assert cached is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        cached = await cache.get(user_id=user_id)
        assert cached is None

    def test_cache_entry_is_expired_property(self) -> None:
        """Test CacheEntry.is_expired property."""
        # Not expired
        entry = CacheEntry(
            sources=[],
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        assert not entry.is_expired

        # Expired
        expired_entry = CacheEntry(
            sources=[],
            created_at=datetime.now(UTC) - timedelta(minutes=10),
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )
        assert expired_entry.is_expired

    def test_cache_entry_should_refresh_property(self) -> None:
        """Test CacheEntry.should_refresh property."""
        # Far from expiry - no refresh needed
        entry = CacheEntry(
            sources=[],
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        assert not entry.should_refresh

        # Near expiry - refresh needed
        near_expiry_entry = CacheEntry(
            sources=[],
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),  # Less than REFRESH_BUFFER
        )
        assert near_expiry_entry.should_refresh


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test concurrent reads don't cause issues."""
        user_id = "test-user-id"
        await cache.set(sources=sample_sources, user_id=user_id)

        # Perform multiple concurrent reads
        tasks = [cache.get(user_id=user_id) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return the same data
        assert all(r is not None for r in results)
        assert all(len(r) == len(sample_sources) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test concurrent writes don't cause race conditions."""
        user_ids = [f"user-{i}" for i in range(10)]

        # Concurrent writes
        tasks = [cache.set(sources=sample_sources, user_id=uid) for uid in user_ids]
        await asyncio.gather(*tasks)

        # Verify all were written
        for uid in user_ids:
            cached = await cache.get(user_id=uid)
            assert cached is not None

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test concurrent reads and writes."""
        user_id = "test-user-id"

        async def reader() -> list[DiscoveredSource] | None:
            await asyncio.sleep(0.001)
            return await cache.get(user_id=user_id)

        async def writer() -> None:
            await cache.set(sources=sample_sources, user_id=user_id)

        # Start concurrent operations
        await asyncio.gather(
            writer(),
            reader(),
            reader(),
            writer(),
            reader(),
        )

        # Cache should have data
        cached = await cache.get(user_id=user_id)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_concurrent_invalidate(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test concurrent invalidations."""
        user_id = "test-user-id"
        await cache.set(sources=sample_sources, user_id=user_id)

        # Concurrent invalidations
        tasks = [cache.invalidate(user_id=user_id) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # First should return 1, rest should return 0
        assert sum(results) == 1

        # Cache should be empty
        cached = await cache.get(user_id=user_id)
        assert cached is None


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_specific_type(self, cache: DiscoveryCache, sample_source: DiscoveredSource) -> None:
        """Test invalidating specific source type."""
        user_id = "test-user-id"

        # Set entries for different types
        await cache.set(sources=[sample_source], user_id=user_id, source_type=DataSourceType.VECTOR_SEARCH)

        genie_source = DiscoveredSource(
            source_id="genie:test",
            source_type=DataSourceType.GENIE,
            name="Test",
            endpoint_name="test",
            status=DiscoveryStatus.READY,
            capabilities=[],
            metadata={},
            discovered_at=datetime.now(UTC),
        )
        await cache.set(sources=[genie_source], user_id=user_id, source_type=DataSourceType.GENIE)

        # Invalidate only VS
        count = await cache.invalidate(user_id=user_id, source_type=DataSourceType.VECTOR_SEARCH)
        assert count == 1

        # VS should be gone, Genie should remain
        vs_cached = await cache.get(user_id=user_id, source_type=DataSourceType.VECTOR_SEARCH)
        assert vs_cached is None

        genie_cached = await cache.get(user_id=user_id, source_type=DataSourceType.GENIE)
        assert genie_cached is not None

    @pytest.mark.asyncio
    async def test_invalidate_all_for_user(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test invalidating all entries for a user."""
        user_id = "test-user-id"

        await cache.set(sources=sample_sources, user_id=user_id)
        await cache.set(sources=[sample_sources[0]], user_id=user_id, source_type=DataSourceType.VECTOR_SEARCH)

        count = await cache.invalidate(user_id=user_id)
        assert count == 2

        # All should be gone
        assert await cache.get(user_id=user_id) is None
        assert await cache.get(user_id=user_id, source_type=DataSourceType.VECTOR_SEARCH) is None

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test invalidating entire cache."""
        user_ids = ["user1", "user2", "user3"]

        for uid in user_ids:
            await cache.set(sources=sample_sources, user_id=uid)

        count = await cache.invalidate_all()
        assert count == 3

        for uid in user_ids:
            assert await cache.get(user_id=uid) is None

    @pytest.mark.asyncio
    async def test_invalidate_requires_identifier(self, cache: DiscoveryCache) -> None:
        """Test that invalidate raises ValueError without user_id or user_token."""
        with pytest.raises(ValueError, match="Either user_id or user_token must be provided"):
            await cache.invalidate()


class TestCacheStatistics:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, cache: DiscoveryCache) -> None:
        """Test stats for empty cache."""
        stats = await cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["active_entries"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_entries(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test stats with entries."""
        await cache.set(sources=sample_sources, user_id="user1")
        await cache.set(sources=sample_sources, user_id="user2", source_type=DataSourceType.VECTOR_SEARCH)

        stats = await cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert "all" in stats["by_source_type"]
        assert "vector_search" in stats["by_source_type"]

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache: DiscoveryCache, sample_sources: list[DiscoveredSource]) -> None:
        """Test cleanup of expired entries."""
        # Add entry with very short TTL
        await cache.set(sources=sample_sources, user_id="user1", ttl=timedelta(milliseconds=1))

        # Add entry with long TTL
        await cache.set(sources=sample_sources, user_id="user2", ttl=timedelta(minutes=5))

        # Wait for first to expire
        await asyncio.sleep(0.01)

        # Cleanup
        removed = await cache.cleanup_expired()
        assert removed == 1

        # Verify
        assert await cache.get(user_id="user1") is None
        assert await cache.get(user_id="user2") is not None


class TestGlobalCacheInstance:
    """Tests for global cache singleton."""

    def test_get_discovery_cache_singleton(self) -> None:
        """Test that get_discovery_cache returns singleton."""
        reset_discovery_cache()

        cache1 = get_discovery_cache()
        cache2 = get_discovery_cache()

        assert cache1 is cache2

        reset_discovery_cache()

    def test_reset_discovery_cache(self) -> None:
        """Test resetting the global cache."""
        reset_discovery_cache()

        cache1 = get_discovery_cache()
        reset_discovery_cache()
        cache2 = get_discovery_cache()

        assert cache1 is not cache2

        reset_discovery_cache()
