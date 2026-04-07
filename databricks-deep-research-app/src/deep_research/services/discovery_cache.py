"""Discovery cache for data source discovery results.

This module implements a thread-safe, TTL-based cache for discovered data sources.
The cache is per-user with configurable TTL.

Key Features:
- 5-minute default TTL (configurable)
- Per-user cache keys (user_id preferred, token fallback for security)
- Async-safe with asyncio.Lock
- Background refresh on near-expiry
- Partial results support (cache per source type)

Security Considerations:
- User IDs and tokens are hashed before use as cache keys
- Cache entries store only discovery results, not tokens
- Hash uses SHA-256 (same pattern as OBODatabricksClient)

Cache Key Priority:
- user_id is preferred (stable identifier, works in local dev)
- user_token as fallback (for OBO scenarios where only token is available)
"""

import asyncio
import hashlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from deep_research.core.logging_utils import get_logger
from deep_research.schemas.data_source import DataSourceType
from deep_research.schemas.discovery import DiscoveredSource

logger = get_logger(__name__)

# Default cache configuration
DEFAULT_CACHE_TTL = timedelta(minutes=5)
REFRESH_BUFFER = timedelta(minutes=1)  # Start refresh when this close to expiry


@dataclass
class CacheEntry:
    """A single cache entry containing discovered sources."""

    sources: list[DiscoveredSource]
    """The discovered sources."""

    created_at: datetime
    """When the entry was created."""

    expires_at: datetime
    """When the entry expires."""

    source_type: DataSourceType | None = None
    """Source type if this is a type-specific entry. None = all types."""

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return datetime.now(UTC) >= self.expires_at

    @property
    def should_refresh(self) -> bool:
        """Check if the entry should be refreshed (near expiry)."""
        return datetime.now(UTC) >= (self.expires_at - REFRESH_BUFFER)


@dataclass
class DiscoveryCache:
    """Thread-safe cache for discovered data sources.

    The cache stores discovery results per-user with TTL-based expiration.
    Keys are generated from hashed user tokens and optional source type.

    Example:
        cache = DiscoveryCache()

        # Store discovered sources
        await cache.set(user_token, sources)

        # Retrieve cached sources
        cached = await cache.get(user_token)

        # Get with specific source type
        vs_sources = await cache.get(user_token, DataSourceType.VECTOR_SEARCH)

        # Invalidate cache
        await cache.invalidate(user_token)
    """

    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    """Cache storage: key -> CacheEntry."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Async lock for thread-safe operations."""

    _ttl: timedelta = DEFAULT_CACHE_TTL
    """Time-to-live for cache entries."""

    _refresh_callbacks: dict[str, Callable[[], Coroutine[None, None, list[DiscoveredSource]]]] = field(
        default_factory=dict
    )
    """Callbacks for background refresh."""

    def _hash_token(self, token: str) -> str:
        """Create a hash of token for cache key (security: avoid storing raw token).

        Uses same hashing approach as OBODatabricksClient for consistency.

        Args:
            token: The user's OAuth token.

        Returns:
            16-character hash prefix.
        """
        return hashlib.sha256(token.encode()).hexdigest()[:16]

    def _make_key(
        self,
        user_id: str | None = None,
        user_token: str | None = None,
        source_type: DataSourceType | None = None,
    ) -> str:
        """Generate cache key from user_id or user_token.

        Priority: user_id > user_token (user_id is more stable and works in local dev).

        Args:
            user_id: User's ID (preferred, always available after auth).
            user_token: User's OAuth token (fallback, will be hashed).
            source_type: Optional source type for type-specific caching.

        Returns:
            Cache key string.

        Raises:
            ValueError: If neither user_id nor user_token is provided.
        """
        if user_id:
            # Use user_id directly (already a stable identifier)
            id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        elif user_token:
            # Fall back to token hash
            id_hash = self._hash_token(user_token)
        else:
            raise ValueError("Either user_id or user_token must be provided")

        type_suffix = f":{source_type.value}" if source_type else ":all"
        return f"discovery:{id_hash}{type_suffix}"

    async def get(
        self,
        user_id: str | None = None,
        user_token: str | None = None,
        source_type: DataSourceType | None = None,
    ) -> list[DiscoveredSource] | None:
        """Get cached discovered sources for a user.

        Args:
            user_id: User's ID (preferred for cache key).
            user_token: User's OAuth token (fallback for cache key).
            source_type: Optional filter by source type. None = get all.

        Returns:
            List of discovered sources if cached and not expired, None otherwise.
        """
        key = self._make_key(user_id=user_id, user_token=user_token, source_type=source_type)

        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                logger.debug(
                    "DISCOVERY_CACHE_MISS",
                    key_suffix=key[-20:],  # Log only suffix for security
                    source_type=source_type.value if source_type else "all",
                )
                return None

            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                logger.debug(
                    "DISCOVERY_CACHE_EXPIRED",
                    key_suffix=key[-20:],
                    source_type=source_type.value if source_type else "all",
                )
                return None

            # Check if background refresh needed
            if entry.should_refresh:
                logger.debug(
                    "DISCOVERY_CACHE_REFRESH_NEEDED",
                    key_suffix=key[-20:],
                    expires_in_seconds=(entry.expires_at - datetime.now(UTC)).total_seconds(),
                )
                # Could trigger background refresh here if callback registered

            logger.debug(
                "DISCOVERY_CACHE_HIT",
                key_suffix=key[-20:],
                source_type=source_type.value if source_type else "all",
                source_count=len(entry.sources),
            )
            return entry.sources

    async def set(
        self,
        sources: list[DiscoveredSource],
        user_id: str | None = None,
        user_token: str | None = None,
        source_type: DataSourceType | None = None,
        ttl: timedelta | None = None,
    ) -> None:
        """Cache discovered sources for a user.

        Args:
            sources: List of discovered sources to cache.
            user_id: User's ID (preferred for cache key).
            user_token: User's OAuth token (fallback for cache key).
            source_type: Source type if this is a type-specific entry.
            ttl: Optional custom TTL. Uses default if not provided.
        """
        key = self._make_key(user_id=user_id, user_token=user_token, source_type=source_type)
        effective_ttl = ttl or self._ttl
        now = datetime.now(UTC)

        entry = CacheEntry(
            sources=sources,
            created_at=now,
            expires_at=now + effective_ttl,
            source_type=source_type,
        )

        async with self._lock:
            self._cache[key] = entry

        logger.info(
            "DISCOVERY_CACHE_SET",
            key_suffix=key[-20:],
            source_type=source_type.value if source_type else "all",
            source_count=len(sources),
            ttl_seconds=effective_ttl.total_seconds(),
        )

    async def invalidate(
        self,
        user_id: str | None = None,
        user_token: str | None = None,
        source_type: DataSourceType | None = None,
    ) -> int:
        """Invalidate cache entries for a user.

        Args:
            user_id: User's ID (preferred for cache key).
            user_token: User's OAuth token (fallback for cache key).
            source_type: If provided, only invalidate this source type.
                        If None, invalidate all entries for the user.

        Returns:
            Number of entries invalidated.

        Raises:
            ValueError: If neither user_id nor user_token is provided.
        """
        # Generate the hash for matching keys
        if user_id:
            id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        elif user_token:
            id_hash = self._hash_token(user_token)
        else:
            raise ValueError("Either user_id or user_token must be provided")

        async with self._lock:
            if source_type:
                # Invalidate specific type
                key = self._make_key(user_id=user_id, user_token=user_token, source_type=source_type)
                if key in self._cache:
                    del self._cache[key]
                    logger.info(
                        "DISCOVERY_CACHE_INVALIDATED",
                        source_type=source_type.value,
                        count=1,
                    )
                    return 1
                return 0

            # Invalidate all entries for user
            keys_to_remove = [k for k in self._cache if f"discovery:{id_hash}" in k]
            for key in keys_to_remove:
                del self._cache[key]

            logger.info(
                "DISCOVERY_CACHE_INVALIDATED",
                source_type="all",
                count=len(keys_to_remove),
            )
            return len(keys_to_remove)

    async def invalidate_all(self) -> int:
        """Invalidate all cache entries.

        Returns:
            Number of entries invalidated.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()

        logger.info("DISCOVERY_CACHE_CLEARED", count=count)
        return count

    async def get_stats(self) -> dict[str, int | dict[str, int] | float]:
        """Get cache statistics.

        Returns:
            Dict with cache stats (total entries, expired, by type).
        """
        async with self._lock:
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if e.is_expired)
            by_type: dict[str, int] = {}

            for entry in self._cache.values():
                type_key = entry.source_type.value if entry.source_type else "all"
                by_type[type_key] = by_type.get(type_key, 0) + 1

            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "by_source_type": by_type,
                "ttl_seconds": self._ttl.total_seconds(),
            }

    async def cleanup_expired(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            keys_to_remove = [k for k, e in self._cache.items() if e.is_expired]
            for key in keys_to_remove:
                del self._cache[key]

        if keys_to_remove:
            logger.info("DISCOVERY_CACHE_CLEANUP", removed=len(keys_to_remove))

        return len(keys_to_remove)


# Global cache instance (singleton pattern)
_discovery_cache: DiscoveryCache | None = None


def get_discovery_cache() -> DiscoveryCache:
    """Get the global discovery cache instance.

    Returns:
        The singleton DiscoveryCache instance.
    """
    global _discovery_cache
    if _discovery_cache is None:
        _discovery_cache = DiscoveryCache()
    return _discovery_cache


def reset_discovery_cache() -> None:
    """Reset the global discovery cache (for testing)."""
    global _discovery_cache
    _discovery_cache = None
