"""Calculation cache with intelligent invalidation."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from .. import get_logger


logger = get_logger(__name__)


class CacheEntry(BaseModel):
    """Single cache entry with metadata."""
    
    value: Any
    timestamp: float
    dependencies: List[str] = Field(default_factory=list)
    computation_time: float = 0.0
    hit_count: int = 0


class CacheStats(BaseModel):
    """Cache performance statistics."""
    
    total_hits: int = 0
    total_misses: int = 0
    unique_keys: int = 0
    hit_rate: float = 0.0
    avg_computation_time: float = 0.0
    memory_usage_mb: float = 0.0
    oldest_entry_age: float = 0.0


class CalculationCache:
    """Cache calculation results with intelligent invalidation.
    
    Features:
    - TTL-based expiration
    - Dependency-aware invalidation (cascade invalidation)
    - Access pattern tracking for optimization
    - Memory usage monitoring
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_size: Maximum number of entries (default 10,000)
        """
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, int] = defaultdict(int)
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, dependencies: Optional[List[str]] = None) -> Optional[Any]:
        """Get value from cache if valid.
        
        Args:
            key: Cache key
            dependencies: List of dependency keys to check
        
        Returns:
            Cached value or None if not found/invalid
        """
        if key not in self.cache:
            self._misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if entry is still valid
        if not self._is_valid(entry, dependencies):
            del self.cache[key]
            self._misses += 1
            return None
        
        # Update access patterns
        entry.hit_count += 1
        self.access_patterns[key] += 1
        self._hits += 1
        
        logger.debug(f"[CACHE HIT] Key: {key}, hits: {entry.hit_count}")
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        dependencies: Optional[List[str]] = None,
        computation_time: float = 0.0
    ) -> None:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            dependencies: List of dependency keys
            computation_time: Time taken to compute value
        """
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store new entry
        self.cache[key] = CacheEntry(
            value=value,
            timestamp=time.time(),
            dependencies=dependencies or [],
            computation_time=computation_time
        )
        
        logger.debug(f"[CACHE SET] Key: {key}, deps: {dependencies}")
    
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        dependencies: Optional[List[str]] = None
    ) -> Any:
        """Get cached result or compute if missing/stale.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            dependencies: List of dependency keys
        
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached_value = self.get(key, dependencies)
        if cached_value is not None:
            return cached_value
        
        # Compute and cache
        start_time = time.time()
        result = compute_func()
        computation_time = time.time() - start_time
        
        self.set(key, result, dependencies, computation_time)
        return result
    
    async def get_or_compute_async(
        self,
        key: str,
        compute_func: Callable,
        dependencies: Optional[List[str]] = None
    ) -> Any:
        """Async version of get_or_compute.
        
        Args:
            key: Cache key
            compute_func: Async function to compute value if not cached
            dependencies: List of dependency keys
        
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached_value = self.get(key, dependencies)
        if cached_value is not None:
            return cached_value
        
        # Compute and cache
        start_time = time.time()
        result = await compute_func()
        computation_time = time.time() - start_time
        
        self.set(key, result, dependencies, computation_time)
        return result
    
    def invalidate(self, key: str) -> None:
        """Invalidate a single cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"[CACHE INVALIDATE] Key: {key}")
    
    def invalidate_cascade(self, key: str) -> int:
        """Invalidate key and all dependent calculations.
        
        Args:
            key: Cache key to invalidate
        
        Returns:
            Number of entries invalidated
        """
        to_invalidate = {key}
        
        # Find all dependent calculations
        # An entry is dependent if it lists the key in its dependencies
        for cache_key, entry in list(self.cache.items()):
            if key in entry.dependencies:
                to_invalidate.add(cache_key)
        
        # Remove all affected entries
        for invalid_key in to_invalidate:
            if invalid_key in self.cache:
                del self.cache[invalid_key]
        
        logger.info(f"[CACHE INVALIDATE CASCADE] Invalidated {len(to_invalidate)} entries starting from {key}")
        return len(to_invalidate)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
        
        Returns:
            Number of entries invalidated
        """
        import re
        
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*')
        regex = re.compile(f'^{regex_pattern}$')
        
        # Find matching keys
        matching_keys = [key for key in self.cache.keys() if regex.match(key)]
        
        # Invalidate all matches
        for key in matching_keys:
            del self.cache[key]
        
        logger.info(f"[CACHE INVALIDATE PATTERN] Pattern '{pattern}' invalidated {len(matching_keys)} entries")
        return len(matching_keys)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        self.access_patterns.clear()
        self._hits = 0
        self._misses = 0
        logger.info(f"[CACHE CLEAR] Cleared {count} entries")
    
    def get_cache_stats(self) -> CacheStats:
        """Return cache performance statistics.
        
        Returns:
            CacheStats with performance metrics
        """
        if not self.cache:
            return CacheStats()
        
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        computation_times = [e.computation_time for e in self.cache.values() if e.computation_time > 0]
        avg_computation_time = sum(computation_times) / len(computation_times) if computation_times else 0.0
        
        # Estimate memory usage (rough approximation)
        import sys
        memory_usage_mb = sys.getsizeof(self.cache) / (1024 * 1024)
        
        # Find oldest entry
        current_time = time.time()
        oldest_age = max(current_time - e.timestamp for e in self.cache.values()) if self.cache else 0.0
        
        return CacheStats(
            total_hits=self._hits,
            total_misses=self._misses,
            unique_keys=len(self.cache),
            hit_rate=hit_rate,
            avg_computation_time=avg_computation_time,
            memory_usage_mb=memory_usage_mb,
            oldest_entry_age=oldest_age
        )
    
    def get_hot_keys(self, top_n: int = 10) -> List[tuple[str, int]]:
        """Get most frequently accessed cache keys.
        
        Args:
            top_n: Number of top keys to return
        
        Returns:
            List of (key, access_count) tuples
        """
        sorted_keys = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_keys[:top_n]
    
    def _is_valid(self, entry: CacheEntry, dependencies: Optional[List[str]]) -> bool:
        """Check if cached entry is still valid.
        
        Args:
            entry: Cache entry to check
            dependencies: Expected dependencies
        
        Returns:
            True if entry is valid, False otherwise
        """
        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            logger.debug("[CACHE] Entry expired (TTL)")
            return False
        
        # Check if dependencies changed
        if dependencies:
            for dep in dependencies:
                if dep in self.cache:
                    dep_entry = self.cache[dep]
                    # If dependency was updated after this entry, invalidate
                    if dep_entry.timestamp > entry.timestamp:
                        logger.debug(f"[CACHE] Entry invalidated (dependency {dep} updated)")
                        return False
        
        return True
    
    def _evict_lru(self, count: int = 1) -> None:
        """Evict least recently used entries.
        
        Args:
            count: Number of entries to evict
        """
        if not self.cache:
            return
        
        # Find entries with lowest access count
        sorted_entries = sorted(
            [(k, v.hit_count) for k, v in self.cache.items()],
            key=lambda x: x[1]
        )
        
        # Evict the least accessed entries
        for key, _ in sorted_entries[:count]:
            del self.cache[key]
            logger.debug(f"[CACHE EVICT] Key: {key}")


__all__ = ["CalculationCache", "CacheStats", "CacheEntry"]

