"""
Intelligent caching system for expensive operations.

This module provides comprehensive caching for search results, embeddings,
LLM responses, and other expensive operations to improve performance and
reduce API costs.
"""

import hashlib
import json
import time
import pickle
import threading
from typing import Any, Optional, Callable, Dict, List, Union, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from collections import OrderedDict
from enum import Enum
import logging

from deep_research_agent.core.error_handler import safe_call
from deep_research_agent.core.search_provider import SearchResult

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Dynamic strategy based on access patterns


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with comprehensive metadata."""
    
    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    ttl: float = 900.0  # 15 minutes default
    hit_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate entry size and normalize data."""
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        if self.ttl <= 0:  # Never expires
            return True
        return time.time() - self.created_at < self.ttl
    
    def is_stale(self, staleness_threshold: float = 300.0) -> bool:
        """Check if entry is stale but still valid."""
        age = time.time() - self.created_at
        return self.ttl > 0 and age > (self.ttl - staleness_threshold)
    
    def access(self):
        """Record access to this entry."""
        self.accessed_at = time.time()
        self.hit_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    def get_last_access_age(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.accessed_at
    
    def _calculate_size(self) -> int:
        """Estimate memory size of the cached value."""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (list, dict)):
                return len(json.dumps(self.value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(self.value))
        except Exception:
            return 1000  # Default size estimate


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    
    max_size: int = 10000  # Maximum number of entries
    max_memory_mb: int = 500  # Maximum memory usage in MB
    default_ttl: float = 900.0  # Default TTL in seconds
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    
    # Persistence settings
    enable_persistence: bool = False
    persist_path: Optional[Path] = None
    persist_interval: float = 300.0  # Persist every 5 minutes
    
    # Performance settings
    cleanup_interval: float = 60.0  # Cleanup every minute
    staleness_threshold: float = 300.0  # 5 minutes
    
    # Category-specific TTLs
    ttl_overrides: Dict[str, float] = field(default_factory=lambda: {
        "search_results": 3600.0,  # 1 hour
        "embeddings": 7200.0,      # 2 hours
        "llm_responses": 1800.0,   # 30 minutes
        "api_responses": 900.0     # 15 minutes
    })


class IntelligentCache:
    """
    Intelligent caching system with multiple strategies and persistence.
    
    Features:
    - Multiple eviction strategies (LRU, LFU, adaptive)
    - TTL-based expiration
    - Memory usage tracking
    - Persistence support
    - Category-specific configurations
    - Performance metrics
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
        # Category-specific caches for better organization
        self.category_caches: Dict[str, OrderedDict] = {}
        
        # Metrics tracking
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_usage_bytes": 0,
            "cleanup_runs": 0,
            "persistence_saves": 0,
            "persistence_loads": 0
        }
        
        # Access pattern tracking for adaptive strategy
        self.access_patterns: Dict[str, List[float]] = {}
        
        # Start background tasks
        self._start_background_tasks()
        
        # Load from persistence if enabled
        if self.config.enable_persistence:
            self._load_from_persistence()
    
    def get(self, key: str, category: str = "default") -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            category: Cache category for organization
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            cache_key = self._make_cache_key(key, category)
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if entry.is_valid():
                    entry.access()
                    self.metrics["hits"] += 1
                    
                    # Move to end for LRU
                    if self.config.strategy == CacheStrategy.LRU:
                        self.cache.move_to_end(cache_key)
                    
                    # Track access patterns
                    self._track_access(cache_key)
                    
                    logger.debug(f"Cache hit for key: {key} (category: {category})")
                    return entry.value
                else:
                    # Expired entry
                    del self.cache[cache_key]
                    self._update_memory_usage()
            
            self.metrics["misses"] += 1
            logger.debug(f"Cache miss for key: {key} (category: {category})")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        category: str = "default",
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            category: Cache category
            ttl: Time to live (overrides default)
            metadata: Additional metadata
        """
        with self.lock:
            cache_key = self._make_cache_key(key, category)
            
            # Determine TTL
            if ttl is None:
                ttl = self.config.ttl_overrides.get(category, self.config.default_ttl)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                ttl=ttl,
                metadata=metadata or {}
            )
            
            # Check if we need to evict before adding
            self._ensure_capacity(entry.size_bytes)
            
            # Add to cache
            self.cache[cache_key] = entry
            self._update_memory_usage()
            
            # Organize by category
            if category not in self.category_caches:
                self.category_caches[category] = OrderedDict()
            self.category_caches[category][cache_key] = entry
            
            logger.debug(f"Cached value for key: {key} (category: {category}, ttl: {ttl}s)")
    
    def delete(self, key: str, category: str = "default") -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            category: Cache category
            
        Returns:
            True if entry was deleted
        """
        with self.lock:
            cache_key = self._make_cache_key(key, category)
            
            if cache_key in self.cache:
                del self.cache[cache_key]
                
                if category in self.category_caches and cache_key in self.category_caches[category]:
                    del self.category_caches[category][cache_key]
                
                self._update_memory_usage()
                return True
            
            return False
    
    def clear(self, category: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            category: If specified, only clear this category
        """
        with self.lock:
            if category:
                # Clear specific category
                if category in self.category_caches:
                    keys_to_remove = list(self.category_caches[category].keys())
                    for key in keys_to_remove:
                        if key in self.cache:
                            del self.cache[key]
                    del self.category_caches[category]
            else:
                # Clear all
                self.cache.clear()
                self.category_caches.clear()
            
            self._update_memory_usage()
            logger.info(f"Cache cleared: {category or 'all categories'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_requests = self.metrics["hits"] + self.metrics["misses"]
            hit_rate = (self.metrics["hits"] / total_requests) if total_requests > 0 else 0.0
            
            # Category stats
            category_stats = {}
            for category, cat_cache in self.category_caches.items():
                category_stats[category] = {
                    "entries": len(cat_cache),
                    "memory_estimate": sum(
                        entry.size_bytes for entry in cat_cache.values()
                    )
                }
            
            return {
                "total_entries": len(self.cache),
                "hit_rate": hit_rate,
                "memory_usage_mb": self.metrics["memory_usage_bytes"] / (1024 * 1024),
                "category_stats": category_stats,
                **self.metrics
            }
    
    def cache_result(
        self, 
        category: str = "default",
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator to cache function results.
        
        Args:
            category: Cache category
            ttl: Time to live
            key_func: Custom key generation function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._make_function_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self.get(cache_key, category)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, category, ttl)
                return result
            
            return wrapper
        return decorator
    
    def _make_cache_key(self, key: str, category: str) -> str:
        """Create cache key with category prefix."""
        return f"{category}:{key}"
    
    def _make_function_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function call."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Check entry count limit
        while len(self.cache) >= self.config.max_size:
            self._evict_entry()
        
        # Check memory limit
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        while (self.metrics["memory_usage_bytes"] + new_entry_size) > max_memory_bytes:
            self._evict_entry()
    
    def _evict_entry(self):
        """Evict entry based on configured strategy."""
        if not self.cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self.cache))
        elif self.config.strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            key = next(iter(self.cache))
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].hit_count)
        else:  # ADAPTIVE
            key = self._adaptive_eviction()
        
        if key in self.cache:
            del self.cache[key]
            self.metrics["evictions"] += 1
            self._update_memory_usage()
    
    def _adaptive_eviction(self) -> str:
        """Intelligent eviction based on access patterns and value."""
        # Score entries based on multiple factors
        scores = {}
        
        for key, entry in self.cache.items():
            # Factors: recency, frequency, age, size
            recency_score = 1.0 / (entry.get_last_access_age() + 1)
            frequency_score = entry.hit_count / 10.0  # Normalize
            age_penalty = entry.get_age() / entry.ttl if entry.ttl > 0 else 0
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Combined score (higher = keep)
            scores[key] = (recency_score + frequency_score) - (age_penalty + size_penalty)
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _track_access(self, key: str):
        """Track access patterns for adaptive caching."""
        now = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(now)
        
        # Keep only recent accesses (last hour)
        hour_ago = now - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > hour_ago
        ]
    
    def _update_memory_usage(self):
        """Update memory usage metrics."""
        self.metrics["memory_usage_bytes"] = sum(
            entry.size_bytes for entry in self.cache.values()
        )
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if not entry.is_valid():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self._update_memory_usage()
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            self.metrics["cleanup_runs"] += 1
    
    def _start_background_tasks(self):
        """Start background cleanup and persistence tasks."""
        def cleanup_task():
            while True:
                time.sleep(self.config.cleanup_interval)
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup failed: {e}")
        
        def persistence_task():
            while self.config.enable_persistence:
                time.sleep(self.config.persist_interval)
                try:
                    self._save_to_persistence()
                except Exception as e:
                    logger.error(f"Cache persistence failed: {e}")
        
        # Start daemon threads
        threading.Thread(target=cleanup_task, daemon=True).start()
        
        if self.config.enable_persistence:
            threading.Thread(target=persistence_task, daemon=True).start()
    
    @safe_call("save_to_persistence", fallback=None)
    def _save_to_persistence(self):
        """Save cache to persistent storage."""
        if not self.config.persist_path:
            return
        
        persist_data = {
            "cache": {
                key: {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "ttl": entry.ttl,
                    "hit_count": entry.hit_count,
                    "metadata": entry.metadata
                }
                for key, entry in self.cache.items()
                if entry.is_valid()  # Only save valid entries
            },
            "metrics": self.metrics
        }
        
        with open(self.config.persist_path, 'wb') as f:
            pickle.dump(persist_data, f)
        
        self.metrics["persistence_saves"] += 1
        logger.debug("Cache saved to persistence")
    
    @safe_call("load_from_persistence", fallback=None)
    def _load_from_persistence(self):
        """Load cache from persistent storage."""
        if not self.config.persist_path or not self.config.persist_path.exists():
            return
        
        try:
            with open(self.config.persist_path, 'rb') as f:
                persist_data = pickle.load(f)
            
            # Restore cache entries
            for key, entry_data in persist_data.get("cache", {}).items():
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    ttl=entry_data["ttl"],
                    hit_count=entry_data["hit_count"],
                    metadata=entry_data.get("metadata", {})
                )
                
                if entry.is_valid():
                    self.cache[key] = entry
            
            # Restore metrics
            if "metrics" in persist_data:
                self.metrics.update(persist_data["metrics"])
            
            self.metrics["persistence_loads"] += 1
            self._update_memory_usage()
            
            logger.info(f"Loaded {len(self.cache)} entries from cache persistence")
            
        except Exception as e:
            logger.error(f"Failed to load cache from persistence: {e}")


class CategoryCache:
    """Category-specific cache interface for easier usage."""
    
    def __init__(self, cache: IntelligentCache, category: str):
        self.cache = cache
        self.category = category
    
    def get(self, key: str) -> Optional[Any]:
        """Get from category cache."""
        return self.cache.get(key, self.category)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set in category cache."""
        self.cache.set(key, value, self.category, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from category cache."""
        return self.cache.delete(key, self.category)
    
    def clear(self):
        """Clear category cache."""
        self.cache.clear(self.category)
    
    def cache_result(self, ttl: Optional[float] = None):
        """Category-specific cache decorator."""
        return self.cache.cache_result(self.category, ttl)


# Global cache manager instance
_global_cache: Optional[IntelligentCache] = None

def get_global_cache() -> IntelligentCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        config = CacheConfig(
            max_size=5000,
            max_memory_mb=200,
            default_ttl=900.0
        )
        _global_cache = IntelligentCache(config)
    return _global_cache

def get_category_cache(category: str) -> CategoryCache:
    """Get category-specific cache interface."""
    return CategoryCache(get_global_cache(), category)

# Convenience cache instances
search_cache = get_category_cache("search_results")
embedding_cache = get_category_cache("embeddings")
llm_cache = get_category_cache("llm_responses")
api_cache = get_category_cache("api_responses")

# Global cache manager instance for direct access
global_cache_manager = get_global_cache()