"""
Global search coordinator to ensure proper spacing between searches from any source.
Implements singleton pattern with thread-safe coordination.
"""

import time
import threading
from typing import List, Dict, Any
from .logging import get_logger

logger = get_logger(__name__)


class SearchCoordinator:
    """
    Coordinates all search requests across the system to prevent rate limiting.
    
    Features:
    - Global coordination across all search components
    - Minimum delay enforcement between ANY searches
    - Thread-safe operations
    - Comprehensive logging
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the search coordinator (only once due to singleton)."""
        if self._initialized:
            return
        
        self._initialized = True
        self._last_search_time = 0
        self._search_count = 0
        self._coordination_lock = threading.Lock()
        
        # Load rate limit configuration
        self._min_delay = self._load_min_delay()
        
        logger.info(f"SearchCoordinator initialized with {self._min_delay}s minimum delay between searches")
    
    def _load_min_delay(self) -> float:
        """Load minimum delay from configuration based on rate_limit setting."""
        try:
            import yaml
            from pathlib import Path
            
            config_paths = [
                Path(__file__).parent.parent.parent / "conf" / "base.yaml",
                Path.cwd() / "conf" / "base.yaml",
                Path(__file__).parent.parent / "agent_config.yaml",
            ]
            
            for path in config_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        rate_limit = config.get("search", {}).get("providers", {}).get("brave", {}).get("rate_limit")
                        if rate_limit:
                            # rate_limit is requests per second, so delay = 1/rate_limit
                            delay = 1.0 / rate_limit
                            logger.info(f"Using rate_limit {rate_limit} req/sec -> {delay:.2f}s delay")
                            return delay
            
            # Default: 1 request per second
            logger.info("No rate_limit config found, using default 1.0s delay")
            return 1.0
            
        except Exception as e:
            logger.warning(f"Failed to load rate_limit config: {e}, using default 1.0s delay")
            return 1.0
    
    def coordinate_search(self, source: str, query: str) -> bool:
        """
        Ensure proper spacing between searches from any source.
        
        Args:
            source: Name of the component making the search (e.g., "background_investigation", "researcher")
            query: Search query (for logging purposes)
            
        Returns:
            True when search can proceed
        """
        with self._coordination_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_search_time
            
            logger.info(f"SEARCH_COORDINATOR: {source} requesting search permission")
            logger.info(f"SEARCH_COORDINATOR: Query preview: '{query[:30]}...'")
            logger.info(f"SEARCH_COORDINATOR: Time since last search: {time_since_last:.1f}s")
            
            if time_since_last < self._min_delay:
                wait_time = self._min_delay - time_since_last
                logger.info(f"SEARCH_COORDINATOR: {source} must wait {wait_time:.1f}s (min delay: {self._min_delay}s)")
                
                # Sleep while holding the lock to ensure no other search can proceed
                time.sleep(wait_time)
                
                logger.info(f"SEARCH_COORDINATOR: {source} completed wait, proceeding with search")
            else:
                logger.info(f"SEARCH_COORDINATOR: {source} can proceed immediately")
            
            # Update timing and counter
            self._last_search_time = time.time()
            self._search_count += 1
            
            logger.info(
                f"SEARCH_COORDINATOR: {source} authorized for search #{self._search_count}"
            )
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        with self._coordination_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_search_time
            
            return {
                "total_searches_coordinated": self._search_count,
                "last_search_time": self._last_search_time,
                "time_since_last_search": time_since_last,
                "min_delay_seconds": self._min_delay,
                "next_search_allowed_in": max(0, self._min_delay - time_since_last)
            }
    
    def reset(self):
        """Reset coordinator state (for testing)"""
        with self._coordination_lock:
            self._last_search_time = 0
            self._search_count = 0
            logger.info("SearchCoordinator state reset")


# Global coordinator instance
search_coordinator = SearchCoordinator()