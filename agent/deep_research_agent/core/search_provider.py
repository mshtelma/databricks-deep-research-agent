"""
Unified search provider interface with fallback support and rate limiting.

This module provides a clean abstraction for different search providers (Brave, Tavily, etc.)
with automatic fallback, unified rate limiting, and consistent result formatting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncContextManager, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import asyncio
import time
import logging
from enum import Enum

from deep_research_agent.core.error_handler import retry, ErrorSeverity, RetryPolicy
from deep_research_agent.core.event_emitter import get_event_emitter
from deep_research_agent.core.types import IntermediateEventType


logger = logging.getLogger(__name__)


class SearchProviderType(Enum):
    """Enumeration of supported search provider types."""
    BRAVE = "brave"
    TAVILY = "tavily" 
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Standardized search result format across all providers."""
    
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clean search result data."""
        self.title = str(self.title).strip()
        self.url = str(self.url).strip()
        self.content = str(self.content).strip()
        self.score = max(0.0, min(1.0, float(self.score)))  # Normalize to 0-1
        
        # Ensure metadata is not None
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create SearchResult from dictionary."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            published_date=data.get("published_date"),
            source=data.get("source", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    
    query: str
    provider: str
    execution_time: float
    result_count: int
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    rate_limited: bool = False


class SearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    All search providers must implement this interface to ensure
    consistent behavior and error handling across the system.
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        provider_type: SearchProviderType,
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.provider_type = provider_type
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.name = provider_type.value
        
        # Provider-specific metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "average_response_time": 0.0
        }
    
    @abstractmethod
    async def search_async(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform async search operation.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    def search(self, query: str, max_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Synchronous wrapper for async search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        # Emit search start event
        correlation_id = kwargs.get('correlation_id')
        stage_id = kwargs.get('stage_id')
        
        event_emitter = get_event_emitter()
        event_emitter.emit_tool_call_start(
            tool_name=f"{self.name}_search",
            parameters={"query": query, "max_results": max_results},
            correlation_id=correlation_id,
            stage_id=stage_id
        )
        
        start_time = time.time()
        try:
            results = asyncio.run(self.search_async(query, max_results, **kwargs))
            
            # Emit search completion event
            execution_time = time.time() - start_time
            event_emitter.emit_tool_call_complete(
                tool_name=f"{self.name}_search",
                success=True,
                result_summary=f"Found {len(results)} results in {execution_time:.2f}s",
                correlation_id=correlation_id,
                stage_id=stage_id,
                execution_time=execution_time,
                results_count=len(results)
            )
            
            # Emit citations for top results
            for i, result in enumerate(results[:3]):  # Top 3 results as citations
                event_emitter.emit_citation_added(
                    title=result.title,
                    url=result.url,
                    snippet=result.content[:200] if result.content else None,
                    correlation_id=correlation_id,
                    stage_id=stage_id
                )
            
            return results
            
        except Exception as e:
            # Emit search error event
            execution_time = time.time() - start_time
            event_emitter.emit_tool_call_error(
                tool_name=f"{self.name}_search",
                error_message=str(e),
                correlation_id=correlation_id,
                stage_id=stage_id,
                is_sanitized=True
            )
            logger.error(f"Search failed for provider {self.name}: {e}")
            return []
    
    @abstractmethod
    def _normalize_result(self, result: Dict[str, Any]) -> SearchResult:
        """
        Normalize provider-specific result to standard SearchResult format.
        
        Args:
            result: Raw result from provider API
            
        Returns:
            Normalized SearchResult object
        """
        pass
    
    def normalize_results(self, raw_results: List[Dict]) -> List[SearchResult]:
        """
        Normalize list of provider-specific results.
        
        Args:
            raw_results: List of raw results from provider
            
        Returns:
            List of normalized SearchResult objects
        """
        normalized = []
        for raw_result in raw_results:
            try:
                normalized_result = self._normalize_result(raw_result)
                if normalized_result:
                    normalized.append(normalized_result)
            except Exception as e:
                logger.warning(f"Failed to normalize result: {e}", extra={"result": raw_result})
        
        return normalized
    
    def update_metrics(self, success: bool, response_time: float, rate_limited: bool = False):
        """Update provider metrics."""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        if rate_limited:
            self.metrics["rate_limited_requests"] += 1
        
        # Update rolling average
        current_avg = self.metrics["average_response_time"]
        total = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get provider health status."""
        total = self.metrics["total_requests"]
        if total == 0:
            success_rate = 1.0
        else:
            success_rate = self.metrics["successful_requests"] / total
        
        return {
            "provider": self.name,
            "success_rate": success_rate,
            "total_requests": total,
            "average_response_time": self.metrics["average_response_time"],
            "rate_limited_percentage": (
                self.metrics["rate_limited_requests"] / max(1, total) * 100
            ),
            "healthy": success_rate > 0.8 and self.metrics["average_response_time"] < 10.0
        }


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    max_concurrent: int = 3
    requests_per_second: float = 1.0  # Simple per-second rate limiting
    cooldown_seconds: float = 1.0    # Minimum delay between requests
    adaptive: bool = True  # Adapt based on provider responses


class RateLimiter:
    """
    Unified rate limiter for search operations with adaptive behavior.
    
    This handles rate limiting across different providers while maintaining
    the sequential batching required for API compliance.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.last_request_times: Dict[str, float] = {}
        self.request_counts: Dict[str, List[float]] = {}
        self.cooldown_until: Dict[str, float] = {}
        
    @asynccontextmanager
    async def acquire(self, provider_name: str) -> AsyncContextManager[None]:
        """
        Acquire rate limit slot for provider.
        
        Args:
            provider_name: Name of the search provider
            
        Yields:
            None (context manager)
        """
        # Initialize provider-specific structures
        if provider_name not in self.semaphores:
            self.semaphores[provider_name] = asyncio.Semaphore(self.config.max_concurrent)
            self.request_counts[provider_name] = []
        
        # Check cooldown period
        if provider_name in self.cooldown_until:
            cooldown_remaining = self.cooldown_until[provider_name] - time.time()
            if cooldown_remaining > 0:
                logger.info(f"Rate limiter cooldown for {provider_name}: {cooldown_remaining:.1f}s")
                await asyncio.sleep(cooldown_remaining)
                del self.cooldown_until[provider_name]
        
        # Check rate limits
        await self._enforce_rate_limit(provider_name)
        
        # Acquire semaphore for concurrent requests
        async with self.semaphores[provider_name]:
            try:
                yield
            finally:
                self.last_request_times[provider_name] = time.time()
    
    async def _enforce_rate_limit(self, provider_name: str):
        """Enforce rate limiting for a provider."""
        now = time.time()
        
        # Simple per-second rate limiting with cooldown only
        
        # Check minimum interval between requests
        if provider_name in self.last_request_times:
            elapsed = now - self.last_request_times[provider_name]
            if elapsed < self.config.cooldown_seconds:
                wait_time = self.config.cooldown_seconds - elapsed
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_counts[provider_name].append(now)
    
    def trigger_cooldown(self, provider_name: str, duration: float = None):
        """Trigger cooldown period for a provider (e.g., after rate limit hit)."""
        duration = duration or self.config.cooldown_seconds * 2
        self.cooldown_until[provider_name] = time.time() + duration
        logger.warning(f"Triggered cooldown for {provider_name}: {duration}s")


class UnifiedSearchManager:
    """
    Manages multiple search providers with fallback and load balancing.
    
    This class provides a unified interface for search operations across
    different providers with automatic fallback and rate limiting.
    """
    
    def __init__(
        self, 
        providers: Dict[str, SearchProvider],
        rate_limiter: Optional[RateLimiter] = None
    ):
        self.providers = providers
        self.rate_limiter = rate_limiter or RateLimiter(RateLimitConfig())
        self.primary_provider: Optional[str] = None
        self.fallback_order: List[str] = list(providers.keys())
        
        # Manager-level metrics
        self.global_metrics = {
            "total_searches": 0,
            "successful_searches": 0,
            "fallback_used": 0,
            "provider_failures": {}
        }
    
    def set_primary_provider(self, provider_name: str):
        """Set primary search provider with fallback order."""
        if provider_name in self.providers:
            self.primary_provider = provider_name
            # Move primary to front of fallback order
            self.fallback_order = [provider_name] + [
                p for p in self.fallback_order if p != provider_name
            ]
    
    @retry("search_with_fallback", "search")
    async def search_with_fallback(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search with automatic fallback to secondary providers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            **kwargs: Provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        self.global_metrics["total_searches"] += 1
        
        last_error = None
        providers_tried = []
        
        for provider_name in self.fallback_order:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            providers_tried.append(provider_name)
            
            try:
                # Apply rate limiting
                async with self.rate_limiter.acquire(provider_name):
                    start_time = time.time()
                    results = await provider.search_async(query, max_results, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update provider metrics
                    provider.update_metrics(
                        success=len(results) > 0,
                        response_time=execution_time
                    )
                    
                    if results:
                        logger.info(
                            f"Search successful with {provider_name}",
                            extra={
                                "query": query,
                                "results_count": len(results),
                                "execution_time": execution_time,
                                "fallback_used": providers_tried[0] != provider_name
                            }
                        )
                        
                        self.global_metrics["successful_searches"] += 1
                        if providers_tried[0] != provider_name:
                            self.global_metrics["fallback_used"] += 1
                        
                        return results
            
            except Exception as e:
                execution_time = time.time() - start_time if 'start_time' in locals() else 0
                last_error = e
                
                # Update provider metrics and global failure tracking
                provider.update_metrics(
                    success=False,
                    response_time=execution_time,
                    rate_limited="rate limit" in str(e).lower()
                )
                
                if provider_name not in self.global_metrics["provider_failures"]:
                    self.global_metrics["provider_failures"][provider_name] = 0
                self.global_metrics["provider_failures"][provider_name] += 1
                
                # Trigger cooldown on rate limiting
                if "rate limit" in str(e).lower():
                    self.rate_limiter.trigger_cooldown(provider_name)
                
                logger.warning(
                    f"Search failed with {provider_name}: {e}",
                    extra={
                        "query": query,
                        "provider": provider_name,
                        "execution_time": execution_time
                    }
                )
        
        # All providers failed
        error_msg = f"All search providers failed. Tried: {providers_tried}. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def search(self, query: str, max_results: int = 5, **kwargs) -> List[SearchResult]:
        """Synchronous wrapper for search with fallback."""
        try:
            return asyncio.run(self.search_with_fallback(query, max_results, **kwargs))
        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            return []
    
    def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        health_status = {
            "global_metrics": self.global_metrics,
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            health_status["providers"][name] = provider.get_health_status()
        
        return health_status
    
    def get_best_provider(self) -> Optional[str]:
        """Get the currently best-performing provider."""
        best_provider = None
        best_score = 0.0
        
        for name, provider in self.providers.items():
            health = provider.get_health_status()
            if health["healthy"]:
                # Score based on success rate and response time
                score = health["success_rate"] * (1.0 / max(0.1, health["average_response_time"]))
                if score > best_score:
                    best_score = score
                    best_provider = name
        
        return best_provider
    
    def reorder_providers_by_performance(self):
        """Reorder fallback providers based on performance."""
        provider_scores = []
        
        for name in self.fallback_order:
            if name in self.providers:
                health = self.providers[name].get_health_status()
                score = health["success_rate"] * (1.0 / max(0.1, health["average_response_time"]))
                provider_scores.append((name, score))
        
        # Sort by score descending
        provider_scores.sort(key=lambda x: x[1], reverse=True)
        self.fallback_order = [name for name, _ in provider_scores]
        
        logger.info(f"Reordered providers by performance: {self.fallback_order}")


# Factory functions for easy provider creation
def create_search_manager(
    provider_configs: Dict[str, Dict[str, Any]],
    rate_limit_config: Optional[RateLimitConfig] = None
) -> UnifiedSearchManager:
    """
    Create unified search manager from configuration.
    
    Args:
        provider_configs: Dictionary mapping provider names to their configs
        rate_limit_config: Optional rate limiting configuration
        
    Returns:
        Configured UnifiedSearchManager instance
    """
    providers = {}
    rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
    
    # This would be extended to create actual provider instances
    # For now, it's a placeholder for the factory pattern
    
    return UnifiedSearchManager(providers, rate_limiter)