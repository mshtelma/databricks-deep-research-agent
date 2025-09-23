"""
Robust search fallback system.

Provides automatic fallback strategies when search tools fail,
ensuring research continues even with provider outages.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum

from deep_research_agent.core import get_logger, SearchResult

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Available search strategies."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    CACHED = "cached"
    SYNTHETIC = "synthetic"
    ENTITY_BASED = "entity_based"


class SearchFallbackManager:
    """Manages search fallback strategies when primary search fails."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fallback manager with configuration."""
        self.config = config or {}
        self.failure_counts = {}  # Track provider failures
        self.circuit_breaker_timeout = timedelta(minutes=5)
        self.max_failures_before_circuit = 3
        
        # Cache for failed queries to avoid retry storms
        self.failed_query_cache = {}
        self.cache_timeout = timedelta(minutes=10)
    
    async def execute_search_with_fallback(
        self,
        query: str,
        search_tools: List[Any],
        max_results: int = 5
    ) -> Tuple[List[SearchResult], SearchStrategy]:
        """
        Execute search with automatic fallback on failure.
        
        Args:
            query: Search query
            search_tools: Available search tools
            max_results: Maximum results to return
            
        Returns:
            Tuple of (search results, strategy used)
        """
        logger.info(f"Executing search with fallback for query: {query}")
        
        # Strategy 1: Try primary search providers
        try:
            results = await self._try_primary_search(query, search_tools, max_results)
            if results:
                logger.info(f"Primary search succeeded with {len(results)} results")
                return results, SearchStrategy.PRIMARY
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            await self._record_search_failure("primary", str(e))
        
        # Strategy 2: Try secondary/backup search providers
        try:
            results = await self._try_secondary_search(query, search_tools, max_results)
            if results:
                logger.info(f"Secondary search succeeded with {len(results)} results")
                return results, SearchStrategy.SECONDARY
        except Exception as e:
            logger.warning(f"Secondary search failed: {e}")
            await self._record_search_failure("secondary", str(e))
        
        # Strategy 3: Check cache for similar queries
        cached_results = await self._try_cached_search(query, max_results)
        if cached_results:
            logger.info(f"Cache search succeeded with {len(cached_results)} results")
            return cached_results, SearchStrategy.CACHED
        
        # Strategy 4: Entity-based fallback search
        try:
            results = await self._try_entity_based_search(query, search_tools, max_results)
            if results:
                logger.info(f"Entity-based search succeeded with {len(results)} results")
                return results, SearchStrategy.ENTITY_BASED
        except Exception as e:
            logger.warning(f"Entity-based search failed: {e}")
        
        # Strategy 5: Generate synthetic placeholder results (last resort)
        logger.error(f"All search strategies failed for query: {query}")
        synthetic_results = await self._generate_synthetic_results(query, max_results)
        return synthetic_results, SearchStrategy.SYNTHETIC
    
    async def _try_primary_search(
        self,
        query: str,
        search_tools: List[Any],
        max_results: int
    ) -> List[SearchResult]:
        """Try primary search tools."""
        if not search_tools:
            raise Exception("No search tools available")
        
        # Use first available tool as primary
        primary_tool = search_tools[0]
        
        # Check circuit breaker
        if self._is_circuit_open("primary"):
            raise Exception("Primary search circuit breaker is open")
        
        try:
            if hasattr(primary_tool, 'search'):
                results = await primary_tool.search(query, limit=max_results)
            elif hasattr(primary_tool, 'run'):
                results = await primary_tool.run(query)
            else:
                raise Exception("Search tool has no search or run method")
            
            # Reset failure count on success
            self.failure_counts["primary"] = 0
            return results if isinstance(results, list) else []
            
        except Exception as e:
            await self._record_search_failure("primary", str(e))
            raise
    
    async def _try_secondary_search(
        self,
        query: str,
        search_tools: List[Any],
        max_results: int
    ) -> List[SearchResult]:
        """Try secondary/backup search tools."""
        if len(search_tools) < 2:
            raise Exception("No secondary search tools available")
        
        # Check circuit breaker
        if self._is_circuit_open("secondary"):
            raise Exception("Secondary search circuit breaker is open")
        
        # Try remaining tools as secondary
        for tool in search_tools[1:]:
            try:
                if hasattr(tool, 'search'):
                    results = await tool.search(query, limit=max_results)
                elif hasattr(tool, 'run'):
                    results = await tool.run(query)
                else:
                    continue
                
                if results:
                    # Reset failure count on success
                    self.failure_counts["secondary"] = 0
                    return results if isinstance(results, list) else []
                    
            except Exception as e:
                logger.debug(f"Secondary tool failed: {e}")
                continue
        
        # All secondary tools failed
        await self._record_search_failure("secondary", "All secondary tools failed")
        raise Exception("All secondary search tools failed")
    
    async def _try_cached_search(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Try to find cached results for similar queries."""
        # This is a placeholder - in a real implementation, you'd:
        # 1. Check a cache/database for similar queries
        # 2. Use semantic similarity to find related results
        # 3. Return cached results if available
        
        logger.debug("Cache search not implemented - returning empty")
        return []
    
    async def _try_entity_based_search(
        self,
        query: str,
        search_tools: List[Any],
        max_results: int
    ) -> List[SearchResult]:
        """Try searching for individual entities mentioned in the query."""
        if not search_tools:
            raise Exception("No search tools available for entity search")
        
        # Extract entities from query (simplified approach)
        entities = self._extract_entities_from_query(query)
        if not entities:
            raise Exception("No entities found in query")
        
        results = []
        tool = search_tools[0]  # Use first available tool
        
        for entity in entities[:3]:  # Limit to 3 entities to avoid spam
            try:
                entity_query = f"{entity} information"
                
                if hasattr(tool, 'search'):
                    entity_results = await tool.search(entity_query, limit=2)
                elif hasattr(tool, 'run'):
                    entity_results = await tool.run(entity_query)
                else:
                    continue
                
                if entity_results:
                    results.extend(entity_results[:2])  # Max 2 results per entity
                
            except Exception as e:
                logger.debug(f"Entity search failed for {entity}: {e}")
                continue
        
        if not results:
            raise Exception("Entity-based search found no results")
        
        return results[:max_results]
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query text (simplified implementation)."""
        # This is a very basic implementation
        # In production, you'd use NER models or entity extraction libraries
        
        common_entities = []
        words = query.lower().split()
        
        # Look for country names, companies, technologies, etc.
        country_indicators = ['spain', 'france', 'germany', 'uk', 'united kingdom', 
                             'switzerland', 'poland', 'bulgaria', 'usa', 'canada']
        
        for word in words:
            if word in country_indicators:
                common_entities.append(word.title())
        
        # Look for capitalized words (potential proper nouns)
        original_words = query.split()
        for word in original_words:
            if len(word) > 2 and word[0].isupper() and word.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at']:
                common_entities.append(word)
        
        return list(set(common_entities))  # Remove duplicates
    
    async def _generate_synthetic_results(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Generate synthetic placeholder results as absolute last resort."""
        logger.warning(f"Generating synthetic results for failed query: {query}")
        
        # Create placeholder results that acknowledge the search failure
        synthetic_results = []
        
        for i in range(min(max_results, 3)):  # Limit synthetic results
            result = SearchResult(
                title=f"Search Unavailable - Information Needed #{i+1}",
                url=f"https://search-unavailable.placeholder/{i+1}",
                content=f"Search services are currently unavailable. "
                       f"Manual research recommended for: {query}. "
                       f"Please verify this information from authoritative sources.",
                snippet=f"Search tools failed - manual verification needed for {query}",
                source="synthetic_fallback",
                timestamp=datetime.now().isoformat(),
                relevance_score=0.1,  # Very low relevance to indicate synthetic nature
                metadata={
                    "synthetic": True,
                    "reason": "all_search_strategies_failed",
                    "original_query": query
                }
            )
            synthetic_results.append(result)
        
        return synthetic_results
    
    async def _record_search_failure(self, strategy: str, error: str):
        """Record search failure for circuit breaker logic."""
        self.failure_counts[strategy] = self.failure_counts.get(strategy, 0) + 1
        
        logger.warning(f"Search failure recorded for {strategy}: {error} "
                      f"(count: {self.failure_counts[strategy]})")
        
        # Record in cache to avoid immediate retry
        cache_key = f"{strategy}_last_failure"
        self.failed_query_cache[cache_key] = {
            "timestamp": datetime.now(),
            "error": error,
            "count": self.failure_counts[strategy]
        }
    
    def _is_circuit_open(self, strategy: str) -> bool:
        """Check if circuit breaker is open for a strategy."""
        failure_count = self.failure_counts.get(strategy, 0)
        
        if failure_count < self.max_failures_before_circuit:
            return False
        
        # Check if timeout has expired
        cache_key = f"{strategy}_last_failure"
        last_failure = self.failed_query_cache.get(cache_key)
        
        if not last_failure:
            return False
        
        time_since_failure = datetime.now() - last_failure["timestamp"]
        if time_since_failure > self.circuit_breaker_timeout:
            # Reset the circuit breaker
            self.failure_counts[strategy] = 0
            return False
        
        return True
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics about fallback usage."""
        return {
            "failure_counts": self.failure_counts.copy(),
            "circuit_breakers": {
                strategy: self._is_circuit_open(strategy)
                for strategy in ["primary", "secondary"]
            },
            "cached_failures": len(self.failed_query_cache)
        }