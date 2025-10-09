"""
Embedding Manager for centralized embedding computation and caching.

This module provides a unified interface for computing and caching embeddings
across the entire research workflow, preventing redundant computations and
enabling semantic similarity operations for grounding verification.
"""

import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging

try:
    from databricks_langchain import DatabricksEmbeddings
    DATABRICKS_EMBEDDINGS_AVAILABLE = True
except ImportError:
    DATABRICKS_EMBEDDINGS_AVAILABLE = False
    DatabricksEmbeddings = None

from .cache_manager import IntelligentCache
from .types import SearchResult
from .exceptions import ResearchAgentError

logger = logging.getLogger(__name__)


class EmbeddingError(ResearchAgentError):
    """Exception for embedding-related errors."""
    pass


class EmbeddingManager:
    """
    Manages embedding computation and caching for the entire workflow.
    
    This class provides a unified interface for:
    - Computing embeddings with automatic caching
    - Enriching SearchResults with embeddings in metadata
    - Reusing embeddings across deduplication, re-ranking, and grounding
    """
    
    def __init__(self, endpoint_name: str = "databricks-gte-large-en", 
                 cache_manager: Optional[IntelligentCache] = None):
        """
        Initialize the embedding manager.
        
        Args:
            endpoint_name: Name of the Databricks embedding endpoint
            cache_manager: Optional cache manager for storing embeddings
        """
        if not DATABRICKS_EMBEDDINGS_AVAILABLE:
            raise EmbeddingError("databricks-langchain not available")
        
        # Initialize DatabricksEmbeddings
        self.embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
        self.endpoint_name = endpoint_name
        
        # Initialize or create cache
        self.cache = cache_manager or IntelligentCache(strategy="ADAPTIVE")
        
        # Stats tracking
        self.stats = {
            "embeddings_computed": 0,
            "embeddings_cached": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(
            f"Initialized EmbeddingManager with DatabricksEmbeddings | "
            f"endpoint={endpoint_name} | cache_enabled=True"
        )
    
    def get_or_compute_embedding(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Get embedding from cache or compute if not found.
        
        Args:
            text: Text to embed
            cache_key: Optional custom cache key, will generate if not provided
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding computation fails
        """
        # Generate cache key if not provided
        if not cache_key:
            cache_key = self._generate_cache_key(text)
        
        # Check cache first
        cached_embedding = self.cache.get(cache_key, category="embeddings")
        if cached_embedding is not None:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for embedding key: {cache_key[:12]}...")
            return np.array(cached_embedding)  # Convert from list back to numpy
        
        # Cache miss - compute embedding
        self.stats["cache_misses"] += 1
        logger.debug(f"Cache miss for embedding key: {cache_key[:12]}... - computing")
        
        try:
            # Use DatabricksEmbeddings.embed_query for single text
            embedding = self.embeddings.embed_query(text)
            embedding_array = np.array(embedding)
            self.stats["embeddings_computed"] += 1
            
            # Cache as list (JSON serializable)
            self.cache.set(
                cache_key, 
                embedding,  # embedding is already a list from DatabricksEmbeddings
                category="embeddings",
                ttl=7200  # 2 hours - embeddings are stable
            )
            self.stats["embeddings_cached"] += 1
            
            logger.debug(f"Computed and cached embedding for key: {cache_key[:12]}...")
            return embedding_array
            
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            raise EmbeddingError(f"Embedding computation failed: {e}")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is between 0 and 1 (cosine similarity can be -1 to 1)
        # For text embeddings, we typically want 0 to 1 range
        return float(max(0.0, similarity))
    
    def enrich_search_results_with_embeddings(self, results: List[Union[SearchResult, Dict]]) -> List[Union[SearchResult, Dict]]:
        """
        Add embeddings to SearchResult metadata for later reuse.
        
        Args:
            results: List of search results to enrich (can be SearchResult objects or dicts)
            
        Returns:
            Same list with embeddings added to metadata
        """
        if not results:
            return results
        
        enriched_count = 0
        
        for result in results:
            # Handle both SearchResult objects and dict representations
            if hasattr(result, 'metadata'):
                # SearchResult object
                metadata = result.metadata
                has_embedding = 'embedding' in metadata
            elif isinstance(result, dict):
                # Dict representation - metadata might be nested or at root level
                metadata = result.get('metadata', {})
                has_embedding = 'embedding' in metadata or 'embedding' in result
            else:
                # Skip unknown types
                continue
                
            if not has_embedding:
                # Extract title and content based on result type
                if hasattr(result, 'title'):
                    # SearchResult object
                    title = result.title
                    content = result.content
                    url = result.url
                elif isinstance(result, dict):
                    # Dict representation
                    title = result.get('title', '')
                    content = result.get('content', '')
                    url = result.get('url', '')
                else:
                    continue
                
                # Create combined text for embedding (title + truncated content)
                text_parts = []
                if title:
                    text_parts.append(title.strip())
                if content:
                    # Truncate content to reasonable size for embedding
                    content_str = content.strip()[:1000]
                    text_parts.append(content_str)
                
                combined_text = ' '.join(text_parts)
                
                if combined_text:
                    # Generate cache key that's specific to this search result
                    cache_key = self._generate_search_result_cache_key(result, combined_text)
                    
                    try:
                        embedding = self.get_or_compute_embedding(combined_text, cache_key)
                        
                        # Store embedding based on result type
                        if hasattr(result, 'metadata'):
                            # SearchResult object
                            result.metadata['embedding'] = embedding.tolist()
                            result.metadata['embedding_cached'] = True
                            result.metadata['embedding_text_length'] = len(combined_text)
                        elif isinstance(result, dict):
                            # Dict representation
                            if 'metadata' not in result:
                                result['metadata'] = {}
                            result['metadata']['embedding'] = embedding.tolist()
                            result['metadata']['embedding_cached'] = True
                            result['metadata']['embedding_text_length'] = len(combined_text)
                        
                        enriched_count += 1
                        
                    except EmbeddingError as e:
                        logger.warning(
                            f"Failed to compute embedding for search result: {e}",
                            url=url,
                            title=title[:50] if title else None
                        )
                        # Continue with other results
                        continue
        
        logger.info(
            f"Enriched {enriched_count}/{len(results)} search results with embeddings"
        )
        
        return results
    
    def get_embedding_from_search_result(self, result: Union[SearchResult, Dict]) -> Optional[np.ndarray]:
        """
        Extract embedding from SearchResult metadata.
        
        Args:
            result: SearchResult or dict with potential embedding in metadata
            
        Returns:
            Embedding array if available, None otherwise
        """
        if hasattr(result, 'metadata'):
            # SearchResult object
            if 'embedding' in result.metadata:
                return np.array(result.metadata['embedding'])
        elif isinstance(result, dict):
            # Dict representation
            metadata = result.get('metadata', {})
            if 'embedding' in metadata:
                return np.array(metadata['embedding'])
            # Also check if embedding is stored at root level
            elif 'embedding' in result:
                return np.array(result['embedding'])
        return None
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for text embedding."""
        # Use SHA256 for consistent, collision-resistant keys
        return f"emb_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    
    def _generate_search_result_cache_key(self, result: Union[SearchResult, Dict], text: str) -> str:
        """Generate a cache key specific to a search result."""
        # Include URL/source and content hash for uniqueness
        if hasattr(result, 'url'):
            # SearchResult object
            source_part = result.url or result.source or "unknown"
        elif isinstance(result, dict):
            # Dict representation
            source_part = result.get('url', '') or result.get('source', '') or "unknown"
        else:
            source_part = "unknown"
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        source_hash = hashlib.md5(source_part.encode('utf-8')).hexdigest()[:8]
        
        return f"sr_emb_{source_hash}_{text_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        cache_stats = self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}
        return {
            **self.stats,
            "endpoint_name": self.endpoint_name,
            "cache_stats": cache_stats
        }
    
    def clear_cache(self, category: str = "embeddings"):
        """Clear embedding cache for a specific category."""
        if hasattr(self.cache, 'clear_category'):
            self.cache.clear_category(category)
            logger.info(f"Cleared embedding cache for category: {category}")


class ObservationEmbeddingIndex:
    """
    Efficient index for observation embeddings to enable semantic deduplication.

    This class maintains a vectorized index of all observation embeddings,
    allowing fast similarity searches across hundreds of observations.
    """

    def __init__(self, embedding_manager: EmbeddingManager, max_observations: int = 400):
        """
        Initialize the observation embedding index.

        Args:
            embedding_manager: The EmbeddingManager instance for computing embeddings
            max_observations: Maximum number of observations to index
        """
        self.embedding_manager = embedding_manager
        self.max_observations = max_observations

        # Storage for embeddings and metadata
        self.embeddings: List[np.ndarray] = []  # Embedding vectors
        self.observation_hashes: List[str] = []  # Hash of observation content
        self.step_ids: List[Optional[str]] = []  # Step IDs for partitioning
        self.observation_texts: List[str] = []  # Original texts (for debugging)

        # Stats
        self.stats = {
            "total_indexed": 0,
            "duplicates_detected": 0,
            "similarity_checks": 0
        }

        logger.info(f"Initialized ObservationEmbeddingIndex with max_observations={max_observations}")

    def add_observation(self, text: str, step_id: Optional[str] = None) -> bool:
        """
        Add an observation to the index.

        Args:
            text: The observation text
            step_id: Optional step ID for partitioning

        Returns:
            True if added (not duplicate), False if duplicate detected
        """
        if not text or not text.strip():
            return False

        # Generate hash for exact duplicate check
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check for exact duplicate by hash
        if text_hash in self.observation_hashes:
            self.stats["duplicates_detected"] += 1
            logger.debug(f"Exact duplicate detected by hash: {text[:50]}...")
            return False

        # Compute embedding for this observation
        try:
            embedding = self.embedding_manager.get_or_compute_embedding(text)
        except Exception as e:
            logger.warning(f"Failed to compute embedding for observation: {e}")
            # Fall back to adding without embedding check
            self._add_to_index(text, text_hash, None, step_id)
            return True

        # Check for semantic duplicates
        is_duplicate = self._check_semantic_duplicate(embedding, text, step_id)

        if is_duplicate:
            self.stats["duplicates_detected"] += 1
            logger.debug(f"Semantic duplicate detected: {text[:50]}...")
            return False

        # Add to index
        self._add_to_index(text, text_hash, embedding, step_id)
        return True

    def _check_semantic_duplicate(self, embedding: np.ndarray, text: str, step_id: Optional[str] = None) -> bool:
        """
        Check if an observation is a semantic duplicate of existing ones.

        Uses different thresholds for within-step vs cross-step comparisons.

        Args:
            embedding: Embedding vector of the new observation
            text: The observation text
            step_id: Optional step ID for partitioning

        Returns:
            True if semantic duplicate found, False otherwise
        """
        if not self.embeddings:
            return False

        self.stats["similarity_checks"] += 1

        # Convert embeddings list to numpy array for vectorized operations
        embeddings_array = np.array(self.embeddings)

        # Compute similarities with all existing embeddings (vectorized)
        similarities = self._compute_batch_similarities(embedding, embeddings_array)

        # Apply different thresholds based on step_id
        for i, similarity in enumerate(similarities):
            existing_step_id = self.step_ids[i] if i < len(self.step_ids) else None

            # Determine threshold based on step context
            if step_id and existing_step_id == step_id:
                # Same step - use lower threshold (more aggressive dedup)
                threshold = 0.85
            else:
                # Different steps - use higher threshold (only exact semantic duplicates)
                threshold = 0.95

            if similarity >= threshold:
                logger.debug(
                    f"Semantic duplicate found (similarity={similarity:.3f}, threshold={threshold}): "
                    f"'{text[:30]}...' similar to '{self.observation_texts[i][:30]}...'"
                )
                return True

        return False

    def _compute_batch_similarities(self, query_embedding: np.ndarray, embeddings_array: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and all embeddings (vectorized).

        Args:
            query_embedding: The query embedding vector
            embeddings_array: Array of existing embeddings

        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings_array))

        query_normalized = query_embedding / query_norm

        # Compute dot products with all embeddings
        dot_products = embeddings_array @ query_normalized

        # Compute norms of all embeddings
        norms = np.linalg.norm(embeddings_array, axis=1)

        # Avoid division by zero
        norms[norms == 0] = 1.0

        # Compute cosine similarities
        similarities = dot_products / norms

        # Ensure positive similarities (text embeddings should be non-negative)
        similarities = np.maximum(similarities, 0.0)

        return similarities

    def _add_to_index(self, text: str, text_hash: str, embedding: Optional[np.ndarray], step_id: Optional[str] = None):
        """
        Add observation to the index.

        Args:
            text: The observation text
            text_hash: Hash of the text
            embedding: Embedding vector (can be None if computation failed)
            step_id: Optional step ID
        """
        # If we're at max capacity, remove oldest observations
        if len(self.observation_hashes) >= self.max_observations:
            # Remove oldest 10% to make room
            remove_count = max(1, self.max_observations // 10)
            self.embeddings = self.embeddings[remove_count:]
            self.observation_hashes = self.observation_hashes[remove_count:]
            self.step_ids = self.step_ids[remove_count:]
            self.observation_texts = self.observation_texts[remove_count:]
            logger.debug(f"Pruned {remove_count} oldest observations from index")

        # Add new observation
        if embedding is not None:
            self.embeddings.append(embedding)
        self.observation_hashes.append(text_hash)
        self.step_ids.append(step_id)
        self.observation_texts.append(text[:100])  # Store truncated for debugging

        self.stats["total_indexed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.stats,
            "current_size": len(self.observation_hashes),
            "has_embeddings": len(self.embeddings),
            "unique_steps": len(set(s for s in self.step_ids if s))
        }

    def clear(self):
        """Clear the entire index."""
        self.embeddings.clear()
        self.observation_hashes.clear()
        self.step_ids.clear()
        self.observation_texts.clear()
        logger.info("Cleared observation embedding index")