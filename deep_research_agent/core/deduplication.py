"""
Semantic deduplication engine for search results.

This module implements intelligent deduplication of search results using
semantic similarity analysis to identify and merge duplicate or highly
similar content while preserving important metadata.
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field

from .types import SearchResult, SearchResultType, DeduplicationMetrics
from .databricks_embeddings import DatabricksEmbeddingClient, DatabricksEmbeddingError
from .logging import get_logger
from .exceptions import ResearchAgentError
from .utils import format_duration

logger = get_logger(__name__)


class DeduplicationError(ResearchAgentError):
    """Exception for deduplication errors."""
    pass


@dataclass
class SimilarityCluster:
    """Represents a cluster of similar search results."""
    representative: SearchResult  # The best result from this cluster
    members: List[SearchResult] = field(default_factory=list)  # All results in cluster
    similarity_scores: List[float] = field(default_factory=list)  # Similarity to representative
    cluster_id: str = ""
    average_similarity: float = 0.0


class SemanticDeduplicator:
    """
    Semantic deduplication engine using embeddings and clustering.
    
    Uses Databricks embedding models to identify semantically similar
    content and merge duplicate results while preserving metadata.
    """
    
    def __init__(
        self,
        embedding_client: DatabricksEmbeddingClient,
        similarity_threshold: float = 0.85,
        min_content_length: int = 50,
        preserve_different_sources: bool = True
    ):
        """
        Initialize the deduplication engine.
        
        Args:
            embedding_client: Client for generating embeddings
            similarity_threshold: Minimum similarity to consider as duplicate (0.0-1.0)
            min_content_length: Minimum content length to consider for deduplication
            preserve_different_sources: Keep results from different sources even if similar
        """
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.min_content_length = min_content_length
        self.preserve_different_sources = preserve_different_sources
        
        logger.info(
            "Initialized semantic deduplicator",
            similarity_threshold=similarity_threshold,
            min_content_length=min_content_length,
            preserve_different_sources=preserve_different_sources
        )
    
    def deduplicate_results(
        self, 
        results: List[SearchResult]
    ) -> Tuple[List[SearchResult], DeduplicationMetrics]:
        """
        Deduplicate a list of search results using semantic similarity.
        
        Args:
            results: List of search results to deduplicate
            
        Returns:
            Tuple of (deduplicated_results, metrics)
            
        Raises:
            DeduplicationError: If deduplication fails
        """
        start_time = time.time()
        
        if not results:
            return [], DeduplicationMetrics()
        
        if len(results) == 1:
            return results, DeduplicationMetrics(
                original_count=1,
                deduplicated_count=1,
                processing_time_seconds=time.time() - start_time
            )
        
        try:
            logger.info(f"Starting deduplication of {len(results)} results")
            
            # Filter results that are too short
            filterable_results = [
                r for r in results 
                if len(r.content.strip()) >= self.min_content_length
            ]
            
            # Keep short results as-is (they'll be added back)
            short_results = [
                r for r in results 
                if len(r.content.strip()) < self.min_content_length
            ]
            
            if not filterable_results:
                logger.info("No results meet minimum content length for deduplication")
                return results, DeduplicationMetrics(
                    original_count=len(results),
                    deduplicated_count=len(results),
                    processing_time_seconds=time.time() - start_time
                )
            
            # Generate embeddings
            embeddings = self._get_embeddings_for_results(filterable_results)
            
            # Find similarity clusters
            clusters = self._cluster_similar_results(filterable_results, embeddings)
            
            # Select representatives from each cluster
            deduplicated = self._select_cluster_representatives(clusters)
            
            # Add back short results
            deduplicated.extend(short_results)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            reduction_percentage = ((len(results) - len(deduplicated)) / len(results)) * 100
            
            metrics = DeduplicationMetrics(
                original_count=len(results),
                deduplicated_count=len(deduplicated),
                reduction_percentage=reduction_percentage,
                clusters_found=len(clusters),
                processing_time_seconds=processing_time
            )
            
            logger.info(
                "Deduplication completed successfully",
                original_count=len(results),
                deduplicated_count=len(deduplicated),
                reduction_percentage=f"{reduction_percentage:.1f}%",
                clusters_found=len(clusters),
                processing_time=format_duration(processing_time)
            )
            
            return deduplicated, metrics
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            # Return original results on failure
            return results, DeduplicationMetrics(
                original_count=len(results),
                deduplicated_count=len(results),
                processing_time_seconds=time.time() - start_time
            )
    
    def _get_embeddings_for_results(self, results: List[SearchResult]) -> np.ndarray:
        """Generate embeddings for search results."""
        try:
            # Prepare content for embedding
            contents = []
            for result in results:
                # Use title + content for better representation
                text = ""
                if result.title:
                    text += f"{result.title}. "
                text += result.content.strip()
                contents.append(text)
            
            # Generate embeddings
            embeddings = self.embedding_client.get_embeddings(contents)
            
            return embeddings
            
        except DatabricksEmbeddingError as e:
            logger.error(f"Failed to generate embeddings for deduplication: {e}")
            raise DeduplicationError(f"Embedding generation failed: {e}")
    
    def _cluster_similar_results(
        self, 
        results: List[SearchResult], 
        embeddings: np.ndarray
    ) -> List[SimilarityCluster]:
        """
        Cluster results based on semantic similarity.
        
        Uses a greedy clustering approach where we iteratively find
        the most similar unassigned results to existing clusters.
        """
        if len(results) != len(embeddings):
            raise DeduplicationError("Results and embeddings count mismatch")
        
        clusters = []
        assigned = set()  # Indices of assigned results
        
        # Compute similarity matrix
        similarity_matrix = self.embedding_client.compute_similarity(embeddings, embeddings)
        
        for i, result in enumerate(results):
            if i in assigned:
                continue
            
            # Start a new cluster with this result as representative
            cluster = SimilarityCluster(
                representative=result,
                cluster_id=f"cluster_{len(clusters)}",
                members=[result],
                similarity_scores=[1.0]  # Perfect similarity to itself
            )
            
            assigned.add(i)
            
            # Find similar results to add to this cluster
            for j, other_result in enumerate(results):
                if j in assigned or i == j:
                    continue
                
                similarity = similarity_matrix[i, j]
                
                # Check if similar enough to cluster
                if similarity >= self.similarity_threshold:
                    # Additional check for source diversity if enabled
                    if (self.preserve_different_sources and 
                        self._should_preserve_different_source(result, other_result)):
                        continue
                    
                    cluster.members.append(other_result)
                    cluster.similarity_scores.append(float(similarity))
                    assigned.add(j)
            
            # Calculate average similarity for the cluster
            if cluster.similarity_scores:
                cluster.average_similarity = np.mean(cluster.similarity_scores)
            
            clusters.append(cluster)
        
        return clusters
    
    def _should_preserve_different_source(
        self, 
        result1: SearchResult, 
        result2: SearchResult
    ) -> bool:
        """
        Determine if results from different sources should be preserved.
        
        This helps maintain source diversity even for similar content.
        """
        if not self.preserve_different_sources:
            return False
        
        # Compare source domains
        source1 = self._extract_domain(result1.source or result1.url or "")
        source2 = self._extract_domain(result2.source or result2.url or "")
        
        # If sources are different and both are high quality, preserve both
        if source1 != source2 and source1 and source2:
            # Additional quality checks could go here
            return True
        
        return False
    
    def _extract_domain(self, url_or_source: str) -> str:
        """Extract domain from URL or source string."""
        if not url_or_source:
            return ""
        
        # Simple domain extraction
        url = url_or_source.lower().strip()
        
        # Remove protocol
        for protocol in ['https://', 'http://', 'www.']:
            if url.startswith(protocol):
                url = url[len(protocol):]
        
        # Get domain part
        domain = url.split('/')[0].split('?')[0]
        
        return domain
    
    def _select_cluster_representatives(
        self, 
        clusters: List[SimilarityCluster]
    ) -> List[SearchResult]:
        """
        Select the best representative from each cluster.
        
        The representative is chosen based on multiple factors:
        - Source authority/reliability
        - Content completeness
        - Recency
        - URL accessibility
        """
        representatives = []
        
        for cluster in clusters:
            if len(cluster.members) == 1:
                # Single member, use as-is
                representatives.append(cluster.representative)
                continue
            
            # Score all members and select the best
            best_result = self._select_best_result_from_cluster(cluster)
            
            # Merge metadata from all cluster members
            merged_result = self._merge_cluster_metadata(best_result, cluster.members)
            
            representatives.append(merged_result)
        
        return representatives
    
    def _select_best_result_from_cluster(self, cluster: SimilarityCluster) -> SearchResult:
        """
        Select the best result from a cluster based on multiple criteria.
        """
        def score_result(result: SearchResult) -> float:
            score = 0.0
            
            # Content completeness (20%)
            content_score = min(len(result.content) / 1000, 1.0) * 0.2
            score += content_score
            
            # Source authority (30%)
            authority_score = self._get_source_authority_score(result) * 0.3
            score += authority_score
            
            # Has title (10%)
            if result.title and result.title.strip():
                score += 0.1
            
            # Has URL (10%)
            if result.url and result.url.strip():
                score += 0.1
            
            # Result score if available (20%)
            if hasattr(result, 'score') and result.score:
                normalized_score = min(result.score / 100, 1.0) * 0.2
                score += normalized_score
            
            # Recency bonus (10%)
            if result.published_date:
                # Simple recency bonus - could be more sophisticated
                score += 0.1
            
            return score
        
        # Score all members and return the best
        scored_members = [(score_result(member), member) for member in cluster.members]
        scored_members.sort(key=lambda x: x[0], reverse=True)
        
        return scored_members[0][1]
    
    def _get_source_authority_score(self, result: SearchResult) -> float:
        """
        Calculate a simple authority score for a source.
        
        This is a basic implementation that could be enhanced with
        a more sophisticated authority database.
        """
        if not result.source and not result.url:
            return 0.5  # Neutral score
        
        source = (result.source or result.url or "").lower()
        
        # High authority sources
        high_authority = [
            'wikipedia', 'arxiv', 'nature', 'science', 'ieee', 'acm',
            'gov', 'edu', 'reuters', 'bbc', 'cnn', 'nytimes'
        ]
        
        # Medium authority sources
        medium_authority = [
            'medium', 'forbes', 'techcrunch', 'wired', 'ars-technica'
        ]
        
        for authority in high_authority:
            if authority in source:
                return 1.0
        
        for authority in medium_authority:
            if authority in source:
                return 0.7
        
        return 0.5  # Default/unknown authority
    
    def _merge_cluster_metadata(
        self, 
        representative: SearchResult, 
        all_members: List[SearchResult]
    ) -> SearchResult:
        """
        Merge metadata from all cluster members into the representative.
        
        This preserves information about duplicates and alternative sources.
        """
        # Create a copy of the representative
        merged = SearchResult(
            content=representative.content,
            source=representative.source,
            url=representative.url,
            title=representative.title,
            score=representative.score,
            published_date=representative.published_date,
            result_type=representative.result_type,
            metadata=representative.metadata.copy()
        )
        
        # Add cluster information
        if len(all_members) > 1:
            # Collect alternative sources
            alt_sources = []
            alt_urls = []
            
            for member in all_members:
                if member != representative:
                    if member.source and member.source not in alt_sources:
                        alt_sources.append(member.source)
                    if member.url and member.url not in alt_urls:
                        alt_urls.append(member.url)
            
            # Update metadata
            merged.metadata.update({
                "is_clustered": True,
                "cluster_size": len(all_members),
                "alternative_sources": alt_sources,
                "alternative_urls": alt_urls,
                "deduplication_info": {
                    "cluster_id": f"cluster_{id(all_members)}",
                    "duplicate_count": len(all_members) - 1,
                    "merged_at": time.time()
                }
            })
        
        return merged
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication engine statistics."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "min_content_length": self.min_content_length,
            "preserve_different_sources": self.preserve_different_sources,
            "embedding_client_stats": self.embedding_client.get_stats()
        }