"""
Result evaluation and coverage analysis for adaptive search.

This module provides intelligent evaluation of search results to determine
coverage completeness, identify gaps, and assess result quality for
adaptive query generation.
"""

import time
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import Counter

from .types import SearchResult, CoverageAnalysis, QueryComplexity
from .databricks_embeddings import DatabricksEmbeddingClient
from .query_intelligence import QueryAnalyzer
from .logging import get_logger
from .utils import parse_search_keywords, truncate_text

logger = get_logger(__name__)


class ResultEvaluator:
    """
    Evaluates search results for coverage and quality.
    
    Uses semantic similarity analysis to determine how well
    search results answer the original query and identify
    gaps that need additional research.
    """
    
    def __init__(
        self,
        embedding_client: DatabricksEmbeddingClient,
        query_analyzer: QueryAnalyzer,
        coverage_threshold: float = 0.7,
        min_results_for_analysis: int = 2
    ):
        """
        Initialize the result evaluator.
        
        Args:
            embedding_client: Client for generating embeddings
            query_analyzer: Query analysis component
            coverage_threshold: Minimum coverage score to consider sufficient
            min_results_for_analysis: Minimum results needed for meaningful analysis
        """
        self.embedding_client = embedding_client
        self.query_analyzer = query_analyzer
        self.coverage_threshold = coverage_threshold
        self.min_results_for_analysis = min_results_for_analysis
        
        logger.info(
            "Initialized result evaluator",
            coverage_threshold=coverage_threshold,
            min_results=min_results_for_analysis
        )
    
    def evaluate_coverage(
        self,
        query: str,
        results: List[SearchResult],
        query_complexity: QueryComplexity = None
    ) -> CoverageAnalysis:
        """
        Evaluate how well results cover the query requirements.
        
        Args:
            query: Original user query
            results: Search results to evaluate
            query_complexity: Pre-analyzed query complexity (optional)
            
        Returns:
            CoverageAnalysis with coverage score and gaps
        """
        start_time = time.time()
        
        if not results:
            return CoverageAnalysis(
                score=0.0,
                missing_aspects=["No results found"],
                needs_refinement=True,
                confidence=1.0
            )
        
        if len(results) < self.min_results_for_analysis:
            return CoverageAnalysis(
                score=0.3,  # Low but not zero for having some results
                missing_aspects=["Insufficient results for comprehensive analysis"],
                covered_aspects=[f"Limited information from {len(results)} source(s)"],
                needs_refinement=True,
                confidence=0.5
            )
        
        try:
            # Analyze query if not provided
            if query_complexity is None:
                query_complexity = self.query_analyzer.analyze_query(query)
            
            # Perform semantic coverage analysis
            semantic_coverage = self._analyze_semantic_coverage(query, results)
            
            # Perform aspect coverage analysis
            aspect_coverage = self._analyze_aspect_coverage(query, results, query_complexity)
            
            # Combine analyses
            combined_analysis = self._combine_coverage_analyses(
                semantic_coverage,
                aspect_coverage,
                query_complexity
            )
            
            logger.info(
                "Coverage evaluation completed",
                query_length=len(query),
                results_count=len(results),
                coverage_score=combined_analysis.score,
                missing_aspects_count=len(combined_analysis.missing_aspects),
                analysis_time=time.time() - start_time
            )
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Coverage evaluation failed: {e}")
            # Return conservative analysis on error
            return CoverageAnalysis(
                score=0.5,
                missing_aspects=["Analysis failed - manual review needed"],
                needs_refinement=True,
                confidence=0.3
            )
    
    def _analyze_semantic_coverage(
        self,
        query: str,
        results: List[SearchResult]
    ) -> CoverageAnalysis:
        """
        Analyze semantic coverage using embeddings.
        
        Computes how semantically similar the results are to the query
        to determine if the query intent is being addressed.
        """
        try:
            # Get embeddings for query and results
            query_embedding = self.embedding_client.get_embeddings([query])
            
            # Combine title and content for better representation
            result_texts = []
            for result in results:
                text = ""
                if result.title:
                    text += f"{result.title}. "
                text += truncate_text(result.content, max_length=500)
                result_texts.append(text)
            
            result_embeddings = self.embedding_client.get_embeddings(result_texts)
            
            # Calculate similarities
            similarities = self.embedding_client.compute_similarity(
                query_embedding, result_embeddings
            )[0]
            
            # Analyze similarity distribution
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            
            # Calculate coverage score
            # High coverage means most results are semantically relevant
            coverage_score = self._calculate_semantic_score(similarities)
            
            # Identify poorly covered aspects
            poor_matches = [
                f"Result {i+1} low relevance ({sim:.2f})"
                for i, sim in enumerate(similarities)
                if sim < 0.5
            ]
            
            well_matches = [
                f"High relevance content (similarity: {max_similarity:.2f})"
            ]
            
            return CoverageAnalysis(
                score=coverage_score,
                missing_aspects=poor_matches if poor_matches else [],
                covered_aspects=well_matches,
                confidence=max_similarity  # Confidence based on best match
            )
            
        except Exception as e:
            logger.error(f"Semantic coverage analysis failed: {e}")
            return CoverageAnalysis(score=0.5, confidence=0.3)
    
    def _calculate_semantic_score(self, similarities: np.ndarray) -> float:
        """Calculate coverage score from similarity values."""
        
        # Weight higher similarities more heavily
        weighted_similarities = similarities ** 2
        
        # Score based on:
        # 1. Average weighted similarity (50%)
        # 2. Presence of high-similarity results (30%) 
        # 3. Consistency across results (20%)
        
        avg_weighted = np.mean(weighted_similarities)
        high_sim_bonus = min(np.sum(similarities > 0.7) / len(similarities), 1.0)
        consistency = 1.0 - np.std(similarities)  # Lower std = more consistent
        
        score = (avg_weighted * 0.5 + 
                high_sim_bonus * 0.3 + 
                consistency * 0.2)
        
        return min(score, 1.0)
    
    def _analyze_aspect_coverage(
        self,
        query: str,
        results: List[SearchResult],
        query_complexity: QueryComplexity
    ) -> CoverageAnalysis:
        """
        Analyze coverage of different aspects/entities in the query.
        
        For complex queries, checks if different aspects are covered
        by analyzing keyword and entity presence in results.
        """
        # Extract expected aspects from query
        query_keywords = set(parse_search_keywords(query))
        query_entities = set(query_complexity.entities)
        expected_aspects = query_keywords.union(query_entities)
        
        if not expected_aspects:
            return CoverageAnalysis(score=0.7, confidence=0.5)  # Neutral for no aspects
        
        # Analyze aspect coverage in results
        covered_aspects = set()
        aspect_coverage_counts = Counter()
        
        for result in results:
            # Combine result text
            result_text = (result.title or "") + " " + result.content
            result_text_lower = result_text.lower()
            
            # Check for aspect presence
            for aspect in expected_aspects:
                if aspect.lower() in result_text_lower:
                    covered_aspects.add(aspect)
                    aspect_coverage_counts[aspect] += 1
        
        # Calculate coverage metrics
        coverage_ratio = len(covered_aspects) / len(expected_aspects)
        
        # Bonus for aspects covered by multiple sources
        multi_source_aspects = [
            aspect for aspect, count in aspect_coverage_counts.items()
            if count > 1
        ]
        multi_source_bonus = len(multi_source_aspects) / max(len(expected_aspects), 1)
        
        # Combine scores
        aspect_score = coverage_ratio * 0.7 + multi_source_bonus * 0.3
        
        # Identify missing and covered aspects
        missing_aspects = list(expected_aspects - covered_aspects)
        covered_list = [
            f"{aspect} (mentioned {aspect_coverage_counts[aspect]} times)"
            for aspect in covered_aspects
        ]
        
        return CoverageAnalysis(
            score=aspect_score,
            missing_aspects=missing_aspects,
            covered_aspects=covered_list,
            confidence=coverage_ratio
        )
    
    def _combine_coverage_analyses(
        self,
        semantic_analysis: CoverageAnalysis,
        aspect_analysis: CoverageAnalysis,
        query_complexity: QueryComplexity
    ) -> CoverageAnalysis:
        """
        Combine semantic and aspect coverage analyses.
        
        Different query types benefit from different analysis weights.
        """
        # Weight analyses based on query characteristics
        if query_complexity.complexity_level == "simple":
            # Simple queries rely more on semantic match
            semantic_weight = 0.7
            aspect_weight = 0.3
        elif query_complexity.intent_type == "factual":
            # Factual queries benefit from aspect coverage
            semantic_weight = 0.5
            aspect_weight = 0.5
        else:
            # Complex/exploratory queries need both
            semantic_weight = 0.6
            aspect_weight = 0.4
        
        # Combine scores
        combined_score = (
            semantic_analysis.score * semantic_weight +
            aspect_analysis.score * aspect_weight
        )
        
        # Combine missing aspects
        combined_missing = []
        if semantic_analysis.missing_aspects:
            combined_missing.extend(semantic_analysis.missing_aspects[:2])  # Limit
        if aspect_analysis.missing_aspects:
            combined_missing.extend(aspect_analysis.missing_aspects[:3])  # Limit
        
        # Combine covered aspects
        combined_covered = []
        if semantic_analysis.covered_aspects:
            combined_covered.extend(semantic_analysis.covered_aspects[:2])
        if aspect_analysis.covered_aspects:
            combined_covered.extend(aspect_analysis.covered_aspects[:3])
        
        # Determine if refinement is needed
        needs_refinement = (
            combined_score < self.coverage_threshold or
            len(combined_missing) > len(combined_covered)
        )
        
        # Combined confidence
        combined_confidence = (
            semantic_analysis.confidence * semantic_weight +
            aspect_analysis.confidence * aspect_weight
        )
        
        return CoverageAnalysis(
            score=combined_score,
            missing_aspects=combined_missing,
            covered_aspects=combined_covered,
            needs_refinement=needs_refinement,
            confidence=combined_confidence
        )
    
    def rank_results_by_quality(
        self,
        results: List[SearchResult],
        query: str = None
    ) -> List[Tuple[SearchResult, float]]:
        """
        Rank results by quality score.
        
        Args:
            results: List of search results to rank
            query: Optional query for relevance scoring
            
        Returns:
            List of (result, quality_score) tuples sorted by quality
        """
        if not results:
            return []
        
        scored_results = []
        
        for result in results:
            quality_score = self._calculate_quality_score(result, query)
            scored_results.append((result, quality_score))
        
        # Sort by quality score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results
    
    def _calculate_quality_score(
        self,
        result: SearchResult,
        query: str = None
    ) -> float:
        """
        Calculate a quality score for a search result.
        
        Considers multiple factors:
        - Content completeness
        - Source authority
        - Recency
        - Relevance (if query provided)
        """
        score = 0.0
        
        # Content completeness (25%)
        content_length = len(result.content.strip())
        content_score = min(content_length / 1000, 1.0) * 0.25
        score += content_score
        
        # Title presence (10%)
        if result.title and result.title.strip():
            score += 0.1
        
        # URL presence (5%)
        if result.url and result.url.strip():
            score += 0.05
        
        # Source authority (20%)
        authority_score = self._get_authority_score(result) * 0.2
        score += authority_score
        
        # Published date bonus (10%)
        if result.published_date:
            score += 0.1
        
        # Result score if available (15%)
        if hasattr(result, 'score') and result.score:
            normalized_score = min(result.score / 100, 1.0) * 0.15
            score += normalized_score
        
        # Relevance score if query provided (15%)
        if query:
            try:
                query_embedding = self.embedding_client.get_embeddings([query])
                result_text = (result.title or "") + " " + result.content[:500]
                result_embedding = self.embedding_client.get_embeddings([result_text])
                
                relevance = self.embedding_client.compute_similarity(
                    query_embedding, result_embedding
                )[0][0]
                
                score += relevance * 0.15
                
            except Exception as e:
                logger.debug(f"Failed to calculate relevance score: {e}")
                # Use keyword matching as fallback
                query_words = set(query.lower().split())
                result_words = set((result.content + " " + (result.title or "")).lower().split())
                
                overlap = len(query_words.intersection(result_words))
                keyword_relevance = min(overlap / len(query_words), 1.0)
                score += keyword_relevance * 0.15
        
        return min(score, 1.0)
    
    def _get_authority_score(self, result: SearchResult) -> float:
        """Calculate authority score for a result source."""
        if not result.source and not result.url:
            return 0.5
        
        source = (result.source or result.url or "").lower()
        
        # High authority domains
        high_authority = [
            'wikipedia', 'arxiv', 'nature', 'science', 'ieee', 'acm',
            'gov', 'edu', 'reuters', 'bbc', 'cnn', 'nytimes',
            'stanford', 'mit', 'harvard', 'oxford', 'cambridge'
        ]
        
        # Medium authority domains
        medium_authority = [
            'medium', 'forbes', 'techcrunch', 'wired', 'ars-technica',
            'stackoverflow', 'github', 'documentation'
        ]
        
        for authority in high_authority:
            if authority in source:
                return 1.0
        
        for authority in medium_authority:
            if authority in source:
                return 0.7
        
        # Check for academic indicators
        academic_indicators = ['journal', 'research', 'study', 'paper', 'conference']
        for indicator in academic_indicators:
            if indicator in source:
                return 0.8
        
        return 0.5  # Default authority
    
    def identify_result_gaps(
        self,
        coverage_analysis: CoverageAnalysis,
        query_complexity: QueryComplexity
    ) -> List[str]:
        """
        Identify specific gaps in results that need additional queries.
        
        Args:
            coverage_analysis: Results of coverage analysis
            query_complexity: Query complexity analysis
            
        Returns:
            List of specific gaps to address
        """
        gaps = []
        
        # Add missing aspects as gaps
        if coverage_analysis.missing_aspects:
            gaps.extend(coverage_analysis.missing_aspects[:3])  # Limit to top 3
        
        # Add complexity-specific gaps
        if query_complexity.complexity_level == "complex":
            if coverage_analysis.score < 0.8:
                gaps.append("detailed technical information")
                gaps.append("implementation examples")
        
        if query_complexity.intent_type == "comparative":
            if "comparison" not in " ".join(coverage_analysis.covered_aspects).lower():
                gaps.append("comparative analysis")
                gaps.append("pros and cons evaluation")
        
        if query_complexity.intent_type == "exploratory":
            if coverage_analysis.score < 0.7:
                gaps.append("expert opinions")
                gaps.append("different perspectives")
        
        if query_complexity.requires_recent_data:
            if "recent" not in " ".join(coverage_analysis.covered_aspects).lower():
                gaps.append("latest developments")
                gaps.append("current trends")
        
        # Remove duplicates while preserving order
        unique_gaps = list(dict.fromkeys(gaps))
        
        return unique_gaps[:5]  # Limit to 5 gaps
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get result evaluator statistics."""
        return {
            "coverage_threshold": self.coverage_threshold,
            "min_results_for_analysis": self.min_results_for_analysis,
            "embedding_client_stats": self.embedding_client.get_stats()
        }