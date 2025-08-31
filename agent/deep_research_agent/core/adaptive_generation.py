"""
Adaptive query generation for intelligent search optimization.

This module implements intelligent query generation that adapts based on
initial search results, coverage analysis, and query complexity to
improve search completeness and relevance.
"""

import re
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from .types import SearchResult, CoverageAnalysis, QueryComplexity
from .query_intelligence import QueryAnalyzer
from .result_evaluation import ResultEvaluator
from .logging import get_logger

logger = get_logger(__name__)


class AdaptationStrategy(str, Enum):
    """Strategies for adaptive query generation."""
    SPECIFICATION = "specification"  # Add specificity to broad queries
    EXPANSION = "expansion"         # Broaden narrow queries
    PIVOTING = "pivoting"          # Explore related aspects
    GAP_FILLING = "gap_filling"    # Target specific missing information
    REFRAMING = "reframing"        # Rephrase for better results


class AdaptiveQueryGenerator:
    """
    Generates adaptive follow-up queries based on initial search results.
    
    Analyzes the quality and coverage of initial results to determine
    the best strategy for generating additional queries that will
    improve overall search completeness.
    """
    
    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        result_evaluator: ResultEvaluator,
        max_adaptive_queries: int = 3,
        min_coverage_for_adaptation: float = 0.3
    ):
        """
        Initialize the adaptive query generator.
        
        Args:
            query_analyzer: Query analysis component
            result_evaluator: Result evaluation component
            max_adaptive_queries: Maximum adaptive queries to generate
            min_coverage_for_adaptation: Minimum coverage before adaptation
        """
        self.query_analyzer = query_analyzer
        self.result_evaluator = result_evaluator
        self.max_adaptive_queries = max_adaptive_queries
        self.min_coverage_for_adaptation = min_coverage_for_adaptation
        
        # Strategy templates for different adaptation approaches
        self.strategy_templates = {
            AdaptationStrategy.SPECIFICATION: [
                "{original_query} detailed explanation",
                "{original_query} step by step guide",
                "{original_query} technical implementation",
                "{entity} {original_query} comprehensive analysis"
            ],
            AdaptationStrategy.EXPANSION: [
                "{original_query} overview and fundamentals",
                "{original_query} related concepts and applications",
                "{original_query} broader context and background",
                "introduction to {original_query}"
            ],
            AdaptationStrategy.PIVOTING: [
                "{original_query} alternatives and options",
                "{original_query} related technologies and approaches",
                "{entity} comparison with alternatives",
                "{original_query} ecosystem and tools"
            ],
            AdaptationStrategy.GAP_FILLING: [
                "{original_query} {missing_aspect}",
                "{missing_aspect} in context of {original_query}",
                "{original_query} {missing_aspect} examples",
                "how {missing_aspect} relates to {original_query}"
            ],
            AdaptationStrategy.REFRAMING: [
                "what is {entity} and how does it work",
                "{entity} benefits and use cases",
                "{entity} best practices and recommendations",
                "getting started with {entity}"
            ]
        }
        
        logger.info(
            "Initialized adaptive query generator",
            max_queries=max_adaptive_queries,
            min_coverage=min_coverage_for_adaptation
        )
    
    def generate_adaptive_queries(
        self,
        original_query: str,
        initial_results: List[SearchResult],
        coverage_analysis: Optional[CoverageAnalysis] = None,
        query_complexity: Optional[QueryComplexity] = None
    ) -> List[str]:
        """
        Generate adaptive queries based on initial results and coverage.
        
        Args:
            original_query: Original user query
            initial_results: Results from initial search
            coverage_analysis: Optional pre-computed coverage analysis
            query_complexity: Optional pre-computed query complexity
            
        Returns:
            List of adaptive queries to improve coverage
        """
        if not original_query.strip():
            return []
        
        # Analyze query complexity if not provided
        if query_complexity is None:
            query_complexity = self.query_analyzer.analyze_query(original_query)
        
        # Evaluate coverage if not provided
        if coverage_analysis is None:
            coverage_analysis = self.result_evaluator.evaluate_coverage(
                original_query, initial_results, query_complexity
            )
        
        # Check if adaptation is needed
        if not self._should_generate_adaptive_queries(coverage_analysis, initial_results):
            logger.info("Coverage sufficient, no adaptive queries needed")
            return []
        
        # Determine adaptation strategy
        strategy = self._select_adaptation_strategy(
            coverage_analysis, query_complexity, initial_results
        )
        
        # Generate queries based on strategy
        adaptive_queries = self._generate_queries_for_strategy(
            strategy, original_query, coverage_analysis, query_complexity
        )
        
        # Filter and validate queries
        validated_queries = self._validate_and_filter_queries(
            adaptive_queries, original_query
        )
        
        logger.info(
            "Generated adaptive queries",
            original_query_length=len(original_query),
            strategy=strategy.value,
            generated_count=len(validated_queries),
            coverage_score=coverage_analysis.score
        )
        
        return validated_queries[:self.max_adaptive_queries]
    
    def _should_generate_adaptive_queries(
        self,
        coverage_analysis: CoverageAnalysis,
        initial_results: List[SearchResult]
    ) -> bool:
        """Determine if adaptive queries should be generated."""
        
        # Don't generate if no initial results
        if not initial_results:
            return False
        
        # Don't generate if coverage is very low (suggests fundamental issue)
        if coverage_analysis.score < self.min_coverage_for_adaptation:
            return False
        
        # Generate if coverage indicates refinement is needed
        if coverage_analysis.needs_refinement:
            return True
        
        # Generate if there are significant missing aspects
        if len(coverage_analysis.missing_aspects) > 2:
            return True
        
        # Generate if coverage is below threshold
        if coverage_analysis.score < 0.75:
            return True
        
        return False
    
    def _select_adaptation_strategy(
        self,
        coverage_analysis: CoverageAnalysis,
        query_complexity: QueryComplexity,
        initial_results: List[SearchResult]
    ) -> AdaptationStrategy:
        """
        Select the best adaptation strategy based on analysis.
        
        Different strategies are optimal for different situations:
        - Specification: When query is too broad
        - Expansion: When query is too narrow  
        - Pivoting: When need different angles
        - Gap filling: When specific aspects are missing
        - Reframing: When results are poor quality
        """
        
        # Gap filling if specific aspects are missing
        if (coverage_analysis.missing_aspects and 
            len(coverage_analysis.missing_aspects) <= 3):
            return AdaptationStrategy.GAP_FILLING
        
        # Reframing if results are very poor
        if coverage_analysis.score < 0.4:
            return AdaptationStrategy.REFRAMING
        
        # Strategy based on query complexity
        if query_complexity.complexity_level == "simple":
            # Simple queries often need more specificity
            if coverage_analysis.score < 0.6:
                return AdaptationStrategy.SPECIFICATION
            else:
                return AdaptationStrategy.EXPANSION
        
        elif query_complexity.complexity_level == "complex":
            # Complex queries might need different angles
            if len(initial_results) < 5:
                return AdaptationStrategy.EXPANSION
            else:
                return AdaptationStrategy.PIVOTING
        
        elif query_complexity.complexity_level == "compound":
            # Compound queries need gap filling for different parts
            return AdaptationStrategy.GAP_FILLING
        
        # Strategy based on intent
        if query_complexity.intent_type == "comparative":
            return AdaptationStrategy.PIVOTING
        elif query_complexity.intent_type == "exploratory":
            return AdaptationStrategy.EXPANSION
        else:  # factual
            return AdaptationStrategy.SPECIFICATION
    
    def _generate_queries_for_strategy(
        self,
        strategy: AdaptationStrategy,
        original_query: str,
        coverage_analysis: CoverageAnalysis,
        query_complexity: QueryComplexity
    ) -> List[str]:
        """Generate queries for a specific adaptation strategy."""
        
        templates = self.strategy_templates.get(strategy, [])
        if not templates:
            return []
        
        generated_queries = []
        
        # Prepare context variables
        context = {
            "original_query": original_query.strip(),
            "entity": query_complexity.entities[0] if query_complexity.entities else "",
            "missing_aspect": (coverage_analysis.missing_aspects[0] 
                             if coverage_analysis.missing_aspects else "details")
        }
        
        # Generate queries from templates
        for template in templates:
            try:
                query = template.format(**context)
                
                # Clean up the generated query
                query = self._clean_generated_query(query)
                
                if query and len(query.strip()) > 5:  # Minimum length check
                    generated_queries.append(query)
                    
            except (KeyError, ValueError) as e:
                logger.debug(f"Template formatting failed: {e}")
                continue
        
        # Add strategy-specific custom queries
        custom_queries = self._generate_custom_queries_for_strategy(
            strategy, original_query, coverage_analysis, query_complexity
        )
        
        generated_queries.extend(custom_queries)
        
        return generated_queries
    
    def _generate_custom_queries_for_strategy(
        self,
        strategy: AdaptationStrategy,
        original_query: str,
        coverage_analysis: CoverageAnalysis,
        query_complexity: QueryComplexity
    ) -> List[str]:
        """Generate custom queries based on specific strategy needs."""
        
        custom_queries = []
        
        if strategy == AdaptationStrategy.GAP_FILLING:
            # Generate specific gap-filling queries
            for missing_aspect in coverage_analysis.missing_aspects[:2]:
                if query_complexity.entities:
                    entity = query_complexity.entities[0]
                    custom_queries.append(f"{entity} {missing_aspect} comprehensive guide")
                else:
                    custom_queries.append(f"{original_query} {missing_aspect} explanation")
        
        elif strategy == AdaptationStrategy.SPECIFICATION:
            # Add temporal and technical specificity
            if query_complexity.requires_recent_data:
                custom_queries.append(f"{original_query} 2024 latest updates")
            
            if query_complexity.entities:
                entity = query_complexity.entities[0]
                custom_queries.append(f"{entity} technical specifications and requirements")
        
        elif strategy == AdaptationStrategy.EXPANSION:
            # Broaden the scope
            if query_complexity.entities:
                entity = query_complexity.entities[0]
                custom_queries.append(f"{entity} ecosystem and related technologies")
                custom_queries.append(f"introduction to {entity} for beginners")
        
        elif strategy == AdaptationStrategy.PIVOTING:
            # Explore alternatives and related concepts
            if query_complexity.entities:
                entity = query_complexity.entities[0]
                custom_queries.append(f"alternatives to {entity}")
                custom_queries.append(f"{entity} vs competitors comparison")
        
        elif strategy == AdaptationStrategy.REFRAMING:
            # Completely different approach
            if query_complexity.entities:
                entity = query_complexity.entities[0]
                custom_queries.append(f"what is {entity} used for")
                custom_queries.append(f"how to get started with {entity}")
        
        return custom_queries
    
    def _clean_generated_query(self, query: str) -> str:
        """Clean and normalize a generated query."""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove empty placeholder artifacts
        query = re.sub(r'\s+\s+', ' ', query)  # Multiple spaces
        query = re.sub(r'\b\s+\b', ' ', query)  # Spaces around word boundaries
        
        # Remove queries that are too similar to original
        # (This would be enhanced with similarity checking)
        
        # Ensure proper capitalization
        if query and not query[0].isupper():
            query = query[0].upper() + query[1:]
        
        return query
    
    def _validate_and_filter_queries(
        self,
        queries: List[str],
        original_query: str
    ) -> List[str]:
        """Validate and filter generated queries."""
        
        validated_queries = []
        seen_queries = {original_query.lower().strip()}
        
        for query in queries:
            # Basic validation
            if not query or len(query.strip()) < 5:
                continue
            
            query_clean = query.strip()
            query_lower = query_clean.lower()
            
            # Remove duplicates (case-insensitive)
            if query_lower in seen_queries:
                continue
            
            # Remove queries too similar to original (simple check)
            if self._queries_too_similar(query_lower, original_query.lower()):
                continue
            
            # Remove malformed queries
            if not self._is_well_formed_query(query_clean):
                continue
            
            validated_queries.append(query_clean)
            seen_queries.add(query_lower)
        
        return validated_queries
    
    def _queries_too_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are too similar (simple implementation)."""
        
        # Split into words
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        # Calculate overlap
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))
        
        similarity = overlap / total_unique if total_unique > 0 else 0
        
        # Consider too similar if > 80% overlap
        return similarity > 0.8
    
    def _is_well_formed_query(self, query: str) -> bool:
        """Check if a query is well-formed."""
        
        # Must have some content
        if len(query.strip()) < 5:
            return False
        
        # Should not be all special characters
        if re.match(r'^[^a-zA-Z0-9]*$', query):
            return False
        
        # Should not have obvious template artifacts
        artifacts = ['{', '}', '{{', '}}', 'None', 'null']
        for artifact in artifacts:
            if artifact in query:
                return False
        
        # Should have at least one word
        if not re.search(r'\b\w+\b', query):
            return False
        
        return True
    
    def analyze_adaptation_effectiveness(
        self,
        original_query: str,
        original_coverage: CoverageAnalysis,
        adaptive_queries: List[str],
        new_results: List[SearchResult],
        new_coverage: CoverageAnalysis
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of adaptive query generation.
        
        Args:
            original_query: Original user query
            original_coverage: Coverage before adaptation
            adaptive_queries: Generated adaptive queries
            new_results: Results from adaptive queries
            new_coverage: Coverage after adaptation
            
        Returns:
            Dictionary with effectiveness metrics
        """
        
        coverage_improvement = new_coverage.score - original_coverage.score
        
        # Analyze gap filling
        original_gaps = set(original_coverage.missing_aspects)
        new_gaps = set(new_coverage.missing_aspects)
        filled_gaps = original_gaps - new_gaps
        
        # Analyze result quality improvement
        quality_scores = [
            self.result_evaluator._calculate_quality_score(result, original_query)
            for result in new_results
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "coverage_improvement": coverage_improvement,
            "coverage_improvement_percentage": (coverage_improvement / max(original_coverage.score, 0.01)) * 100,
            "gaps_filled": len(filled_gaps),
            "gaps_filled_list": list(filled_gaps),
            "new_gaps_introduced": len(new_gaps - original_gaps),
            "adaptive_queries_count": len(adaptive_queries),
            "new_results_count": len(new_results),
            "average_result_quality": avg_quality,
            "confidence_improvement": new_coverage.confidence - original_coverage.confidence,
            "adaptation_successful": coverage_improvement > 0.1 and len(filled_gaps) > 0
        }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get adaptive query generator statistics."""
        return {
            "max_adaptive_queries": self.max_adaptive_queries,
            "min_coverage_for_adaptation": self.min_coverage_for_adaptation,
            "available_strategies": [strategy.value for strategy in AdaptationStrategy],
            "template_counts": {
                strategy.value: len(templates) 
                for strategy, templates in self.strategy_templates.items()
            }
        }