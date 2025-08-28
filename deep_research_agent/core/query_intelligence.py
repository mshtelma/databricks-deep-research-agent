"""
Query intelligence and analysis for adaptive search optimization.

This module provides intelligent analysis of user queries to understand
their complexity, intent, and characteristics, enabling better search
strategy selection and adaptive query generation.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime, timedelta

from .types import QueryComplexity
from .logging import get_logger
from .utils import parse_search_keywords

logger = get_logger(__name__)


class QueryAnalyzer:
    """
    Intelligent query analyzer for understanding user intent and complexity.
    
    Analyzes queries to determine:
    - Complexity level (simple, complex, compound)
    - Intent type (factual, comparative, exploratory)
    - Key entities and concepts
    - Expected result characteristics
    """
    
    def __init__(self):
        """Initialize the query analyzer with patterns and keywords."""
        
        # Complexity indicators
        self.complex_patterns = [
            r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b',  # Comparison
            r'\bdifference[s]?\b', r'\bsimilar(ities)?\b',  # Comparison
            r'\badvantages?\b', r'\bdisadvantages?\b',  # Pros/cons
            r'\bbest\b', r'\bworst\b', r'\btop\b',  # Ranking
            r'\bhow to\b', r'\bstep[s]?\b', r'\bprocess\b',  # Process
            r'\bwhy\b', r'\bbecause\b', r'\breason[s]?\b',  # Causal
            r'\bimpact\b', r'\beffect[s]?\b', r'\bconsequence[s]?\b',  # Impact
            r'\btrend[s]?\b', r'\bfuture\b', r'\bprediction[s]?\b',  # Trends
        ]
        
        # Compound query indicators
        self.compound_patterns = [
            r'\band\b', r'\bor\b', r'\bbut\b', r'\balso\b',  # Conjunctions
            r';', r'\?.*\?',  # Multiple questions
            r'\bmultiple\b', r'\bseveral\b', r'\bvarious\b',  # Multiple aspects
        ]
        
        # Intent classification patterns
        self.factual_patterns = [
            r'^what is\b', r'^who is\b', r'^where is\b', r'^when is\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexplain\b',
            r'\bfact[s]?\b', r'\binformation\b', r'\bdetail[s]?\b',
        ]
        
        self.comparative_patterns = [
            r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b', r'\bbetter\b',
            r'\bdifference\b', r'\bsimilar\b', r'\bcontrast\b',
            r'\bbest\b', r'\bworst\b', r'\btop\b', r'\branking\b',
        ]
        
        self.exploratory_patterns = [
            r'\bhow\b', r'\bwhy\b', r'\bshould\b', r'\bcould\b',
            r'\bmight\b', r'\bwould\b', r'\bpossible\b',
            r'\btrend[s]?\b', r'\bfuture\b', r'\bpotential\b',
            r'\bopinion[s]?\b', r'\bview[s]?\b', r'\bperspective[s]?\b',
        ]
        
        # Temporal indicators
        self.recent_indicators = [
            r'\blatest\b', r'\brecent\b', r'\bcurrent\b', r'\bnew\b',
            r'\btoday\b', r'\bnow\b', r'\bthis year\b', r'\b2024\b',
            r'\bupdate[d]?\b', r'\bbreaking\b', r'\bjust\b',
        ]
        
        # Entity extraction patterns
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Simple name pattern
            'organization': r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc|Corp|LLC|Ltd|Company)\b',
            'technology': r'\b(?:AI|ML|API|SDK|iOS|Android|Python|JavaScript|React|Vue|Angular)\b',
            'date': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        }
        
        logger.info("Initialized query analyzer")
    
    def analyze_query(self, query: str) -> QueryComplexity:
        """
        Perform comprehensive analysis of a user query.
        
        Args:
            query: User query string
            
        Returns:
            QueryComplexity object with analysis results
        """
        if not query or not query.strip():
            return QueryComplexity()
        
        query_clean = query.strip().lower()
        
        # Determine complexity level
        complexity_level = self._classify_complexity(query_clean)
        
        # Determine intent type
        intent_type = self._classify_intent(query_clean)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine expected sources
        expected_sources = self._estimate_source_count(complexity_level, intent_type)
        
        # Check if recent data is needed
        requires_recent_data = self._requires_recent_data(query_clean)
        
        analysis = QueryComplexity(
            complexity_level=complexity_level,
            intent_type=intent_type,
            entities=entities,
            expected_sources=expected_sources,
            requires_recent_data=requires_recent_data
        )
        
        logger.info(
            "Analyzed query",
            complexity=complexity_level,
            intent=intent_type,
            entities_count=len(entities),
            expected_sources=expected_sources,
            requires_recent=requires_recent_data
        )
        
        return analysis
    
    def _classify_complexity(self, query: str) -> str:
        """Classify query complexity level."""
        
        # Check for compound indicators
        compound_score = sum(1 for pattern in self.compound_patterns 
                           if re.search(pattern, query, re.IGNORECASE))
        
        if compound_score >= 2:
            return "compound"
        
        # Check for complex indicators
        complex_score = sum(1 for pattern in self.complex_patterns 
                          if re.search(pattern, query, re.IGNORECASE))
        
        if complex_score >= 1 or len(query.split()) > 10:
            return "complex"
        
        return "simple"
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent type."""
        
        # Count pattern matches for each intent type
        factual_score = sum(1 for pattern in self.factual_patterns 
                           if re.search(pattern, query, re.IGNORECASE))
        
        comparative_score = sum(1 for pattern in self.comparative_patterns 
                               if re.search(pattern, query, re.IGNORECASE))
        
        exploratory_score = sum(1 for pattern in self.exploratory_patterns 
                               if re.search(pattern, query, re.IGNORECASE))
        
        # Determine intent based on highest score
        scores = {
            'factual': factual_score,
            'comparative': comparative_score,
            'exploratory': exploratory_score
        }
        
        intent = max(scores.items(), key=lambda x: x[1])[0]
        
        # Default to factual if no clear intent
        if scores[intent] == 0:
            intent = 'factual'
        
        return intent
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query."""
        entities = []
        
        # Extract using patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Extract keywords as entities
        keywords = parse_search_keywords(query)
        
        # Filter and combine
        all_entities = list(set(entities + keywords[:5]))  # Top 5 keywords
        
        # Remove common words that aren't really entities
        stop_entities = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through'
        }
        
        filtered_entities = [
            entity for entity in all_entities 
            if entity.lower() not in stop_entities and len(entity) > 2
        ]
        
        return filtered_entities[:10]  # Limit to top 10
    
    def _estimate_source_count(self, complexity: str, intent: str) -> int:
        """Estimate optimal number of sources based on complexity and intent."""
        
        base_count = {
            'simple': 3,
            'complex': 5,
            'compound': 7
        }.get(complexity, 3)
        
        # Adjust based on intent
        intent_adjustments = {
            'factual': 0,
            'comparative': 2,  # Need multiple sources for comparison
            'exploratory': 1   # Need diverse perspectives
        }
        
        adjustment = intent_adjustments.get(intent, 0)
        
        return min(base_count + adjustment, 8)  # Cap at 8 sources
    
    def _requires_recent_data(self, query: str) -> bool:
        """Determine if the query requires recent/current information."""
        
        # Check for temporal indicators
        recent_score = sum(1 for pattern in self.recent_indicators 
                          if re.search(pattern, query, re.IGNORECASE))
        
        # Check for specific recent years
        current_year = datetime.now().year
        recent_years = [str(year) for year in range(current_year - 1, current_year + 1)]
        
        for year in recent_years:
            if year in query:
                recent_score += 1
        
        return recent_score > 0
    
    def generate_query_variations(self, original_query: str, count: int = 3) -> List[str]:
        """
        Generate variations of the original query for broader search coverage.
        
        Args:
            original_query: Original user query
            count: Number of variations to generate
            
        Returns:
            List of query variations
        """
        analysis = self.analyze_query(original_query)
        variations = []
        
        # Strategy 1: Add specificity
        if analysis.complexity_level == "simple":
            if analysis.entities:
                entity = analysis.entities[0]
                variations.append(f"{original_query} {entity} detailed information")
        
        # Strategy 2: Broaden scope
        if analysis.intent_type == "factual":
            variations.append(f"{original_query} examples applications")
        
        # Strategy 3: Add temporal context
        if analysis.requires_recent_data:
            variations.append(f"{original_query} {datetime.now().year}")
        else:
            variations.append(f"{original_query} latest developments")
        
        # Strategy 4: Add context for complex queries
        if analysis.complexity_level in ["complex", "compound"]:
            variations.append(f"{original_query} comprehensive guide")
        
        # Strategy 5: Alternative phrasing
        # Simple synonym replacement
        synonyms = {
            'how to': 'methods for',
            'what is': 'definition of',
            'best': 'top',
            'compare': 'difference between',
            'advantages': 'benefits'
        }
        
        varied_query = original_query.lower()
        for original, replacement in synonyms.items():
            if original in varied_query:
                variations.append(varied_query.replace(original, replacement))
                break
        
        # Remove duplicates and limit count
        unique_variations = list(dict.fromkeys(variations))  # Preserve order
        
        return unique_variations[:count]
    
    def suggest_follow_up_queries(
        self, 
        original_query: str, 
        coverage_gaps: List[str]
    ) -> List[str]:
        """
        Suggest follow-up queries based on identified coverage gaps.
        
        Args:
            original_query: Original user query
            coverage_gaps: List of missing aspects
            
        Returns:
            List of suggested follow-up queries
        """
        analysis = self.analyze_query(original_query)
        follow_ups = []
        
        # Generate targeted queries for each gap
        for gap in coverage_gaps:
            if analysis.entities:
                # Use entities to create specific queries
                entity = analysis.entities[0]
                follow_ups.append(f"{entity} {gap} detailed explanation")
            else:
                # Generic gap-filling query
                follow_ups.append(f"{original_query} {gap}")
        
        # Add queries based on complexity and intent
        if analysis.complexity_level == "simple" and analysis.intent_type == "factual":
            # Simple factual queries benefit from examples
            follow_ups.append(f"{original_query} real world examples")
            follow_ups.append(f"{original_query} practical applications")
        
        elif analysis.intent_type == "comparative":
            # Comparative queries benefit from criteria
            follow_ups.append(f"{original_query} evaluation criteria")
            follow_ups.append(f"{original_query} pros and cons")
        
        elif analysis.intent_type == "exploratory":
            # Exploratory queries benefit from different perspectives
            follow_ups.append(f"{original_query} expert opinions")
            follow_ups.append(f"{original_query} case studies")
        
        # Remove duplicates and limit
        unique_follow_ups = list(dict.fromkeys(follow_ups))
        
        return unique_follow_ups[:3]  # Limit to 3 follow-up queries
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get query analyzer statistics."""
        return {
            "patterns": {
                "complex_patterns": len(self.complex_patterns),
                "compound_patterns": len(self.compound_patterns),
                "factual_patterns": len(self.factual_patterns),
                "comparative_patterns": len(self.comparative_patterns),
                "exploratory_patterns": len(self.exploratory_patterns),
                "recent_indicators": len(self.recent_indicators),
            },
            "entity_types": list(self.entity_patterns.keys())
        }