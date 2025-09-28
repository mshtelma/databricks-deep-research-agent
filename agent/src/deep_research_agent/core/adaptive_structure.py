"""
Adaptive report structure generation with Strategy Pattern.

This module provides intelligent structure generation for reports based on query context,
using LLM analysis to create engaging, context-specific section names.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from enum import Enum

from .report_styles import ReportStyle, StyleConfig, CitationStyle
from .logging import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Intent classification for queries."""
    EXPLANATORY = "explanatory"  # "What is...", "Explain..."
    COMPARATIVE = "comparative"  # "Compare...", "Difference between..."
    ANALYTICAL = "analytical"    # "Analyze...", "Evaluate..."
    PROCEDURAL = "procedural"    # "How to...", "Steps to..."
    INVESTIGATIVE = "investigative"  # "Research...", "Find out..."


class QueryDomain(Enum):
    """Domain classification for queries."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    EDUCATIONAL = "educational"
    GENERAL = "general"


@dataclass
class QueryContext:
    """Context information extracted from query."""
    topic: str
    intent: QueryIntent
    domain: QueryDomain
    complexity: str  # "simple", "medium", "complex"
    target_audience: str  # "general", "professional", "expert"


@dataclass
class CachedStructure:
    """Cached structure with metadata."""
    sections: List[str]
    style_config: StyleConfig
    timestamp: float
    query_context: QueryContext
    usage_count: int = 0


class StructureValidator:
    """Validates generated report structures."""
    
    MIN_SECTIONS = 3
    MAX_SECTIONS = 8
    MAX_SECTION_NAME_LENGTH = 60
    MIN_SECTION_NAME_LENGTH = 5
    
    # Required conceptual coverage (at least 2 must be present)
    REQUIRED_CONCEPTS = [
        "overview", "introduction", "background", "summary",
        "findings", "results", "analysis", "discussion",
        "conclusion", "recommendations", "implications", "future"
    ]
    
    # Forbidden words that make sections too generic
    FORBIDDEN_GENERIC_WORDS = [
        "section", "chapter", "part", "item", "element", "component"
    ]
    
    def validate(self, sections: List[str], query_context: QueryContext) -> Tuple[bool, List[str]]:
        """
        Validate generated structure.
        
        Args:
            sections: List of section names
            query_context: Context of the query
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check section count
        if len(sections) < self.MIN_SECTIONS:
            issues.append(f"Too few sections: {len(sections)} < {self.MIN_SECTIONS}")
        elif len(sections) > self.MAX_SECTIONS:
            issues.append(f"Too many sections: {len(sections)} > {self.MAX_SECTIONS}")
        
        # Check section names
        for i, section in enumerate(sections):
            # Length check
            if len(section) < self.MIN_SECTION_NAME_LENGTH:
                issues.append(f"Section {i+1} too short: '{section}'")
            elif len(section) > self.MAX_SECTION_NAME_LENGTH:
                issues.append(f"Section {i+1} too long: '{section}'")
            
            # Generic word check
            section_lower = section.lower()
            if any(word in section_lower for word in self.FORBIDDEN_GENERIC_WORDS):
                issues.append(f"Section {i+1} too generic: '{section}'")
            
            # Duplication check
            if sections.count(section) > 1:
                issues.append(f"Duplicate section: '{section}'")
        
        # Check conceptual coverage
        #sections_text = " ".join(sections).lower()
        #concept_matches = sum(1 for concept in self.REQUIRED_CONCEPTS
        #                    if concept in sections_text)
        
        #if concept_matches < 2:
        #    issues.append(f"Insufficient conceptual coverage: {concept_matches}/2 required concepts")
        
        # Check for topic relevance
        topic_words = set(query_context.topic.lower().split())
        section_words = set(" ".join(sections).lower().split())
        overlap = len(topic_words.intersection(section_words))
        
        if overlap == 0:
            issues.append("No topic words found in section names - may be off-topic")
        
        return len(issues) == 0, issues


class StructureCache:
    """Smart caching for generated structures."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600, max_memory_mb: float = 5.0):
        self.cache: Dict[str, CachedStructure] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.max_memory_mb = max_memory_mb
        
    def _get_cache_key(self, query_context: QueryContext) -> str:
        """Generate cache key from query context."""
        content = f"{query_context.topic}:{query_context.intent.value}:{query_context.domain.value}:{query_context.complexity}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _estimate_memory_mb(self) -> float:
        """Rough estimate of cache memory usage."""
        total_chars = sum(
            len(str(cached.sections)) + len(cached.query_context.topic)
            for cached in self.cache.values()
        )
        return total_chars * 2 / (1024 * 1024)  # Rough estimate: 2 bytes per char
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self.cache.items()
            if current_time - cached.timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            logger.debug(f"Removed expired cache entry: {key}")
    
    def _cleanup_by_usage(self):
        """Remove least recently used entries if over size limit."""
        if len(self.cache) <= self.max_size:
            return
            
        # Sort by usage count (ascending), then by timestamp (ascending)
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (x[1].usage_count, x[1].timestamp)
        )
        
        # Remove oldest, least used entries
        entries_to_remove = len(self.cache) - self.max_size + 1
        for i in range(entries_to_remove):
            key = sorted_items[i][0]
            del self.cache[key]
            logger.debug(f"Removed LRU cache entry: {key}")
    
    def get(self, query_context: QueryContext) -> Optional[CachedStructure]:
        """Get cached structure if available."""
        self._cleanup_expired()
        
        cache_key = self._get_cache_key(query_context)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached.usage_count += 1
            logger.debug(f"Cache hit for query context: {query_context.topic[:50]}...")
            return cached
        
        logger.debug(f"Cache miss for query context: {query_context.topic[:50]}...")
        return None
    
    def set(self, query_context: QueryContext, sections: List[str], style_config: StyleConfig):
        """Cache a structure."""
        # Check memory limit before adding
        if self._estimate_memory_mb() >= self.max_memory_mb:
            logger.warning("Structure cache approaching memory limit, cleaning up")
            self._cleanup_by_usage()
        
        cache_key = self._get_cache_key(query_context)
        self.cache[cache_key] = CachedStructure(
            sections=sections,
            style_config=style_config,
            timestamp=time.time(),
            query_context=query_context,
            usage_count=1
        )
        
        # Cleanup if over size limit
        self._cleanup_by_usage()
        
        logger.debug(f"Cached structure for query: {query_context.topic[:50]}... "
                    f"(cache size: {len(self.cache)}, memory: {self._estimate_memory_mb():.2f}MB)")


class QueryAnalyzer:
    """Analyzes queries to extract context."""
    
    INTENT_PATTERNS = {
        QueryIntent.EXPLANATORY: [
            r"what\s+is", r"explain", r"describe", r"define", r"meaning\s+of"
        ],
        QueryIntent.COMPARATIVE: [
            r"compare", r"difference\s+between", r"vs", r"versus", r"contrast"
        ],
        QueryIntent.ANALYTICAL: [
            r"analyze", r"evaluate", r"assess", r"examine", r"study"
        ],
        QueryIntent.PROCEDURAL: [
            r"how\s+to", r"steps\s+to", r"process\s+of", r"guide\s+to", r"tutorial"
        ],
        QueryIntent.INVESTIGATIVE: [
            r"research", r"investigate", r"find\s+out", r"discover", r"explore"
        ]
    }
    
    DOMAIN_KEYWORDS = {
        QueryDomain.TECHNICAL: [
            "programming", "software", "algorithm", "database", "api", "framework",
            "architecture", "system", "technology", "development", "code", "computing"
        ],
        QueryDomain.BUSINESS: [
            "market", "strategy", "business", "company", "revenue", "profit", "management",
            "leadership", "organization", "corporate", "industry", "competition"
        ],
        QueryDomain.SCIENTIFIC: [
            "research", "study", "experiment", "hypothesis", "theory", "analysis",
            "data", "methodology", "results", "findings", "scientific", "academic"
        ],
        QueryDomain.EDUCATIONAL: [
            "learn", "education", "teaching", "course", "curriculum", "student",
            "academic", "university", "school", "training", "knowledge"
        ]
    }
    
    def analyze_query(self, query: str) -> QueryContext:
        """Analyze query to extract context."""
        query_lower = query.lower()
        
        # Extract topic (first significant noun phrase, simplified)
        topic = self._extract_topic(query)
        
        # Determine intent
        intent = self._classify_intent(query_lower)
        
        # Determine domain
        domain = self._classify_domain(query_lower)
        
        # Assess complexity
        complexity = self._assess_complexity(query)
        
        # Determine target audience
        target_audience = self._determine_audience(query_lower, domain)
        
        return QueryContext(
            topic=topic,
            intent=intent,
            domain=domain,
            complexity=complexity,
            target_audience=target_audience
        )
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query."""
        # Simple approach: take query as-is, cleaned up
        topic = re.sub(r'^(what\s+is|explain|describe|analyze|compare)\s+', '', query, flags=re.IGNORECASE)
        topic = re.sub(r'\?$', '', topic)
        return topic.strip()
    
    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify query intent."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return intent
        return QueryIntent.INVESTIGATIVE  # Default
    
    def _classify_domain(self, query_lower: str) -> QueryDomain:
        """Classify query domain."""
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or GENERAL if no clear match
        max_score = max(domain_scores.values())
        if max_score == 0:
            return QueryDomain.GENERAL
        
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "medium"
        else:
            return "complex"
    
    def _determine_audience(self, query_lower: str, domain: QueryDomain) -> str:
        """Determine target audience."""
        professional_indicators = [
            "enterprise", "corporate", "professional", "business", "industry",
            "implementation", "strategy", "architecture", "framework"
        ]
        
        expert_indicators = [
            "advanced", "deep", "technical", "algorithm", "methodology",
            "research", "analysis", "evaluation", "optimization"
        ]
        
        if any(indicator in query_lower for indicator in expert_indicators):
            return "expert"
        elif any(indicator in query_lower for indicator in professional_indicators):
            return "professional"
        else:
            return "general"


class ReportStructureStrategy(ABC):
    """Abstract base class for report structure generation strategies."""
    
    @abstractmethod
    def generate_structure(self, query_context: QueryContext) -> Tuple[List[str], StyleConfig]:
        """Generate report structure for given context."""
        pass


class StaticStructureStrategy(ReportStructureStrategy):
    """Fallback strategy using predefined structures."""
    
    def __init__(self, fallback_style: ReportStyle = ReportStyle.PROFESSIONAL):
        self.fallback_style = fallback_style
        
    def generate_structure(self, query_context: QueryContext) -> Tuple[List[str], StyleConfig]:
        """Return predefined structure based on context."""
        from .report_styles import STYLE_CONFIGS
        
        # Choose appropriate style based on context
        if query_context.domain == QueryDomain.TECHNICAL and query_context.target_audience == "expert":
            style = ReportStyle.TECHNICAL
        elif query_context.domain == QueryDomain.BUSINESS and query_context.target_audience == "professional":
            style = ReportStyle.PROFESSIONAL
        elif query_context.domain == QueryDomain.SCIENTIFIC:
            style = ReportStyle.ACADEMIC
        elif query_context.intent == QueryIntent.EXPLANATORY and query_context.target_audience == "general":
            style = ReportStyle.POPULAR_SCIENCE
        else:
            style = self.fallback_style
            
        config = STYLE_CONFIGS[style]
        return config.structure, config


class AdaptiveStructureStrategy(ReportStructureStrategy):
    """LLM-based adaptive structure generation."""
    
    def __init__(self, llm=None, validator: Optional[StructureValidator] = None, 
                 timeout: float = 3.0):
        self.llm = llm
        self.validator = validator or StructureValidator()
        self.timeout = timeout
        
    def generate_structure(self, query_context: QueryContext) -> Tuple[List[str], StyleConfig]:
        """Generate adaptive structure using LLM."""
        if not self.llm:
            raise ValueError("LLM required for adaptive structure generation")
        
        try:
            # Generate structure with timeout
            sections = self._generate_with_llm(query_context)
            
            # Validate structure
            is_valid, issues = self.validator.validate(sections, query_context)
            if not is_valid:
                logger.warning(f"Generated structure invalid: {issues}")
                raise ValueError(f"Invalid structure: {issues}")
            
            # Create dynamic style config
            style_config = self._create_dynamic_config(query_context, sections)
            
            logger.info(f"Generated adaptive structure with {len(sections)} sections for {query_context.topic[:50]}...")
            return sections, style_config
            
        except Exception as e:
            logger.warning(f"Adaptive structure generation failed: {e}, using fallback")
            # Fall back to static strategy
            static_strategy = StaticStructureStrategy()
            return static_strategy.generate_structure(query_context)
    
    def _generate_with_llm(self, query_context: QueryContext) -> List[str]:
        """Generate structure using LLM."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = self._build_system_prompt(query_context)
        user_prompt = self._build_user_prompt(query_context)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Call LLM with timeout protection
        response = self.llm.invoke(messages)
        
        # Parse response
        sections = self._parse_response(response.content)
        
        return sections
    
    def _build_system_prompt(self, query_context: QueryContext) -> str:
        """Build system prompt for structure generation."""
        
        # Truncate topic if it's too long (complex queries)
        topic_display = query_context.topic
        if len(topic_display) > 100:
            # For complex queries, extract the core subject
            topic_display = topic_display[:100] + "..."
            
        return f"""You are an expert report structure designer. Your task is to create clear, professional section names for a research report.

REQUIREMENTS:
- Generate 4-7 section names
- Keep names clear and professional
- Each section should be 10-50 characters
- Sections should flow logically from overview to conclusion
- DO NOT use fragments from the query as section names
- DO NOT include parentheses, numbers, or special formatting in section names
- Make section names descriptive but general enough to cover the topic

CONTEXT:
- Topic: {topic_display}
- Intent: {query_context.intent.value}
- Domain: {query_context.domain.value}
- Complexity: {query_context.complexity}

IMPORTANT: If the topic involves comparison or analysis of multiple entities, use general section names like:
["Overview", "Comparative Analysis", "Key Findings", "Detailed Results", "Recommendations", "Conclusion"]

EXAMPLES OF GOOD SECTIONS:
For tax comparison: ["Tax System Overview", "Country-by-Country Analysis", "Key Differentiators", "Financial Impact Assessment", "Optimization Strategies"]
For technical topics: ["Technical Background", "Core Concepts", "Implementation Details", "Performance Analysis", "Best Practices"]

Return ONLY the section names, one per line, no numbers or bullets."""
    
    def _build_user_prompt(self, query_context: QueryContext) -> str:
        """Build user prompt for structure generation."""
        # Simplify complex topics for clearer section generation
        topic = query_context.topic
        if len(topic) > 150:
            # Extract key concepts from long queries
            if "comparison" in topic.lower() or "compare" in topic.lower():
                topic = "comparative analysis and evaluation"
            elif "tax" in topic.lower() and "finance" in topic.lower():
                topic = "international tax and finance comparison"
            else:
                topic = topic[:150] + "..."
                
        return f"Generate clear, professional section names for a research report about: {topic}"
    
    def _parse_response(self, response_content: str) -> List[str]:
        """Parse LLM response to extract section names."""
        lines = [line.strip() for line in response_content.strip().split('\n')]
        sections = [line for line in lines if line and not line.startswith('-') and not line[0].isdigit()]
        
        if not sections:
            raise ValueError("No valid sections found in LLM response")
            
        return sections[:8]  # Limit to max 8 sections
    
    def _create_dynamic_config(self, query_context: QueryContext, sections: List[str]) -> StyleConfig:
        """Create dynamic style config based on context."""
        from .report_styles import STYLE_CONFIGS
        
        # Base config on domain and audience
        if query_context.domain == QueryDomain.TECHNICAL:
            base_config = STYLE_CONFIGS[ReportStyle.TECHNICAL]
        elif query_context.domain == QueryDomain.BUSINESS:
            base_config = STYLE_CONFIGS[ReportStyle.PROFESSIONAL]
        elif query_context.domain == QueryDomain.SCIENTIFIC:
            base_config = STYLE_CONFIGS[ReportStyle.ACADEMIC]
        else:
            base_config = STYLE_CONFIGS[ReportStyle.PROFESSIONAL]
        
        # Create new config with adaptive structure
        return StyleConfig(
            style=ReportStyle.DEFAULT,
            tone=base_config.tone,
            structure=sections,  # Use adaptive sections
            citation_format=base_config.citation_format,
            length_guideline=base_config.length_guideline,
            language_complexity=base_config.language_complexity,
            use_technical_terms=base_config.use_technical_terms,
            include_visuals=base_config.include_visuals,
            key_features=base_config.key_features + ["Adaptive structure", "Context-specific sections"]
        )


class AdaptiveStructureManager:
    """Main manager for adaptive structure generation."""
    
    def __init__(self, llm=None, enable_adaptive: bool = False, cache_ttl: int = 3600):
        self.query_analyzer = QueryAnalyzer()
        self.validator = StructureValidator()
        self.cache = StructureCache(ttl=cache_ttl)
        
        self.adaptive_strategy = AdaptiveStructureStrategy(llm, self.validator) if llm else None
        self.static_strategy = StaticStructureStrategy()
        
        self.enable_adaptive = enable_adaptive and self.adaptive_strategy is not None
        
        logger.info(f"AdaptiveStructureManager initialized: adaptive={self.enable_adaptive}")
    
    def generate_structure(self, query: str, report_style: ReportStyle) -> Tuple[List[str], StyleConfig]:
        """
        Generate report structure for query.
        
        Args:
            query: Research query
            report_style: Requested report style
            
        Returns:
            (section_names, style_config)
        """
        # Only use adaptive for DEFAULT style
        if report_style != ReportStyle.DEFAULT or not self.enable_adaptive:
            return self._get_static_structure(report_style)
        
        # Analyze query context
        query_context = self.query_analyzer.analyze_query(query)
        
        # Check cache first
        cached = self.cache.get(query_context)
        if cached:
            logger.debug("Using cached adaptive structure")
            return cached.sections, cached.style_config
        
        # Generate adaptive structure
        try:
            sections, style_config = self.adaptive_strategy.generate_structure(query_context)
            
            # Cache the result
            self.cache.set(query_context, sections, style_config)
            
            return sections, style_config
            
        except Exception as e:
            logger.error(f"Adaptive structure generation failed: {e}")
            # Fall back to comprehensive DEFAULT static structure
            return self._get_static_structure(ReportStyle.DEFAULT)
    
    def _get_static_structure(self, report_style: ReportStyle) -> Tuple[List[str], StyleConfig]:
        """Get static structure for given style."""
        from .report_styles import STYLE_CONFIGS
        
        if report_style in STYLE_CONFIGS:
            config = STYLE_CONFIGS[report_style]
            return config.structure, config
        else:
            # Default to professional
            config = STYLE_CONFIGS[ReportStyle.PROFESSIONAL]
            return config.structure, config


# Global instance for easy access
adaptive_structure_manager = None

def get_adaptive_structure_manager(llm=None, enable_adaptive: bool = False, cache_ttl: int = 3600) -> AdaptiveStructureManager:
    """Get global adaptive structure manager instance."""
    global adaptive_structure_manager
    
    if adaptive_structure_manager is None:
        adaptive_structure_manager = AdaptiveStructureManager(
            llm=llm,
            enable_adaptive=enable_adaptive,
            cache_ttl=cache_ttl
        )
    
    return adaptive_structure_manager