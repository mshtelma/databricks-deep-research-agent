"""
Snippet analyzer for smart web content fetching decisions.

Analyzes search result snippets to determine:
1. Do we already have enough information in snippets?
2. Which results are worth fetching full content for?
3. What specific data are we missing?
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .types import EnrichedSearchResult, FetchingConfig

logger = logging.getLogger(__name__)


@dataclass
class SnippetAnalysis:
    """Analysis results for a set of search snippets."""
    has_sufficient_data: bool  # Do snippets have enough info already?
    data_density_score: float  # 0-1, higher = more data in snippets
    missing_data_types: List[str]  # What data types are missing
    fetch_recommendations: List[int]  # Indices of results to fetch
    rationale: str  # Explanation of decision


class SnippetAnalyzer:
    """
    Analyzes search snippets to make smart fetching decisions.

    Uses generic pattern detection that adapts to any research topic.
    """

    @staticmethod
    def extract_topic_patterns(research_context: str) -> Dict[str, re.Pattern]:
        """
        Dynamically generate patterns based on research context.

        Instead of hardcoded patterns, this extracts key terms from
        the query and creates patterns for ANY research topic.

        Args:
            research_context: Research question/context

        Returns:
            Dict of pattern_name -> compiled regex
        """
        patterns = {}

        # Universal data indicators (work for ANY topic)
        patterns['numeric_data'] = re.compile(
            r'\d+\.?\d*\s*[%$€£¥₹]?|\d{1,3}(,\d{3})+',
            re.I
        )
        patterns['comparisons'] = re.compile(
            r'versus|vs\.|compared|higher|lower|more|less|better|worse',
            re.I
        )
        patterns['structured_data'] = re.compile(
            r'table|chart|graph|list|breakdown|summary|overview',
            re.I
        )
        patterns['examples'] = re.compile(
            r'example|e\.g\.|for instance|such as|including',
            re.I
        )
        patterns['definitions'] = re.compile(
            r'defined as|meaning|refers to|is a|are a',
            re.I
        )
        patterns['temporal'] = re.compile(
            r'20\d{2}|current|latest|updated|recent|historical',
            re.I
        )

        # Extract key terms from research context
        context_words = re.findall(r'\b[a-z]{4,}\b', research_context.lower())

        # Create patterns for context-specific terms (top 5)
        for word in context_words[:5]:
            if len(word) > 4:  # Skip short common words
                # Match word and its variants
                stem = word[:len(word)-2] if len(word) > 6 else word
                patterns[f'context_{word}'] = re.compile(rf'\b{stem}\w*\b', re.I)

        return patterns

    @staticmethod
    def identify_high_value_domains(research_context: str) -> List[str]:
        """
        Dynamically identify authoritative domains based on topic.

        Comprehensive domain detection across 30+ categories.

        Args:
            research_context: Research question/context

        Returns:
            List of regex patterns for authoritative domains
        """
        # Universal authoritative domains
        domains = [
            r'\.gov',
            r'\.edu',
            r'\.org',
            r'wikipedia\.org',
            r'scholar\.google',
            r'britannica\.com',
        ]

        context_lower = research_context.lower()

        # MEDICAL & HEALTH
        if any(term in context_lower for term in ['medical', 'health', 'disease', 'treatment', 'drug', 'pharma', 'clinical']):
            domains.extend([
                'nih.gov', 'cdc.gov', 'who.int', 'pubmed.ncbi',
                'nejm.org', 'thelancet.com', 'bmj.com',
                'mayoclinic.org', 'clevelandclinic.org',
                'johnshopkins.edu', 'webmd.com', 'medscape.com',
            ])

        # LEGAL & REGULATORY
        if any(term in context_lower for term in ['legal', 'law', 'court', 'regulation', 'patent', 'copyright']):
            domains.extend([
                'justia.com', 'findlaw.com', 'law.cornell.edu',
                'supremecourt.gov', 'uscourts.gov',
                'law.harvard.edu', 'law.stanford.edu',
                'regulations.gov', 'uspto.gov', 'copyright.gov',
            ])

        # FINANCE & ECONOMICS
        if any(term in context_lower for term in ['finance', 'investment', 'market', 'stock', 'economy', 'banking', 'crypto']):
            domains.extend([
                'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com',
                'economist.com', 'forbes.com', 'fortune.com',
                'marketwatch.com', 'seekingalpha.com',
                'morningstar.com', 'yahoo.finance',
                'federalreserve.gov', 'imf.org', 'worldbank.org',
            ])

        # TECHNOLOGY & PROGRAMMING
        if any(term in context_lower for term in ['tech', 'software', 'programming', 'code', 'ai', 'machine learning', 'data']):
            domains.extend([
                'github.com', 'stackoverflow.com', 'medium.com',
                'dev.to', 'techcrunch.com', 'wired.com',
                'arstechnica.com', 'theverge.com',
                'ieee.org', 'acm.org', 'developer.mozilla.org',
            ])

        # SCIENCE & RESEARCH
        if any(term in context_lower for term in ['science', 'research', 'study', 'physics', 'chemistry', 'biology']):
            domains.extend([
                'nature.com', 'science.org', 'sciencemag.org',
                'arxiv.org', 'pubmed.ncbi', 'plos.org',
                'cell.com', 'pnas.org', 'springer.com',
            ])

        # STATISTICS & DATA
        if any(term in context_lower for term in ['statistic', 'data', 'survey', 'census', 'demographic', 'metric']):
            domains.extend([
                'statista.com', 'pewresearch.org', 'gallup.com',
                'census.gov', 'bls.gov', 'data.gov',
                'worldometers.info', 'ourworldindata.org',
                'stats.oecd.org', 'ec.europa.eu/eurostat',
            ])

        # MAJOR NEWS OUTLETS
        if any(term in context_lower for term in ['news', 'current', 'event', 'politics', 'world']):
            domains.extend([
                'nytimes.com', 'washingtonpost.com', 'wsj.com',
                'bbc.com', 'theguardian.com', 'reuters.com',
                'apnews.com', 'npr.org', 'pbs.org',
            ])

        # BUSINESS & COMPANIES
        if any(term in context_lower for term in ['business', 'company', 'corporate', 'startup']):
            domains.extend([
                'businessinsider.com', 'forbes.com', 'fortune.com',
                'hbr.org', 'mckinsey.com', 'bcg.com',
                'deloitte.com', 'pwc.com', 'ey.com', 'kpmg.com',
            ])

        # TAX & ACCOUNTING
        if any(term in context_lower for term in ['tax', 'accounting', 'audit', 'fiscal', 'revenue']):
            domains.extend([
                'irs.gov', 'hmrc.gov.uk', 'ato.gov.au',
                'taxfoundation.org', 'pwc.com/tax',
                'ey.com/tax', 'worldwide-tax.com',
            ])

        # ENVIRONMENT & CLIMATE
        if any(term in context_lower for term in ['environment', 'climate', 'sustainability', 'renewable']):
            domains.extend([
                'epa.gov', 'noaa.gov', 'climate.gov',
                'ipcc.ch', 'unep.org', 'carbonbrief.org',
            ])

        # Remove duplicates
        return list(set(domains))

    def __init__(self, research_context: str = ""):
        """
        Initialize analyzer with research context.

        Args:
            research_context: Current research step description
        """
        self.research_context = research_context.lower()
        self.data_patterns = self.extract_topic_patterns(research_context)
        self.authority_domains = self.identify_high_value_domains(research_context)

    def analyze_snippets(
        self,
        search_results: List[EnrichedSearchResult],
        config: FetchingConfig
    ) -> SnippetAnalysis:
        """
        Analyze snippets to determine fetching strategy.

        Args:
            search_results: Search results with snippets
            config: Fetching configuration

        Returns:
            SnippetAnalysis with recommendations
        """

        # Analyze data density across all snippets
        all_snippets_text = " ".join(
            r.content + " " + r.title
            for r in search_results
        )

        data_density = self._calculate_data_density(all_snippets_text)
        missing_data = self._identify_missing_data(all_snippets_text)

        # Determine if we need to fetch
        has_sufficient = (
            data_density >= 0.3 and  # At least 30% data coverage
            len(missing_data) <= 2    # Missing at most 2 data types
        )

        # Score each result for fetch-worthiness
        scored_results = [
            (i, self._calculate_fetch_score(result, i, missing_data))
            for i, result in enumerate(search_results)
        ]

        # ULTRA-SIMPLE ALGORITHM: Sort by score, pick best N with domain diversity
        # Sort ALL results by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Single-pass selection with domain diversity
        fetch_recommendations = []
        seen_domains = {}  # domain -> count

        # Calculate target minimum (never exceed available results)
        target_minimum = min(config.min_pages_to_fetch, len(scored_results))
        logger.info(f"🔍 SNIPPET ANALYZER: min_pages_to_fetch={config.min_pages_to_fetch}, "
                   f"available_results={len(scored_results)}, target_minimum={target_minimum}")

        for idx, score in scored_results:
            # Check if we should stop
            if len(fetch_recommendations) >= config.max_pages_to_fetch:
                break

            # For results beyond minimum, apply score threshold
            if len(fetch_recommendations) >= target_minimum:
                if score < config.fetch_score_threshold:
                    break  # No more high-quality results

            result = search_results[idx]
            domain_count = seen_domains.get(result.domain, 0)

            # Apply domain diversity filter
            # Use relaxed limit (3) for minimum guarantee, strict (2) for expansion
            if len(fetch_recommendations) < target_minimum:
                domain_limit = 3  # Relaxed for minimum
            else:
                domain_limit = config.max_per_domain  # Strict for expansion

            if domain_count >= domain_limit:
                continue

            fetch_recommendations.append(idx)
            seen_domains[result.domain] = domain_count + 1
            search_results[idx].fetch_score = score

        # FORCE minimum fetches if we didn't meet it (domain diversity may have blocked)
        if len(fetch_recommendations) < target_minimum:
            logger.warning(f"Domain diversity prevented reaching minimum ({len(fetch_recommendations)}/{target_minimum}), "
                          f"forcing additional fetches")

            # Add remaining best results to meet minimum, ignoring domain limits
            remaining = [
                (idx, score) for idx, score in scored_results
                if idx not in fetch_recommendations
            ]

            needed = target_minimum - len(fetch_recommendations)
            for idx, score in remaining[:needed]:
                fetch_recommendations.append(idx)
                search_results[idx].fetch_score = score
                logger.debug(f"Force-added result {idx} to meet minimum (score={score:.2f})")

        # DIAGNOSTIC: Verify minimum was met
        logger.info(f"🔍 SNIPPET ANALYZER: Final fetch_recommendations count: {len(fetch_recommendations)}")
        logger.info(f"🔍 SNIPPET ANALYZER: Target minimum: {target_minimum}, "
                   f"Met minimum: {len(fetch_recommendations) >= target_minimum}")

        # Generate rationale
        guaranteed = min(target_minimum, len(fetch_recommendations))
        additional = len(fetch_recommendations) - guaranteed

        if len(fetch_recommendations) == 0:
            rationale = f"No results available (checked {len(scored_results)} results)"
        elif additional == 0:
            rationale = (
                f"Fetching guaranteed minimum {guaranteed} best-scored results "
                f"(density={data_density:.2f})"
            )
        else:
            rationale = (
                f"Fetching {guaranteed} guaranteed + {additional} high-scoring results "
                f"(total {len(fetch_recommendations)})"
            )

        return SnippetAnalysis(
            has_sufficient_data=has_sufficient,
            data_density_score=data_density,
            missing_data_types=missing_data,
            fetch_recommendations=fetch_recommendations,
            rationale=rationale
        )

    def _calculate_data_density(self, text: str) -> float:
        """Calculate how data-rich the text is (0-1)."""
        if not text:
            return 0.0

        total_patterns = len(self.data_patterns)
        matched_patterns = 0
        match_density = 0

        for pattern_name, pattern in self.data_patterns.items():
            matches = pattern.findall(text)
            if matches:
                matched_patterns += 1
                # More matches = higher density
                match_density += min(len(matches) / 10, 1.0)

        # Combine pattern coverage and match density
        pattern_coverage = matched_patterns / total_patterns if total_patterns > 0 else 0
        avg_density = match_density / total_patterns if total_patterns > 0 else 0

        return (pattern_coverage * 0.6) + (avg_density * 0.4)

    def _identify_missing_data(self, text: str) -> List[str]:
        """Identify what data types are missing from snippets."""
        missing = []

        # Check universal patterns
        if 'numeric_data' in self.data_patterns:
            if not self.data_patterns['numeric_data'].search(text):
                missing.append("numeric_data")

        if 'structured_data' in self.data_patterns:
            if not self.data_patterns['structured_data'].search(text):
                missing.append("tables_charts")

        if 'temporal' in self.data_patterns:
            if not self.data_patterns['temporal'].search(text):
                missing.append("current_year_data")

        if 'examples' in self.data_patterns:
            if not self.data_patterns['examples'].search(text):
                missing.append("examples")

        return missing

    def _calculate_fetch_score(
        self,
        result: EnrichedSearchResult,
        position: int,
        missing_data: List[str]
    ) -> float:
        """
        Calculate the value of fetching this specific result.

        Scoring factors (totals 1.0):
        1. Domain authority (0-0.3)
        2. Missing data potential (0-0.3)
        3. Search ranking (0-0.2)
        4. Snippet promises (0-0.2)
        """
        score = 0.0

        # 1. Domain authority (30%)
        if any(domain in result.domain for domain in self.authority_domains):
            score += 0.3

        # 2. Missing data potential (30%)
        content_lower = (result.content + " " + result.title).lower()
        for data_type in missing_data[:3]:
            if data_type in content_lower or data_type.replace('_', ' ') in content_lower:
                score += 0.1

        # 3. Search ranking (20%)
        position_score = max(0, (10 - position) / 50)
        score += position_score

        # 4. Snippet promises more detail (20%)
        promise_phrases = [
            "full breakdown", "complete guide", "detailed",
            "comprehensive", "all rates", "calculator",
            "examples", "see more", "read more"
        ]
        if any(phrase in content_lower for phrase in promise_phrases):
            score += 0.2

        return min(score, 1.0)

    def _snippet_has_complete_data(self, result: EnrichedSearchResult) -> bool:
        """Check if snippet already contains complete data."""
        # If snippet is long and has multiple data patterns, might be complete
        if len(result.content) < 300:
            return False

        pattern_matches = sum(
            1 for pattern in self.data_patterns.values()
            if pattern.search(result.content)
        )

        # If has 5+ different data patterns in snippet, probably complete
        return pattern_matches >= 5
