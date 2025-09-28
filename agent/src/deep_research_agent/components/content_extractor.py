"""
Content extraction utilities for the research agent.

This module provides utilities for extracting and processing content
from various sources and formats including search results, LLM responses,
and structured data.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

from ..core import (
    get_logger,
    ContentExtractionError,
    SearchResult,
    Citation,
    SearchResultType,
    truncate_text,
    validate_url,
    extract_domain,
    parse_search_keywords
)

logger = get_logger(__name__)


class ContentExtractor:
    """Extracts and processes content from various sources."""
    
    def __init__(self):
        """Initialize content extractor."""
        pass
    
    def extract_search_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        source_type: SearchResultType = SearchResultType.WEB
    ) -> List[SearchResult]:
        """
        Extract structured search results from raw API responses.
        
        Args:
            raw_results: Raw search results from API
            source_type: Type of search results (web, vector, etc.)
            
        Returns:
            List of structured SearchResult objects
            
        Raises:
            ContentExtractionError: If extraction fails
        """
        try:
            extracted_results = []
            
            for raw_result in raw_results:
                try:
                    result = self._extract_single_search_result(raw_result, source_type)
                    if result:
                        extracted_results.append(result)
                except Exception as e:
                    logger.warning(
                        "Failed to extract single search result", 
                        error=e, 
                        raw_result=str(raw_result)[:200]
                    )
                    continue
            
            logger.info(
                f"Extracted {len(extracted_results)} search results",
                source_type=source_type.value,
                total_extracted=len(extracted_results),
                total_raw=len(raw_results)
            )
            
            return extracted_results
        except Exception as e:
            logger.error("Failed to extract search results", error=e, source_type=source_type.value)
            raise ContentExtractionError(f"Failed to extract search results: {e}")
    
    def _extract_single_search_result(
        self, 
        raw_result: Dict[str, Any], 
        source_type: SearchResultType
    ) -> Optional[SearchResult]:
        """
        Extract a single search result from raw data.
        
        Args:
            raw_result: Single raw search result
            source_type: Type of search result
            
        Returns:
            SearchResult object or None if extraction fails
        """
        try:
            # Common field mappings
            content = self._extract_content_field(raw_result)
            title = self._extract_title_field(raw_result)
            url = self._extract_url_field(raw_result)
            source = self._extract_source_field(raw_result, url)
            score = self._extract_score_field(raw_result)
            published_date = self._extract_date_field(raw_result)
            
            # Basic validation
            if not content and not title:
                logger.debug("Skipping result with no content or title", raw_result=raw_result)
                return None
            
            # Create metadata
            metadata = {
                "original_result": raw_result,
                "extraction_timestamp": "",
                "content_length": len(content) if content else 0,
                "has_url": bool(url),
                "domain": extract_domain(url) if url else None
            }
            
            return SearchResult(
                content=content or "",
                source=source,
                url=url,
                title=title,
                score=score,
                published_date=published_date,
                result_type=source_type,
                metadata=metadata
            )
        except Exception as e:
            logger.debug("Failed to extract single search result", error=e)
            return None
    
    def _extract_content_field(self, result: Dict[str, Any]) -> str:
        """Extract content field from result."""
        content_fields = ['content', 'text', 'snippet', 'body', 'description', 'summary']
        
        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).strip()
                if content:
                    return content
        
        return ""
    
    def _extract_title_field(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract title field from result."""
        title_fields = ['title', 'name', 'heading', 'subject']
        
        for field in title_fields:
            if field in result and result[field]:
                title = str(result[field]).strip()
                if title:
                    return title
        
        return None
    
    def _extract_url_field(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract URL field from result."""
        url_fields = ['url', 'link', 'uri', 'href']
        
        for field in url_fields:
            if field in result and result[field]:
                url = str(result[field]).strip()
                if validate_url(url):
                    return url
        
        return None
    
    def _extract_source_field(self, result: Dict[str, Any], url: Optional[str] = None) -> str:
        """Extract source field from result."""
        source_fields = ['source', 'site', 'domain', 'publisher', 'author']
        
        for field in source_fields:
            if field in result and result[field]:
                source = str(result[field]).strip()
                if source:
                    return source
        
        # Fallback to domain from URL
        if url:
            domain = extract_domain(url)
            if domain:
                return domain
        
        return "unknown"
    
    def _extract_score_field(self, result: Dict[str, Any]) -> float:
        """Extract relevance score from result."""
        score_fields = ['score', 'relevance', 'rank', 'confidence']
        
        for field in score_fields:
            if field in result and result[field] is not None:
                try:
                    return float(result[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _extract_date_field(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract publication date from result."""
        date_fields = ['published_date', 'date', 'timestamp', 'published', 'created_at']
        
        for field in date_fields:
            if field in result and result[field]:
                date_str = str(result[field]).strip()
                if date_str:
                    return date_str
        
        return None
    
    def extract_citations(self, search_results: List[SearchResult]) -> List[Citation]:
        """
        Extract citations from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of Citation objects
        """
        try:
            citations = []
            seen_urls = set()
            
            for result in search_results:
                # Skip duplicates based on URL
                if result.url and result.url in seen_urls:
                    continue
                
                if result.url:
                    seen_urls.add(result.url)
                
                citation = Citation(
                    source=result.source,
                    url=result.url,
                    title=result.title,
                    snippet=truncate_text(result.content, max_length=150) if result.content else None
                )
                
                citations.append(citation)
            
            logger.info(f"Extracted {len(citations)} citations from {len(search_results)} results")
            return citations
        except Exception as e:
            logger.error("Failed to extract citations", error=e)
            raise ContentExtractionError(f"Failed to extract citations: {e}")
    
    def clean_content(self, content: str) -> str:
        """
        Clean and normalize content text.
        
        Args:
            content: Raw content text
            
        Returns:
            Cleaned content text
        """
        try:
            if not content:
                return ""
            
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove common unwanted patterns
            patterns_to_remove = [
                r'\[Advertisement\]',
                r'\[Sponsored\]',
                r'Click here.*?(?=\.|$)',
                r'Read more.*?(?=\.|$)',
                r'Subscribe.*?(?=\.|$)',
                r'Sign up.*?(?=\.|$)',
            ]
            
            for pattern in patterns_to_remove:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Remove HTML tags if any
            content = re.sub(r'<[^>]+>', '', content)
            
            # Clean up punctuation
            content = re.sub(r'([.!?])\1+', r'\1', content)  # Remove repeated punctuation
            
            # Strip and normalize
            content = content.strip()
            
            return content
        except Exception as e:
            logger.warning("Failed to clean content", error=e)
            return content  # Return original if cleaning fails
    
    def extract_key_information(
        self, 
        content: str, 
        query: str, 
        max_sentences: int = 3
    ) -> str:
        """
        Extract key information from content relevant to query.
        
        Args:
            content: Full content text
            query: Original search query
            max_sentences: Maximum sentences to extract
            
        Returns:
            Extracted key information
        """
        try:
            if not content:
                return ""
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return content
            
            # Score sentences based on query relevance
            query_keywords = parse_search_keywords(query.lower())
            scored_sentences = []
            
            for sentence in sentences:
                score = self._score_sentence_relevance(sentence.lower(), query_keywords)
                if score > 0:
                    scored_sentences.append((sentence, score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in scored_sentences[:max_sentences]]
            
            if top_sentences:
                result = '. '.join(top_sentences)
                if not result.endswith('.'):
                    result += '.'
                return result
            else:
                # Fallback to first few sentences
                return '. '.join(sentences[:max_sentences]) + '.'
        except Exception as e:
            logger.warning("Failed to extract key information", error=e)
            return truncate_text(content, max_length=300)
    
    def _score_sentence_relevance(self, sentence: str, query_keywords: List[str]) -> float:
        """
        Score sentence relevance to query keywords.
        
        Args:
            sentence: Sentence text (lowercase)
            query_keywords: List of query keywords
            
        Returns:
            Relevance score
        """
        if not query_keywords:
            return 1.0  # Default score if no keywords
        
        score = 0.0
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        for keyword in query_keywords:
            if keyword in sentence_words:
                score += 1.0
            elif any(keyword in word for word in sentence_words):
                score += 0.5  # Partial match
        
        # Normalize by number of keywords
        return score / len(query_keywords)
    
    def summarize_results(self, search_results: List[SearchResult], max_length: int = 500) -> str:
        """
        Create a summary of search results.
        
        Args:
            search_results: List of search results
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        try:
            if not search_results:
                return "No search results available."
            
            summaries = []
            total_length = 0
            
            for result in search_results:
                if total_length >= max_length:
                    break
                
                # Create short summary for each result
                content = result.content or ""
                if len(content) > 100:
                    content = truncate_text(content, max_length=100)
                
                source_info = result.source
                if result.title:
                    source_info = f"{result.title} ({result.source})"
                
                summary = f"[{source_info}]: {content}"
                
                if total_length + len(summary) <= max_length:
                    summaries.append(summary)
                    total_length += len(summary)
                else:
                    # Add partial summary to reach max_length
                    remaining_length = max_length - total_length
                    if remaining_length > 20:  # Only add if meaningful length remains
                        partial_summary = summary[:remaining_length - 3] + "..."
                        summaries.append(partial_summary)
                    break
            
            return "\n\n".join(summaries)
        except Exception as e:
            logger.error("Failed to summarize results", error=e)
            return "Failed to generate summary."


# Create singleton instance
content_extractor = ContentExtractor()