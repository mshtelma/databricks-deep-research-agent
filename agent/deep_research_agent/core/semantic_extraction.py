"""
Semantic extraction components for intelligent data extraction.

Replaces naive pattern matching with LLM-based semantic understanding
for better entity and data extraction accuracy.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .table_models import TableCell

logger = logging.getLogger(__name__)


class SemanticEntityExtractor:
    """
    Semantic entity extraction using LLM with intelligent fallbacks.
    
    Replaces the naive regex patterns in table_generator.py that capture
    random text fragments as entities.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.cache: Dict[str, List[str]] = {}
        self.max_cache_size = 100
        
    def extract_entities(
        self, 
        research_data: List[Dict[str, Any]], 
        entity_type: str,
        max_entities: int = 10
    ) -> List[str]:
        """
        Extract entities semantically, not with regex patterns.
        
        Args:
            research_data: List of research data items with 'content' field
            entity_type: Type of entities to extract (e.g., "countries", "companies")
            max_entities: Maximum number of entities to return
            
        Returns:
            List of extracted entity names
        """
        
        # Create cache key
        content_hash = hash(str([item.get("content", "")[:100] for item in research_data]))
        cache_key = f"{entity_type}:{content_hash}"
        
        if cache_key in self.cache:
            logger.debug(f"Using cached entities for {entity_type}")
            return self.cache[cache_key][:max_entities]
        
        if not self.llm:
            logger.info(f"No LLM available, using improved fallback for {entity_type}")
            entities = self._fallback_extraction(research_data, entity_type)
        else:
            try:
                entities = self._semantic_extraction(research_data, entity_type)
            except Exception as e:
                logger.warning(f"Semantic extraction failed: {e}, using fallback")
                entities = self._fallback_extraction(research_data, entity_type)
        
        # Cache results (with size limit)
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        entities_limited = entities[:max_entities]
        self.cache[cache_key] = entities_limited
        
        logger.info(f"Extracted {len(entities_limited)} {entity_type}: {entities_limited}")
        return entities_limited
    
    def _semantic_extraction(
        self, 
        research_data: List[Dict[str, Any]], 
        entity_type: str
    ) -> List[str]:
        """Extract entities using LLM semantic understanding."""
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(research_data)
        
        system_prompt = f"""
        You are an expert data analyst. Extract all {entity_type} mentioned in this research data.
        
        Rules:
        - Only include actual {entity_type}, not partial matches or related terms
        - Return entity names exactly as they appear (proper capitalization)
        - Avoid duplicates and variations of the same entity
        - Skip generic terms that aren't specific {entity_type}
        - Maximum 15 entities
        
        Return ONLY a JSON list of entity names, no other text:
        ["Entity1", "Entity2", "Entity3"]
        
        If no clear {entity_type} are found, return: []
        """
        
        user_prompt = f"""
        Extract {entity_type} from this research data:
        
        {data_summary}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse JSON response
        response_text = response.content.strip()
        
        # Handle common response variations
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        entities = json.loads(response_text)
        
        if not isinstance(entities, list):
            logger.warning(f"Expected list, got {type(entities)}")
            return []
            
        # Clean and validate entities
        cleaned_entities = []
        for entity in entities:
            if isinstance(entity, str) and len(entity.strip()) > 1:
                cleaned = entity.strip()
                if cleaned and cleaned not in cleaned_entities:
                    cleaned_entities.append(cleaned)
        
        return cleaned_entities[:15]  # Limit to prevent overwhelming
    
    def _fallback_extraction(
        self, 
        research_data: List[Dict[str, Any]], 
        entity_type: str
    ) -> List[str]:
        """
        Improved fallback extraction using better heuristics.
        
        This is much better than the current naive regex approach
        but still not as good as semantic extraction.
        """
        
        entities = set()
        
        # Define better patterns based on entity type
        patterns = self._get_improved_patterns(entity_type)
        
        for data_item in research_data:
            content = str(data_item.get("content", ""))
            
            # Use improved pattern matching with context validation
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Validate the match with context
                    if self._validate_entity_match(match, content, entity_type):
                        entities.add(match.strip().title())
        
        # Filter out obvious false positives
        filtered_entities = []
        for entity in entities:
            if self._is_valid_entity(entity, entity_type):
                filtered_entities.append(entity)
        
        return sorted(filtered_entities)[:10]  # Limit and sort
    
    def _prepare_data_summary(self, research_data: List[Dict[str, Any]]) -> str:
        """Prepare research data summary for LLM analysis."""
        
        summaries = []
        for i, data_item in enumerate(research_data[:5], 1):  # Limit to first 5 items
            content = str(data_item.get("content", ""))[:500]  # Limit content length
            summaries.append(f"Source {i}: {content}")
        
        return "\n\n".join(summaries)
    
    def _get_improved_patterns(self, entity_type: str) -> List[str]:
        """Get improved regex patterns based on entity type."""
        
        entity_patterns = {
            "countries": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s+(?:is|has|will|was|were|government|economy|tax|income))',
                r'\b(United States|United Kingdom|New Zealand|South Korea|Saudi Arabia|South Africa)\b',
                r'\b([A-Z][a-z]+)\s+(?:government|economy|tax system|residents|citizens)'
            ],
            "companies": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc|LLC|Ltd|Corporation|Corp)\b',
                r'\b([A-Z]+[a-z]*)\s+(?:reported|announced|stated|revenue|profit)'
            ],
            "currencies": [
                r'\b(USD|EUR|GBP|JPY|AUD|CAD|CHF)\b',
                r'\b(US Dollar|Euro|British Pound|Japanese Yen)\b'
            ],
            "default": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
            ]
        }
        
        return entity_patterns.get(entity_type.lower(), entity_patterns["default"])
    
    def _validate_entity_match(self, match: str, context: str, entity_type: str) -> bool:
        """Validate that a regex match is actually a valid entity."""
        
        # Skip very short matches
        if len(match.strip()) < 2:
            return False
            
        # Skip common English words that aren't entities
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'throughout', 'despite'
        }
        
        if match.lower() in stop_words:
            return False
        
        # Additional validation based on entity type
        if entity_type.lower() == "countries":
            # Countries should have geographical or political context nearby
            context_words = ['tax', 'government', 'economy', 'residents', 'citizens', 'policy']
            context_lower = context.lower()
            match_pos = context_lower.find(match.lower())
            
            if match_pos == -1:
                return False
                
            # Check surrounding context (50 chars before and after)
            surrounding = context_lower[max(0, match_pos-50):match_pos+len(match)+50]
            return any(word in surrounding for word in context_words)
        
        return True
    
    def _is_valid_entity(self, entity: str, entity_type: str) -> bool:
        """Final validation of extracted entity."""
        
        # Skip entities that are too generic
        generic_terms = {
            'Data', 'Information', 'Research', 'Study', 'Analysis', 'Report',
            'Content', 'Text', 'Document', 'Source', 'Result', 'Finding'
        }
        
        if entity in generic_terms:
            return False
            
        # Entity should have reasonable length
        if len(entity) < 2 or len(entity) > 50:
            return False
            
        return True


class StructuredDataMatcher:
    """
    Structured data matching using LLM semantic understanding.
    
    Replaces the fuzzy string matching in _find_data_value that causes
    the N/A epidemic by matching unrelated text.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.cache: Dict[str, TableCell] = {}
        self.max_cache_size = 200
    
    def find_data_value(
        self, 
        research_data: List[Dict[str, Any]], 
        entity: str, 
        metric: str
    ) -> TableCell:
        """
        Find actual data value using structured matching.
        
        Args:
            research_data: List of research data items
            entity: Entity name to find data for
            metric: Metric/attribute to extract
            
        Returns:
            TableCell with extracted value or clear "not available" status
        """
        
        # Create cache key
        cache_key = f"{entity}:{metric}:{len(research_data)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.llm:
            result = self._fallback_matching(research_data, entity, metric)
        else:
            try:
                result = self._semantic_matching(research_data, entity, metric)
            except Exception as e:
                logger.warning(f"Semantic matching failed for {entity}-{metric}: {e}")
                result = self._fallback_matching(research_data, entity, metric)
        
        # Cache result
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def _semantic_matching(
        self, 
        research_data: List[Dict[str, Any]], 
        entity: str, 
        metric: str
    ) -> TableCell:
        """Use LLM to find specific data value."""
        
        # Prepare focused data for analysis
        relevant_data = self._find_relevant_data(research_data, entity, metric)
        
        if not relevant_data:
            return TableCell(
                value=None,
                formatted_value="Data not available",
                confidence=0.0,
                source="No relevant data found"
            )
        
        system_prompt = f"""
        You are a data extraction specialist. Find the specific value for "{metric}" for "{entity}".
        
        Rules:
        - Only return data that explicitly mentions both {entity} AND {metric}
        - Return the exact numeric value if found
        - If no clear match exists, return null
        - Don't guess or estimate
        - Don't use data from similar but different entities
        
        Return JSON:
        {{
            "value": "exact value or null if not found",
            "confidence": 0.0-1.0,
            "source_text": "relevant excerpt from data",
            "reasoning": "brief explanation of match/no-match"
        }}
        """
        
        user_prompt = f"""
        Find {metric} for {entity} in this data:
        
        {relevant_data}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse response
        try:
            result = json.loads(response.content.strip())
            
            value = result.get("value")
            confidence = result.get("confidence", 0.0)
            source_text = result.get("source_text", "")
            reasoning = result.get("reasoning", "")
            
            # Convert value if needed
            if value is not None and value != "null":
                formatted_value = self._format_value(value, metric)
                return TableCell(
                    value=value,
                    formatted_value=formatted_value,
                    confidence=confidence,
                    source=source_text[:100],  # Limit source text length
                    citation_id=None,
                    is_estimated=confidence < 0.8
                )
            else:
                return TableCell(
                    value=None,
                    formatted_value="Data not available",
                    confidence=0.0,
                    source=f"No clear match found: {reasoning}"
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_matching(research_data, entity, metric)
    
    def _fallback_matching(
        self, 
        research_data: List[Dict[str, Any]], 
        entity: str, 
        metric: str
    ) -> TableCell:
        """
        Improved fallback matching with better validation.
        
        This is much more conservative than the current fuzzy matching
        that creates false positives.
        """
        
        for data_item in research_data:
            content = str(data_item.get("content", ""))
            
            # Both entity and metric must be present
            if entity.lower() not in content.lower() or metric.lower() not in content.lower():
                continue
            
            # Look for numeric patterns near both entity and metric
            entity_pos = content.lower().find(entity.lower())
            metric_pos = content.lower().find(metric.lower())
            
            # They should be reasonably close (within 200 characters)
            if abs(entity_pos - metric_pos) > 200:
                continue
            
            # Extract surrounding context
            start_pos = min(entity_pos, metric_pos) - 50
            end_pos = max(entity_pos, metric_pos) + max(len(entity), len(metric)) + 50
            context = content[max(0, start_pos):end_pos]
            
            # Look for numbers in the context
            numbers = re.findall(r'[\d,]+\.?\d*', context)
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', context)
            
            if numbers or percentages:
                # Use the first reasonable number found
                value = percentages[0] if percentages else numbers[0]
                formatted_value = self._format_value(value, metric)
                
                return TableCell(
                    value=value,
                    formatted_value=formatted_value,
                    confidence=0.6,  # Lower confidence for fallback
                    source=context[:100],
                    is_estimated=True
                )
        
        # No match found
        return TableCell(
            value=None,
            formatted_value="Data not available",
            confidence=0.0,
            source="No matching data found in sources"
        )
    
    def _find_relevant_data(
        self, 
        research_data: List[Dict[str, Any]], 
        entity: str, 
        metric: str
    ) -> str:
        """Find data items that mention both entity and metric."""
        
        relevant_items = []
        
        for data_item in research_data:
            content = str(data_item.get("content", ""))
            
            # Check if both entity and metric are mentioned
            if entity.lower() in content.lower() and metric.lower() in content.lower():
                # Extract relevant paragraph(s)
                sentences = content.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    if entity.lower() in sentence.lower() or metric.lower() in sentence.lower():
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    relevant_items.append('. '.join(relevant_sentences))
        
        return '\n\n'.join(relevant_items[:3])  # Limit to 3 most relevant items
    
    def _format_value(self, value: Any, metric: str) -> str:
        """Format value based on metric type."""
        if value is None:
            return "Data not available"
        
        metric_lower = metric.lower()
        
        try:
            # Handle percentage values
            if "%" in str(value) or "rate" in metric_lower or "percent" in metric_lower:
                numeric_value = float(re.search(r'(\d+(?:\.\d+)?)', str(value)).group(1))
                return f"{numeric_value:.1f}%"
            
            # Handle currency values
            if any(word in metric_lower for word in ["income", "salary", "cost", "price", "rent", "tax"]):
                # Remove commas and convert to float
                numeric_value = float(str(value).replace(',', ''))
                return f"${numeric_value:,.0f}"
            
            # Handle general numeric values
            if isinstance(value, (int, float)):
                return f"{value:,.2f}"
            elif str(value).replace(',', '').replace('.', '').isdigit():
                numeric_value = float(str(value).replace(',', ''))
                return f"{numeric_value:,.0f}"
            
        except (ValueError, AttributeError):
            pass
        
        return str(value)
