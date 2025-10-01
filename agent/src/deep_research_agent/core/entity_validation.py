"""
Entity Validation Module

Provides utilities for extracting, validating, and tracking entities throughout
the research pipeline to ensure content only mentions requested entities.
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntityValidationMode(Enum):
    """Validation strictness levels"""
    LENIENT = "lenient"      # Allow some entity drift
    MODERATE = "moderate"    # Standard validation
    STRICT = "strict"        # Reject any mention of wrong entities


@dataclass
class EntityViolation:
    """Record of an entity validation violation"""
    content: str
    mentioned_entities: Set[str]
    forbidden_entities: Set[str]
    violation_type: str
    timestamp: str
    context: str = ""


@dataclass
class EntityValidationResult:
    """Result of entity validation"""
    is_valid: bool
    mentioned_entities: Set[str]
    violations: Set[str]
    coverage_score: float  # 0-1, how many requested entities are covered
    confidence: float      # 0-1, confidence in validation result


class EntityExtractor:
    """Extracts entities (countries, organizations, etc.) from text"""
    
    def __init__(self):
        # Build comprehensive entity alias mapping
        self.entity_aliases = self._build_entity_aliases()
        self.country_patterns = self._build_country_patterns()
        
    def _build_entity_aliases(self) -> Dict[str, Set[str]]:
        """Build mapping of canonical entity names to their aliases"""
        return {
            "Spain": {"Spain", "Spanish", "ES", "ESP"},
            "France": {"France", "French", "FR", "FRA"},
            "United Kingdom": {"United Kingdom", "UK", "Britain", "British", "England", "Great Britain", "GB", "GBR"},
            "Switzerland": {"Switzerland", "Swiss", "CH", "CHE", "Zug"},
            "Germany": {"Germany", "German", "DE", "DEU", "Deutschland"},
            "Poland": {"Poland", "Polish", "PL", "POL"},
            "Bulgaria": {"Bulgaria", "Bulgarian", "BG", "BGR"},
            
            # Common alternatives that should NOT be allowed
            "Denmark": {"Denmark", "Danish", "DK", "DNK"},
            "Finland": {"Finland", "Finnish", "FI", "FIN"},
            "Sweden": {"Sweden", "Swedish", "SE", "SWE"},
            "Netherlands": {"Netherlands", "Dutch", "Holland", "NL", "NLD"},
            "Norway": {"Norway", "Norwegian", "NO", "NOR"},
            "Austria": {"Austria", "Austrian", "AT", "AUT"},
            "Italy": {"Italy", "Italian", "IT", "ITA"},
        }
    
    def _build_country_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for detecting countries"""
        patterns = []
        
        # Create patterns for all known entities
        for canonical, aliases in self.entity_aliases.items():
            for alias in aliases:
                # Word boundary patterns to avoid false matches
                pattern = rf'\b{re.escape(alias)}\b'
                patterns.append(re.compile(pattern, re.IGNORECASE))
                
        return patterns
    
    def extract_entities(self, text: str) -> Set[str]:
        """Extract all mentioned entities from text, returning canonical names"""
        if not text:
            return set()
        
        # CRITICAL FIX: Ensure text is always a string for regex operations
        if not isinstance(text, str):
            # If it's a list, join the elements
            if isinstance(text, list):
                text = " ".join(str(item) for item in text if item)
            else:
                text = str(text)
        
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return set()
            
        mentioned = set()
        
        for canonical, aliases in self.entity_aliases.items():
            for alias in aliases:
                pattern = rf'\b{re.escape(alias)}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    mentioned.add(canonical)
                    break  # Found one alias, no need to check others for this entity
                    
        return mentioned
    
    def normalize_entity_list(self, entities: List[str]) -> Set[str]:
        """Convert entity list to canonical names"""
        normalized = set()
        
        for entity in entities:
            # Ensure entity is a string
            if not isinstance(entity, str):
                entity = str(entity)
            
            # Skip empty entities
            if not entity or not entity.strip():
                continue
                
            # Find canonical name for this entity
            canonical = self._find_canonical_name(entity)
            if canonical:
                normalized.add(canonical)
            else:
                # If not found in aliases, use as-is
                normalized.add(entity.strip())
                
        return normalized
    
    def _find_canonical_name(self, entity: str) -> Optional[str]:
        """Find canonical name for an entity"""
        # Ensure entity is a string and is not empty
        if not isinstance(entity, str):
            entity = str(entity)
        
        if not entity:
            return None
            
        entity_clean = entity.strip()
        if not entity_clean:
            return None
        
        for canonical, aliases in self.entity_aliases.items():
            if entity_clean in aliases or entity_clean.lower() in {a.lower() for a in aliases}:
                return canonical
                
        return None


class EntityValidator:
    """Validates content against allowed entities"""
    
    def __init__(self, requested_entities: List[str], mode: EntityValidationMode = EntityValidationMode.MODERATE):
        self.extractor = EntityExtractor()
        self.requested_entities = self.extractor.normalize_entity_list(requested_entities)
        self.mode = mode
        self.violations: List[EntityViolation] = []
        
        logger.info(f"EntityValidator initialized with entities: {self.requested_entities}, mode: {mode}")
    
    def validate_content(self, content: str, context: str = "") -> EntityValidationResult:
        """Validate that content only mentions allowed entities"""
        if not content:
            return EntityValidationResult(
                is_valid=True,
                mentioned_entities=set(),
                violations=set(),
                coverage_score=0.0,
                confidence=1.0
            )
        
        # Extract all mentioned entities
        mentioned = self.extractor.extract_entities(content)
        
        # Find violations (entities mentioned but not requested)
        violations = mentioned - self.requested_entities
        
        # Calculate coverage (how many requested entities are covered)
        coverage = len(mentioned & self.requested_entities) / max(len(self.requested_entities), 1)
        
        # Determine if valid based on mode
        is_valid = self._determine_validity(mentioned, violations)
        
        # Record violations
        if violations and not is_valid:
            violation = EntityViolation(
                content=content[:200] + "..." if len(content) > 200 else content,
                mentioned_entities=mentioned,
                forbidden_entities=violations,
                violation_type=f"entity_scope_{self.mode.value}",
                timestamp=str(logger),
                context=context
            )
            self.violations.append(violation)
            logger.warning(f"Entity violation: mentioned {violations}, allowed {self.requested_entities}")
        
        return EntityValidationResult(
            is_valid=is_valid,
            mentioned_entities=mentioned,
            violations=violations,
            coverage_score=coverage,
            confidence=0.9 if mentioned else 0.5  # Lower confidence for empty content
        )
    
    def _determine_validity(self, mentioned: Set[str], violations: Set[str]) -> bool:
        """Determine if content is valid based on validation mode"""
        if self.mode == EntityValidationMode.STRICT:
            return len(violations) == 0
        elif self.mode == EntityValidationMode.MODERATE:
            # Allow up to 1 violation if we have good coverage
            return len(violations) <= 1
        else:  # LENIENT
            # Allow up to 2 violations
            return len(violations) <= 2
    
    def filter_observations(self, observations: List[Union[Dict[str, Any], 'StructuredObservation']]) -> List[Union[Dict[str, Any], 'StructuredObservation']]:
        """Filter observations to only include those with valid entities"""
        from .observation_models import StructuredObservation
        filtered = []
        
        for obs in observations:
            # Handle both StructuredObservation and dict types
            if isinstance(obs, StructuredObservation):
                content = obs.content or ""
                validation = self.validate_content(content, context="observation_filtering")
                
                if validation.is_valid:
                    # For StructuredObservation, we can't modify directly (immutable)
                    # Create a new observation with entity metadata
                    # Ensure entity_tags is a list before concatenation
                    existing_tags = obs.entity_tags if isinstance(obs.entity_tags, list) else []
                    new_tags = list(validation.mentioned_entities)
                    combined_tags = existing_tags + new_tags
                    
                    new_obs = StructuredObservation(
                        content=obs.content,
                        entity_tags=combined_tags,
                        metric_values={
                            **obs.metric_values,
                            'entity_validated': True,
                            'entity_coverage': validation.coverage_score
                        },
                        confidence=obs.confidence,
                        source_id=obs.source_id,
                        extraction_method=obs.extraction_method,
                        step_id=obs.step_id  # CRITICAL: Preserve step_id for section-specific filtering
                    )
                    filtered.append(new_obs)
                else:
                    logger.debug(f"Filtered observation mentioning forbidden entities: {validation.violations}")
                    
            elif isinstance(obs, dict):
                content = obs.get('content', '') or obs.get('text', '')
                validation = self.validate_content(content, context="observation_filtering")
                
                if validation.is_valid:
                    # For dict observations, add entity metadata as before
                    obs_copy = obs.copy()  # Don't modify original
                    obs_copy['entity_validated'] = True
                    obs_copy['mentioned_entities'] = list(validation.mentioned_entities)
                    obs_copy['entity_coverage'] = validation.coverage_score
                    filtered.append(obs_copy)
                else:
                    logger.debug(f"Filtered observation mentioning forbidden entities: {validation.violations}")
            else:
                # Handle other types by converting to string
                content = str(obs)
                validation = self.validate_content(content, context="observation_filtering")
                
                if validation.is_valid:
                    filtered.append(obs)  # Keep as-is for unknown types
                else:
                    logger.debug(f"Filtered observation mentioning forbidden entities: {validation.violations}")
        
        logger.info(f"Entity filtering: {len(filtered)}/{len(observations)} observations passed")
        return filtered
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Get report on entity coverage and violations"""
        return {
            "requested_entities": list(self.requested_entities),
            "total_violations": len(self.violations),
            "violation_types": list(set(v.violation_type for v in self.violations)),
            "forbidden_entities_seen": list(set().union(*(v.forbidden_entities for v in self.violations))),
            "validation_mode": self.mode.value
        }


def extract_entities_from_query(query: str, llm=None) -> List[str]:
    """Extract entities from user query using flexible LLM-based extraction"""
    if not query:
        return []
    
    # Try LLM-based extraction first (flexible for any entity type)
    if llm:
        entities = _extract_entities_with_llm(query, llm)
        if entities:
            return entities
    
    # Fallback: Use pattern-based extraction for common cases
    extractor = EntityExtractor()
    entities = extractor.extract_entities(query)
    
    # If no entities found, try some heuristics
    if not entities:
        # Look for common patterns like "comparison of X, Y, Z"
        comparison_pattern = r'(?:comparison of|across|between)\s+([^.!?]+)'
        match = re.search(comparison_pattern, query, re.IGNORECASE)
        if match:
            candidate_text = match.group(1)
            entities = extractor.extract_entities(candidate_text)
    
    return sorted(list(entities))


def _extract_entities_with_llm(query: str, llm) -> List[str]:
    """Extract entities using LLM for maximum flexibility"""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are an expert entity extractor. Extract all specific entities (countries, companies, cities, products, people, etc.) mentioned in the user's query.

Rules:
1. Return ONLY entities that are explicitly mentioned in the query
2. Return the canonical/standard form of each entity
3. DO NOT add any entities that aren't directly mentioned
4. Return as a JSON list of strings
5. If no entities found, return an empty list []

Examples:
- "Spain, France, and UK" → ["Spain", "France", "United Kingdom"]
- "compare Apple vs Microsoft" → ["Apple", "Microsoft"]
- "New York vs Los Angeles housing" → ["New York", "Los Angeles"]
- "tax rates in 2024" → []"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract entities from: {query}")
        ]

        # Log the prompt being sent
        logger.info(f"🔍 LLM_PROMPT [entity_extraction]: {messages[1].content[:300]}...")

        response = llm.invoke(messages)

        # CRITICAL FIX: Handle structured responses properly
        # Don't use extract_text_from_response because it might return raw dicts
        # Instead, directly handle the response structure
        result_text = ""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                # Databricks structured format: [{"type": "reasoning", ...}, {"type": "text", ...}]
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        result_text = item.get('text', '')
                        break
            elif isinstance(content, str):
                result_text = content
            else:
                result_text = str(content)
        else:
            result_text = str(response)

        result = result_text.strip()

        # Log the response received
        logger.info(f"🔍 LLM_RESPONSE [entity_extraction]: {result[:300]}...")

        # DEFENSIVE: If result is still a dict/object string representation, return empty
        if result.startswith("{") or result.startswith("{'"):
            logger.warning(f"🚨 ENTITY_EXTRACTION: Got raw dict response, returning empty list")
            return []

        # Parse JSON response
        if result.startswith('[') and result.endswith(']'):
            import json
            entities = json.loads(result)
            # DEFENSIVE: Ensure all entities are strings, not dicts
            parsed_entities = []
            for e in entities:
                if isinstance(e, str) and e.strip():
                    parsed_entities.append(e.strip())
                elif isinstance(e, dict):
                    logger.warning(f"🚨 ENTITY_EXTRACTION: Skipping dict entity: {e}")
                elif e:
                    parsed_entities.append(str(e).strip())

            logger.info(f"🔍 ENTITY_EXTRACTION: Parsed {len(parsed_entities)} entities: {parsed_entities}")
            return parsed_entities

        return []

    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")
        return []


# Global validator instance (will be initialized by planner)
_global_validator: Optional[EntityValidator] = None


def get_global_validator() -> Optional[EntityValidator]:
    """Get the global entity validator instance"""
    return _global_validator


def set_global_validator(validator: EntityValidator) -> None:
    """Set the global entity validator instance"""
    global _global_validator
    _global_validator = validator


def validate_content_global(content: str, context: str = "") -> EntityValidationResult:
    """Validate content using global validator"""
    validator = get_global_validator()
    if validator:
        return validator.validate_content(content, context)
    else:
        # No validator configured, allow everything
        return EntityValidationResult(
            is_valid=True,
            mentioned_entities=set(),
            violations=set(),
            coverage_score=1.0,
            confidence=0.0  # Low confidence because no validation
        )