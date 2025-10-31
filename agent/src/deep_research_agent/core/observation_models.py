"""
Structured observation models for enhanced data extraction and table population.

These models provide a structured way to store and access research observations
with entity/metric metadata for better table generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Iterable
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class ExtractionMethod(str, Enum):
    """Method used to extract the observation data."""
    LLM = "llm"
    PATTERN = "pattern"
    EXPLICIT = "explicit"
    SEARCH = "search"


class StructuredObservation(BaseModel):
    """
    Enhanced observation with structured metadata for table population.

    This replaces simple string observations with structured data that can
    be easily extracted for table generation.

    Smart Content Management:
    - For content > 1000 chars: Stores LLM summary in 'content' and full text in 'full_content'
    - For content <= 1000 chars: Stores original in 'content', full_content is None
    - All agents use content (summary) for prompts, preserving tokens
    - Fact checker can access full_content for detailed verification
    """
    content: str
    entity_tags: List[str] = Field(default_factory=list)
    metric_values: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source_id: Optional[str] = None
    extraction_method: ExtractionMethod = ExtractionMethod.LLM
    step_id: Optional[str] = None  # CRITICAL: Enables filtering observations by step for section-specific content
    section_title: Optional[str] = None  # Optional: Makes observations self-describing for debugging/display

    # Smart summarization fields
    full_content: Optional[str] = None  # Original full text if content is summarized
    is_summarized: bool = False  # True if content is LLM-generated summary
    original_length: int = 0  # Character count of original content before summarization
    
    # Calculation feedback tracking
    feedback_source: Optional[str] = None  # Identifies if observation comes from calculation feedback (e.g., "calculation_feedback")

    @validator('content', pre=True)
    def validate_content(cls, v):
        """
        Validator to handle list-type content from reasoning models.

        Reasoning models return: [{'type': 'reasoning'...}, {'type': 'text'...}]
        This validator extracts the actual text from such responses.
        """
        if isinstance(v, list):
            # Handle reasoning model response format
            # Import here to avoid circular dependency
            try:
                from deep_research_agent.core.response_handlers import DatabricksResponseParser
                # Create a mock message object with the list content
                class MockMessage:
                    def __init__(self, content):
                        self.content = content

                mock_msg = MockMessage(v)
                parsed = DatabricksResponseParser().parse(mock_msg)
                return parsed.content
            except Exception as e:
                # Fallback: convert list to string if parsing fails
                import logging
                logging.warning(f"Failed to parse list content with DatabricksResponseParser: {e}. Using str() fallback.")
                return str(v)
        elif v is None:
            return ""
        elif isinstance(v, str):
            return v
        else:
            return str(v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "entity_tags": self.entity_tags,
            "metric_values": self.metric_values,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "extraction_method": self.extraction_method.value,
            "step_id": self.step_id,
            "section_title": self.section_title,
            "full_content": self.full_content,
            "is_summarized": self.is_summarized,
            "original_length": self.original_length,
            "feedback_source": self.feedback_source,
        }

    def __str__(self) -> str:
        """Backward compatibility with string observations."""
        if self.content is None:
            return ""
        if not isinstance(self.content, str):
            return str(self.content)
        return self.content

    def get_metric_value(self, metric: str, default: Any = None) -> Any:
        """Get a metric value with fallback."""
        return self.metric_values.get(metric, default)

    def has_entity(self, entity: str) -> bool:
        """Check if observation contains a specific entity."""
        entity_lower = entity.lower()
        return any(tag.lower() == entity_lower for tag in self.entity_tags)

    def has_metric(self, metric: str) -> bool:
        """Check if observation contains a specific metric."""
        metric_lower = metric.lower()
        return any(key.lower() == metric_lower for key in self.metric_values.keys())

    def get_content_for_prompt(self) -> str:
        """
        Get content optimized for LLM prompts.

        Returns:
            Summarized content if available, otherwise full content
        """
        return self.content

    def get_full_content(self) -> str:
        """
        Get full original content for fact-checking/verification.

        Returns:
            Full content if summarized, otherwise regular content
        """
        return self.full_content or self.content

    def get_stats(self) -> Dict[str, Any]:
        """
        Get content statistics for debugging/monitoring.

        Returns:
            Dict with length info, summarization status, compression ratio
        """
        return {
            "display_length": len(self.content),
            "full_length": len(self.full_content) if self.full_content else len(self.content),
            "is_summarized": self.is_summarized,
            "compression_ratio": round(len(self.content) / self.original_length, 2) if self.original_length > 0 else 1.0
        }

    @classmethod
    def from_string(
        cls,
        content: str,
        confidence: float = 0.5,
        step_id: Optional[str] = None,
        section_title: Optional[str] = None
    ) -> 'StructuredObservation':
        """
        Create from legacy string observation with basic parsing.

        Args:
            content: Observation text content
            confidence: Confidence score (0.0-1.0)
            step_id: Optional step identifier for section filtering
            section_title: Optional section title for debugging/display
        """
        return cls(
            content=content,
            confidence=confidence,
            extraction_method=ExtractionMethod.PATTERN,
            step_id=step_id,
            section_title=section_title
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredObservation':
        """
        Reconstruct StructuredObservation from dictionary.

        This is the critical deserialization method for converting state dicts
        back to StructuredObservation objects at agent entry points.

        Args:
            data: Dictionary with observation fields (from to_dict() or state)

        Returns:
            StructuredObservation instance

        Note:
            Handles both complete dicts and partial dicts with defaults.
            Gracefully handles string extraction_method for enum conversion.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Handle extraction_method as either string or enum
        extraction_method = data.get("extraction_method", "llm")
        if isinstance(extraction_method, str):
            try:
                extraction_method = ExtractionMethod(extraction_method.lower())
            except ValueError:
                # Default to LLM if invalid value
                extraction_method = ExtractionMethod.LLM

        # Extract and clean content field to prevent dict representations
        raw_content = data.get("content", "")

        # Handle various content types
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, dict):
            # Content is nested dict - extract text
            # This handles cases where content itself is an observation dict
            if 'content' in raw_content:
                # Recursive dict - extract inner content
                content = raw_content.get('content', '')
            else:
                # Try to get text field, otherwise empty
                content = raw_content.get('text', '')

            # Ensure we got a string
            if not isinstance(content, str):
                content = ''
                logger.warning(f"Dict content with non-string inner content: {raw_content}")
        elif isinstance(raw_content, list):
            # Join list items
            content = " ".join(str(item) for item in raw_content)
        else:
            # Convert to string as last resort
            content = str(raw_content) if raw_content else ""

        return cls(
            content=content,
            entity_tags=data.get("entity_tags", []),
            metric_values=data.get("metric_values", {}),
            confidence=data.get("confidence", 1.0),
            source_id=data.get("source_id"),
            extraction_method=extraction_method,
            step_id=data.get("step_id"),
            section_title=data.get("section_title"),
            full_content=data.get("full_content"),  # Preserve summarization
            is_summarized=data.get("is_summarized", False),
            original_length=data.get("original_length", 0),
            feedback_source=data.get("feedback_source")
        )


def ensure_structured_observation(obs: Union[str, StructuredObservation, Dict]) -> StructuredObservation:
    """
    Convert string observations to structured observations for backward compatibility.

    Args:
        obs: Either a string, StructuredObservation, or dict

    Returns:
        StructuredObservation instance
    """
    if isinstance(obs, str):
        return StructuredObservation.from_string(obs)
    elif isinstance(obs, StructuredObservation):
        # ✅ FIX: Return defensive copy to prevent mutation (Pydantic v1 uses .copy())
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🔍 [ensure_structured_observation] Creating defensive COPY (original id={id(obs)}, step_id={getattr(obs, 'step_id', 'NONE')})")
        return obs.copy()
    elif isinstance(obs, dict):
        # Handle dict observations properly - use from_dict not str()!
        return StructuredObservation.from_dict(obs)
    else:
        # Last resort for unknown types
        return StructuredObservation.from_string(str(obs))


def observations_to_research_data(observations: List[Union[str, StructuredObservation]]) -> List[Dict[str, Any]]:
    """
    Convert observations to research data format expected by semantic extractors.
    
    Args:
        observations: List of observations (string or structured)
        
    Returns:
        List of research data items with content and metadata
    """
    research_data = []
    
    for obs in observations:
        structured_obs = ensure_structured_observation(obs)
        
        # Base research data item
        data_item = {
            "content": structured_obs.content,
            "confidence": structured_obs.confidence,
            "source_id": structured_obs.source_id,
            "extraction_method": structured_obs.extraction_method.value
        }
        
        # Add entity/metric metadata if available
        if structured_obs.entity_tags:
            data_item["entities"] = structured_obs.entity_tags
        
        if structured_obs.metric_values:
            data_item["metrics"] = structured_obs.metric_values
            
        research_data.append(data_item)
    
    return research_data


def observation_to_text(
    obs: Union[str, StructuredObservation, Dict[str, Any]]
) -> str:
    """Return a human-readable string for any observation payload."""

    import logging
    logger = logging.getLogger(__name__)

    if isinstance(obs, StructuredObservation):
        return obs.content

    if isinstance(obs, dict):
        # CRITICAL FIX: Don't return str(dict)!
        # Check if this is a serialized StructuredObservation
        if 'content' in obs:
            content = obs['content']

            # Handle nested dict content (shouldn't happen but defensive)
            if isinstance(content, dict) and 'content' in content:
                # Recursive call for nested observation dicts
                return observation_to_text(content)

            # Return content if it's a string
            if isinstance(content, str):
                return content

            # Try to convert non-string content
            if content:
                return str(content)

        # Last resort - this should rarely happen
        logger.warning(f"Unable to extract text from dict observation: {obs}")
        return ""

    return str(obs)


def observations_to_text_list(
    observations: Iterable[Union[str, StructuredObservation, Dict[str, Any]]]
) -> List[str]:
    """Convert a sequence of observations into plain-text strings."""

    return [observation_to_text(obs) for obs in observations]
