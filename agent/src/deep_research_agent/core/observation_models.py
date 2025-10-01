"""
Structured observation models for enhanced data extraction and table population.

These models provide a structured way to store and access research observations
with entity/metric metadata for better table generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Iterable
from enum import Enum


class ExtractionMethod(str, Enum):
    """Method used to extract the observation data."""
    LLM = "llm"
    PATTERN = "pattern"
    EXPLICIT = "explicit"
    SEARCH = "search"


@dataclass
class StructuredObservation:
    """
    Enhanced observation with structured metadata for table population.

    This replaces simple string observations with structured data that can
    be easily extracted for table generation.
    """
    content: str
    entity_tags: List[str] = field(default_factory=list)
    metric_values: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_id: Optional[str] = None
    extraction_method: ExtractionMethod = ExtractionMethod.LLM
    step_id: Optional[str] = None  # CRITICAL: Enables filtering observations by step for section-specific content
    
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
    
    @classmethod
    def from_string(cls, content: str, confidence: float = 0.5) -> 'StructuredObservation':
        """Create from legacy string observation with basic parsing."""
        return cls(
            content=content,
            confidence=confidence,
            extraction_method=ExtractionMethod.PATTERN
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "entity_tags": self.entity_tags,
            "metric_values": self.metric_values,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "extraction_method": self.extraction_method.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredObservation':
        """Create from dictionary."""
        return cls(
            content=data.get("content", ""),
            entity_tags=data.get("entity_tags", []),
            metric_values=data.get("metric_values", {}),
            confidence=data.get("confidence", 1.0),
            source_id=data.get("source_id"),
            extraction_method=ExtractionMethod(data.get("extraction_method", "llm"))
        )


def ensure_structured_observation(obs: Union[str, StructuredObservation]) -> StructuredObservation:
    """
    Convert string observations to structured observations for backward compatibility.
    
    Args:
        obs: Either a string or StructuredObservation
        
    Returns:
        StructuredObservation instance
    """
    if isinstance(obs, str):
        return StructuredObservation.from_string(obs)
    elif isinstance(obs, StructuredObservation):
        return obs
    else:
        # Handle other types by converting to string first
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

    if isinstance(obs, StructuredObservation):
        return obs.content

    if isinstance(obs, dict):
        return str(obs.get("content", obs))

    return str(obs)


def observations_to_text_list(
    observations: Iterable[Union[str, StructuredObservation, Dict[str, Any]]]
) -> List[str]:
    """Convert a sequence of observations into plain-text strings."""

    return [observation_to_text(obs) for obs in observations]
