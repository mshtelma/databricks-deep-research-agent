"""
ObservationConverter - Universal converter for observation data.

This module provides consistent conversion between different observation formats
used throughout the research agent system.
"""

from typing import Union, Dict, Any, List, Optional
import logging

from .observation_models import StructuredObservation

logger = logging.getLogger(__name__)


class ObservationConverter:
    """
    Universal converter for observation data.
    
    Handles conversion between:
    - Raw strings
    - Dictionaries (both structured and arbitrary)
    - StructuredObservation objects
    
    This ensures type safety and consistency across the system.
    """
    
    @staticmethod
    def to_structured(data: Union[str, Dict, StructuredObservation, Any]) -> StructuredObservation:
        """
        Convert any observation data to StructuredObservation.
        
        Args:
            data: Input data in any supported format
            
        Returns:
            StructuredObservation instance
            
        Note:
            This method is defensive and will handle ANY input type,
            ensuring the system doesn't crash on unexpected data.
        """
        try:
            # Already a StructuredObservation
            if isinstance(data, StructuredObservation):
                return data
            
            # Dictionary - check if it's a serialized StructuredObservation
            elif isinstance(data, dict):
                # Has the structure of a StructuredObservation
                if "content" in data:
                    try:
                        return StructuredObservation.from_dict(data)
                    except Exception as e:
                        logger.debug(f"Failed to parse dict as StructuredObservation: {e}")
                        # Fall through to create new observation
                
                # Arbitrary dict - convert to observation
                return StructuredObservation(
                    content=str(data),
                    entity_tags=list(data.keys()) if len(data) < 10 else [],
                    metric_values={"original_type": "dict", "key_count": len(data)}
                )
            
            # String - simple conversion
            elif isinstance(data, str):
                return StructuredObservation.from_string(data)
            
            # List - convert to string representation
            elif isinstance(data, list):
                content = "\n".join(str(item) for item in data)
                return StructuredObservation(
                    content=content,
                    metric_values={"original_type": "list", "length": len(data)}
                )
            
            # None - return empty observation
            elif data is None:
                return StructuredObservation(
                    content="",
                    metric_values={"original_type": "None"}
                )
            
            # Any other type - convert to string
            else:
                return StructuredObservation(
                    content=str(data),
                    metric_values={"original_type": type(data).__name__}
                )
                
        except Exception as e:
            logger.error(f"Error converting observation data: {e}")
            # Last resort - return empty observation to prevent crash
            return StructuredObservation(
                content=str(data) if data else "",
                metric_values={"conversion_error": str(e)}
            )
    
    @staticmethod
    def normalize_list(observations: Optional[List[Any]]) -> List[StructuredObservation]:
        """
        Convert a list of mixed observation types to StructuredObservations.
        
        Args:
            observations: List of observations in any format (can be None)
            
        Returns:
            List of StructuredObservation instances
            
        Note:
            Handles None, empty lists, and mixed types gracefully.
        """
        if not observations:
            return []
        
        normalized = []
        for i, obs in enumerate(observations):
            try:
                normalized.append(ObservationConverter.to_structured(obs))
            except Exception as e:
                logger.error(f"Failed to normalize observation {i}: {e}")
                # Add placeholder to maintain list integrity
                normalized.append(StructuredObservation(
                    content=f"[Error processing observation {i}]",
                    metric_values={"error": str(e)}
                ))
        
        return normalized
    
    @staticmethod
    def to_dict(observation: Union[StructuredObservation, Dict, str]) -> Dict[str, Any]:
        """
        Convert observation to dictionary format.
        
        Args:
            observation: Observation in any format
            
        Returns:
            Dictionary representation
        """
        if isinstance(observation, StructuredObservation):
            return observation.to_dict()
        elif isinstance(observation, dict):
            return observation
        else:
            # Convert to structured first, then to dict
            structured = ObservationConverter.to_structured(observation)
            return structured.to_dict()
    
    @staticmethod
    def extract_content(observation: Union[StructuredObservation, Dict, str]) -> str:
        """
        Extract text content from any observation format.
        
        Args:
            observation: Observation in any format
            
        Returns:
            String content
        """
        if isinstance(observation, StructuredObservation):
            return observation.content
        elif isinstance(observation, dict):
            return observation.get("content", str(observation))
        elif isinstance(observation, str):
            return observation
        else:
            return str(observation)