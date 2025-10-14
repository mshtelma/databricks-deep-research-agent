"""Clean data interface for LLM-generated calculation code.

This module provides a simplified API for accessing extracted research data,
making it easier for LLMs to generate correct and readable Python calculations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..report_generation.models import DataPoint
from .. import get_logger


logger = get_logger(__name__)


class MetricDataContext:
    """Clean data interface for LLM-generated calculation code.
    
    This class provides a simple, safe API for accessing extracted research data
    in calculation code. It abstracts away the complexity of navigating raw data
    structures and provides pandas DataFrames for convenient computation.
    
    Example usage in generated code:
        >>> ctx = MetricDataContext(data_points)
        >>> spain_income = ctx.get_scalar("Spain", "gross_income")
        >>> spain_tax = ctx.get_scalar("Spain", "total_tax_paid")
        >>> effective_rate = (spain_tax / spain_income) * 100
    """
    
    def __init__(
        self,
        extracted_data: List[DataPoint],
        constants: Optional[Dict[str, Any]] = None
    ):
        """Initialize context with extracted data points.
        
        Args:
            extracted_data: List of DataPoint objects from research
            constants: Optional domain-specific constants (e.g., tax thresholds)
        """
        self._data_points = extracted_data
        self._constants = constants or {}
        self._entity_index: Dict[str, Dict[str, Any]] = {}
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build internal indices for fast lookup."""
        for dp in self._data_points:
            entity = dp.entity
            metric = dp.metric
            
            if entity not in self._entity_index:
                self._entity_index[entity] = {}
            
            # Store the value (prefer numeric conversion)
            value = dp.value
            if isinstance(value, str):
                # Try to convert string to number
                try:
                    value = float(value.replace(',', '').replace('€', '').replace('$', '').strip())
                except (ValueError, AttributeError):
                    pass  # Keep as string
            
            self._entity_index[entity][metric] = {
                'value': value,
                'unit': getattr(dp, 'unit', 'unitless'),
                'confidence': getattr(dp, 'confidence', 0.8),
                'source': getattr(dp, 'source_observation_id', None)
            }
    
    def get_scalar(self, entity: str, metric: str) -> Optional[Union[float, str]]:
        """Get a single scalar value by entity and metric name.
        
        Args:
            entity: Entity name (e.g., "Spain", "Product A")
            metric: Metric name (e.g., "gross_income", "tax_rate")
        
        Returns:
            The metric value, or None if not found
        
        Example:
            >>> income = ctx.get_scalar("Spain", "gross_income")
            >>> # Returns: 50000.0
        """
        if entity not in self._entity_index:
            logger.debug(f"Entity '{entity}' not found in data context")
            return None
        
        if metric not in self._entity_index[entity]:
            logger.debug(f"Metric '{metric}' not found for entity '{entity}'")
            return None
        
        return self._entity_index[entity][metric]['value']
    
    def get_table(self, entity: str) -> pd.DataFrame:
        """Get all metrics for an entity as a pandas DataFrame.
        
        Args:
            entity: Entity name
        
        Returns:
            DataFrame with columns: metric, value, unit, confidence
        
        Example:
            >>> df = ctx.get_table("Spain")
            >>> # Returns DataFrame with all Spain metrics
        """
        if entity not in self._entity_index:
            logger.warning(f"Entity '{entity}' not found, returning empty DataFrame")
            return pd.DataFrame(columns=['metric', 'value', 'unit', 'confidence'])
        
        rows = []
        for metric, data in self._entity_index[entity].items():
            rows.append({
                'metric': metric,
                'value': data['value'],
                'unit': data['unit'],
                'confidence': data['confidence']
            })
        
        return pd.DataFrame(rows)
    
    def get_all_entities(self) -> List[str]:
        """Get list of all available entities.
        
        Returns:
            Sorted list of entity names
        
        Example:
            >>> entities = ctx.get_all_entities()
            >>> # Returns: ["France", "Spain", "UK"]
        """
        return sorted(self._entity_index.keys())
    
    def get_available_metrics(self, entity: str) -> List[str]:
        """Get list of available metrics for an entity.
        
        Args:
            entity: Entity name
        
        Returns:
            Sorted list of metric names, or empty list if entity not found
        
        Example:
            >>> metrics = ctx.get_available_metrics("Spain")
            >>> # Returns: ["gross_income", "tax_rate", "total_tax_paid"]
        """
        if entity not in self._entity_index:
            return []
        
        return sorted(self._entity_index[entity].keys())
    
    @property
    def constants(self) -> Dict[str, Any]:
        """Get domain-specific constants.
        
        Constants might include tax thresholds, conversion rates, standard values, etc.
        
        Returns:
            Dictionary of constant name -> value
        
        Example:
            >>> tax_threshold = ctx.constants.get('spain_tax_bracket_1', 12450)
        """
        return self._constants
    
    def has_entity(self, entity: str) -> bool:
        """Check if entity exists in the data context.
        
        Args:
            entity: Entity name to check
        
        Returns:
            True if entity exists, False otherwise
        """
        return entity in self._entity_index
    
    def has_metric(self, entity: str, metric: str) -> bool:
        """Check if specific metric exists for an entity.
        
        Args:
            entity: Entity name
            metric: Metric name
        
        Returns:
            True if metric exists for entity, False otherwise
        """
        return (entity in self._entity_index and 
                metric in self._entity_index[entity])
    
    def get_metadata(self, entity: str, metric: str) -> Optional[Dict[str, Any]]:
        """Get full metadata for a specific metric.
        
        Args:
            entity: Entity name
            metric: Metric name
        
        Returns:
            Dict with value, unit, confidence, source, or None if not found
        """
        if not self.has_metric(entity, metric):
            return None
        
        return self._entity_index[entity][metric].copy()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        entity_count = len(self._entity_index)
        total_metrics = sum(len(metrics) for metrics in self._entity_index.values())
        return (f"MetricDataContext(entities={entity_count}, "
                f"total_metrics={total_metrics})")


__all__ = ["MetricDataContext"]


