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
        self._dimensional_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
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
            
            metadata = {
                'value': value,
                'unit': getattr(dp, 'unit', 'unitless'),
                'confidence': getattr(dp, 'confidence', 0.8),
                'source': getattr(dp, 'source_observation_id', None)
            }
            
            self._entity_index[entity][metric] = metadata
            
            # Build dimensional index for multi-dimensional support
            # Parse entity name for dimensions (e.g., "Spain_Single" -> {"country": "Spain", "scenario": "Single"})
            dimensions = self._parse_entity_dimensions(entity)
            if dimensions:
                dim_key = self._make_dimension_key(dimensions)
                if dim_key not in self._dimensional_index:
                    self._dimensional_index[dim_key] = {}
                self._dimensional_index[dim_key][metric] = metadata
    
    def _parse_entity_dimensions(self, entity: str) -> Optional[Dict[str, str]]:
        """Parse entity name into dimensions.
        
        Examples:
            "Spain_Single" -> {"country": "Spain", "scenario": "Single"}
            "Q1_2024" -> {"quarter": "Q1", "year": "2024"}
            "Spain" -> {"entity": "Spain"} (single dimension)
        """
        if '_' not in entity:
            return {"entity": entity}
        
        parts = entity.split('_')
        if len(parts) == 2:
            # Try to infer dimension types
            # Common patterns: Country_Scenario, Entity_Year, Quarter_Year
            return {"dim1": parts[0], "dim2": parts[1]}
        elif len(parts) == 3:
            return {"dim1": parts[0], "dim2": parts[1], "dim3": parts[2]}
        
        return {"entity": entity}
    
    def _make_dimension_key(self, dimensions: Dict[str, str]) -> str:
        """Create a consistent key from dimension dictionary."""
        return "_".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
    
    def get_scalar(
        self,
        entity: str,
        metric: str,
        dimensions: Optional[Dict[str, str]] = None
    ) -> Optional[Union[float, str]]:
        """Get a single scalar value by entity and metric name.
        
        Args:
            entity: Entity name (e.g., "Spain", "Product A")
            metric: Metric name (e.g., "gross_income", "tax_rate")
            dimensions: Optional dimension filter (e.g., {"scenario": "Single"})
        
        Returns:
            The metric value, or None if not found
        
        Example:
            >>> income = ctx.get_scalar("Spain", "gross_income")
            >>> # Returns: 50000.0
            >>> income_single = ctx.get_scalar("Spain", "gross_income", {"scenario": "Single"})
            >>> # Returns: 50000.0 (for Spain in Single scenario)
        """
        # If dimensions provided, try dimensional lookup first
        if dimensions:
            # Build entity key with dimensions
            entity_with_dims = self._build_entity_key(entity, dimensions)
            if entity_with_dims in self._entity_index and metric in self._entity_index[entity_with_dims]:
                return self._entity_index[entity_with_dims][metric]['value']
            
            # Try dimensional index lookup
            dim_key = self._make_dimension_key({**dimensions, "entity": entity})
            if dim_key in self._dimensional_index and metric in self._dimensional_index[dim_key]:
                return self._dimensional_index[dim_key][metric]['value']
        
        # Fallback to simple entity lookup
        if entity not in self._entity_index:
            # Try fuzzy matching on entity name
            fuzzy_match = self._find_similar_entity(entity)
            if fuzzy_match:
                logger.info(f"Entity '{entity}' not found, using similar entity '{fuzzy_match}'")
                entity = fuzzy_match
            else:
                logger.debug(f"Entity '{entity}' not found in data context. Available: {list(self._entity_index.keys())[:10]}")
                return None
        
        if metric not in self._entity_index[entity]:
            # Try fuzzy matching on metric name
            fuzzy_metric = self._find_similar_metric(entity, metric)
            if fuzzy_metric:
                logger.info(f"Metric '{metric}' not found for '{entity}', using similar metric '{fuzzy_metric}'")
                metric = fuzzy_metric
            else:
                logger.debug(f"Metric '{metric}' not found for entity '{entity}'. Available: {list(self._entity_index[entity].keys())[:10]}")
                return None
        
        return self._entity_index[entity][metric]['value']
    
    def _find_similar_entity(self, target: str) -> Optional[str]:
        """Find similar entity name using fuzzy matching.
        
        Args:
            target: Target entity name
        
        Returns:
            Similar entity name, or None if no close match
        """
        if not self._entity_index:
            return None
        
        target_lower = target.lower().replace('_', '').replace('-', '').replace(' ', '')
        best_match = None
        min_distance = float('inf')
        
        for entity in self._entity_index.keys():
            entity_lower = entity.lower().replace('_', '').replace('-', '').replace(' ', '')
            
            # Exact match after normalization
            if target_lower == entity_lower:
                return entity
            
            # Check if one contains the other
            if target_lower in entity_lower or entity_lower in target_lower:
                if len(entity_lower) - len(target_lower) < min_distance:
                    min_distance = len(entity_lower) - len(target_lower)
                    best_match = entity
        
        # Return match if it's reasonably close
        if best_match and min_distance <= 5:
            return best_match
        
        return None
    
    def _find_similar_metric(self, entity: str, target: str) -> Optional[str]:
        """Find similar metric name using fuzzy matching.
        
        Args:
            entity: Entity name
            target: Target metric name
        
        Returns:
            Similar metric name, or None if no close match
        """
        if entity not in self._entity_index:
            return None
        
        metrics = self._entity_index[entity].keys()
        if not metrics:
            return None
        
        target_lower = target.lower().replace('_', '').replace('-', '').replace(' ', '')
        best_match = None
        min_distance = float('inf')
        
        for metric in metrics:
            metric_lower = metric.lower().replace('_', '').replace('-', '').replace(' ', '')
            
            # Exact match after normalization
            if target_lower == metric_lower:
                return metric
            
            # Check if one contains the other
            if target_lower in metric_lower or metric_lower in target_lower:
                if len(metric_lower) - len(target_lower) < min_distance:
                    min_distance = len(metric_lower) - len(target_lower)
                    best_match = metric
        
        # Return match if it's reasonably close
        if best_match and min_distance <= 5:
            return best_match
        
        return None
    
    def _build_entity_key(self, entity: str, dimensions: Dict[str, str]) -> str:
        """Build entity key from base entity and dimensions.
        
        Examples:
            entity="Spain", dimensions={"scenario": "Single"} -> "Spain_Single"
            entity="Q1", dimensions={"year": "2024"} -> "Q1_2024"
        """
        dim_values = list(dimensions.values())
        if entity not in dim_values:
            return f"{entity}_{'_'.join(dim_values)}"
        return '_'.join(dim_values)
    
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
    
    def get_by_dimensions(
        self,
        metric: str,
        dimensions: Dict[str, str]
    ) -> Optional[Union[float, str]]:
        """Get metric value by dimensions only (without explicit entity).
        
        Args:
            metric: Metric name
            dimensions: Dimension mapping (e.g., {"country": "Spain", "scenario": "Single"})
        
        Returns:
            The metric value, or None if not found
        
        Example:
            >>> income = ctx.get_by_dimensions(
            ...     "net_income",
            ...     {"country": "Spain", "scenario": "Single"}
            ... )
        """
        dim_key = self._make_dimension_key(dimensions)
        if dim_key in self._dimensional_index and metric in self._dimensional_index[dim_key]:
            return self._dimensional_index[dim_key][metric]['value']
        
        # Try building entity key and looking up
        entity_key = '_'.join(dimensions.values())
        return self.get_scalar(entity_key, metric)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        entity_count = len(self._entity_index)
        total_metrics = sum(len(metrics) for metrics in self._entity_index.values())
        dimensional_count = len(self._dimensional_index)
        return (f"MetricDataContext(entities={entity_count}, "
                f"total_metrics={total_metrics}, "
                f"dimensional_combinations={dimensional_count})")


__all__ = ["MetricDataContext"]


