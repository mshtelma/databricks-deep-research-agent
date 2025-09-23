"""
Presentation requirements analysis for intelligent report formatting.

This module provides the data structures and analysis for determining
optimal presentation formats based on query semantics and background information.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from .requirements import TableSpecification, RequiredDataPoint, TableStructureType

logger = logging.getLogger(__name__)


class PresentationType(str, Enum):
    """Types of presentation formats."""
    NARRATIVE = "narrative"
    TABLE = "table" 
    MIXED = "mixed"
    LIST = "list"
    TIMELINE = "timeline"


class TableType(str, Enum):
    """Types of table structures."""
    COMPARATIVE = "comparative"
    SUMMARY = "summary"
    MATRIX = "matrix"
    RANKING = "ranking"
    BREAKDOWN = "breakdown"
    NONE = "none"


@dataclass
class PresentationRequirements:
    """
    Planner's analysis of optimal presentation format.
    
    This replaces primitive pattern matching with semantic analysis
    of what presentation formats would best serve the user's query.
    """
    needs_table: bool
    table_reasoning: str
    optimal_table_type: TableType
    suggested_entities: List[str]
    suggested_metrics: List[str]
    primary_presentation: PresentationType
    confidence: float
    
    # Additional metadata
    entity_reasoning: str = ""
    metric_reasoning: str = ""
    alternative_presentations: List[PresentationType] = None
    
    def __post_init__(self):
        """Initialize alternative presentations if not provided."""
        if self.alternative_presentations is None:
            self.alternative_presentations = []
    
    def to_table_specification(self) -> Optional[TableSpecification]:
        """Convert to table specification if table is needed."""
        if not self.needs_table or self.optimal_table_type == TableType.NONE:
            return None
            
        # Convert our enum to the requirements module enum
        structure_type_mapping = {
            TableType.COMPARATIVE: TableStructureType.COMPARATIVE,
            TableType.SUMMARY: TableStructureType.SUMMARY,
            TableType.MATRIX: TableStructureType.DATA_MATRIX,
            TableType.RANKING: TableStructureType.RANKING,
            TableType.BREAKDOWN: TableStructureType.SUMMARY,
        }
        
        structure_type = structure_type_mapping.get(
            self.optimal_table_type, 
            TableStructureType.COMPARATIVE
        )
        
        return TableSpecification(
            structure_type=structure_type,
            rows_represent=" or ".join(self.suggested_entities) if self.suggested_entities else "entities", 
            columns_represent=" and ".join(self.suggested_metrics) if self.suggested_metrics else "attributes",
            required_data_points=[
                metric.replace(" ", "_").lower()
                for metric in self.suggested_metrics[:5]  # Limit to 5 key metrics
            ],
            format_specification="markdown"
        )
    
    def is_high_confidence(self) -> bool:
        """Check if this analysis is high confidence."""
        return self.confidence >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "needs_table": self.needs_table,
            "table_reasoning": self.table_reasoning,
            "optimal_table_type": self.optimal_table_type.value,
            "suggested_entities": self.suggested_entities,
            "suggested_metrics": self.suggested_metrics,
            "primary_presentation": self.primary_presentation.value,
            "confidence": self.confidence,
            "entity_reasoning": self.entity_reasoning,
            "metric_reasoning": self.metric_reasoning,
            "alternative_presentations": [alt.value for alt in self.alternative_presentations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresentationRequirements':
        """Create from dictionary."""
        return cls(
            needs_table=data.get("needs_table", False),
            table_reasoning=data.get("table_reasoning", ""),
            optimal_table_type=TableType(data.get("optimal_table_type", "none")),
            suggested_entities=data.get("suggested_entities", []),
            suggested_metrics=data.get("suggested_metrics", []),
            primary_presentation=PresentationType(data.get("primary_presentation", "narrative")),
            confidence=data.get("confidence", 0.0),
            entity_reasoning=data.get("entity_reasoning", ""),
            metric_reasoning=data.get("metric_reasoning", ""),
            alternative_presentations=[
                PresentationType(alt) for alt in data.get("alternative_presentations", [])
            ]
        )
