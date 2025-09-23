"""
Table data models for table generation system.

Separated from table_generator to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Any, Optional, List, Dict


@dataclass
class TableCell:
    """Represents a single table cell with metadata."""
    value: Any
    formatted_value: str
    confidence: float = 1.0
    source: Optional[str] = None
    citation_id: Optional[str] = None
    is_estimated: bool = False
    
    def __str__(self) -> str:
        return self.formatted_value


@dataclass 
class TableRow:
    """Represents a table row with cells and metadata."""
    row_id: str
    label: str
    cells: List['TableCell']
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
