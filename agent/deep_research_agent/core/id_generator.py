"""
Centralized ID generation for plan components.

This module provides consistent ID generation for steps and sections,
ensuring they match properly during execution.
"""

import re
from typing import Tuple, Optional


class PlanIDGenerator:
    """Centralized ID generation for plan components."""
    
    @staticmethod
    def generate_step_id(index: int) -> str:
        """Generate consistent step ID."""
        return f"step_{index:03d}"
    
    @staticmethod
    def generate_section_id(index: int) -> str:
        """Generate section ID that matches step ID."""
        # CRITICAL: Return step ID format, not section_XXX
        # This ensures sections and steps have matching IDs
        return f"step_{index:03d}"
    
    @staticmethod
    def parse_id(id_str: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse any ID format to get type and number."""
        if not id_str:
            return None, None
            
        # Match step_001, section_002, req_validation_003, etc.
        match = re.match(r"(step|section|req)_(?:validation_)?(\d+)", id_str)
        if match:
            return match.group(1), int(match.group(2))
        
        # Try to extract just numbers
        number_match = re.search(r'(\d+)', id_str)
        if number_match:
            return "unknown", int(number_match.group(1))
            
        return None, None
    
    @staticmethod
    def normalize_id(raw_id: Optional[str]) -> str:
        if raw_id is None:
            return PlanIDGenerator.generate_step_id(1)

        raw = str(raw_id).strip()
        if not raw:
            return PlanIDGenerator.generate_step_id(1)

        id_type, number = PlanIDGenerator.parse_id(raw)
        if id_type in {"step", "section", "req"} and number is not None:
            return PlanIDGenerator.generate_step_id(number)
        if number is not None:
            return PlanIDGenerator.generate_step_id(number)

        return raw
    
    @staticmethod
    def extract_index_from_title(title: str) -> Optional[int]:
        """Extract numeric index from section titles like 'Executive Summary' -> 1."""
        # Common section patterns and their typical order
        common_sections = {
            "executive summary": 1,
            "introduction": 1, 
            "overview": 1,
            "methodology": 2,
            "methods": 2,
            "approach": 2,
            "analysis": 3,
            "findings": 3,
            "results": 3,
            "discussion": 4,
            "evaluation": 4,
            "comparison": 4,
            "conclusion": 5,
            "recommendations": 5,
            "summary": 5,
            "appendix": 6,
            "sources": 6,
            "references": 6
        }
        
        title_lower = title.lower().strip()
        
        # First try exact match
        if title_lower in common_sections:
            return common_sections[title_lower]
        
        # Try partial matches
        for pattern, index in common_sections.items():
            if pattern in title_lower or title_lower in pattern:
                return index
        
        # Look for numbers in the title
        number_match = re.search(r'\b(\d+)\b', title)
        if number_match:
            return int(number_match.group(1))
        
        # Default to None if no pattern matches
        return None