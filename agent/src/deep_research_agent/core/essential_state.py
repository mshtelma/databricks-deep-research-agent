"""
Essential state management with preservation guarantees.

This module provides state pruning that preserves critical information
while reducing memory usage by removing unnecessary fields.
"""

from typing import Dict, Any, Set, List
from . import get_logger

logger = get_logger(__name__)


class EssentialStateManager:
    """Manages state preservation during pruning operations."""
    
    # Fields that must NEVER be removed as they contain core research data
    CRITICAL_FIELDS = {
        # Core research data
        "research_topic",
        "original_query", 
        "observations",
        "search_results",
        "citations",
        "section_research_results",
        
        # Plan and execution state
        "current_plan",
        "current_step",
        "plan_quality_score",
        "plan_iterations",
        "steps_completed",
        
        # Agent coordination
        "current_agent",
        "handoff_history", 
        "routing_decisions",
        
        # Quality and verification
        "factuality_score",
        "confidence_score",
        "verification_results",
        "grounding_data",
        
        # Results and output
        "final_report",
        "report_style",
        "structured_entities",
        
        # Session management
        "query_session_id",
        "workflow_status"
    }
    
    # Fields that can be safely removed to save memory  
    REMOVABLE_FIELDS = {
        # Debug and logging
        "debug_info",
        "execution_traces", 
        "performance_metrics",
        "timing_data",
        
        # Temporary processing
        "temp_analysis",
        "intermediate_results",
        "scratch_data",
        
        # Cache-like data that can be regenerated
        "cached_embeddings",
        "url_metadata_cache",
        "processed_content_cache",
        
        # Verbose logging
        "detailed_logs",
        "step_by_step_logs"
    }
    
    # Fields that should be preserved if they contain data, removed if empty
    CONDITIONAL_FIELDS = {
        "reflexion_results",
        "entity_validation_errors", 
        "search_failures",
        "api_call_logs",
        "tool_execution_logs"
    }
    
    @classmethod
    def create_essential_state(cls, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create essential state by preserving critical fields and removing unnecessary ones.
        
        Args:
            full_state: Complete state dictionary
            
        Returns:
            Essential state with critical data preserved
        """
        essential_state = {}
        preserved_count = 0
        removed_count = 0
        conditional_kept = 0
        conditional_removed = 0
        
        # Always preserve critical fields
        for field in cls.CRITICAL_FIELDS:
            if field in full_state:
                essential_state[field] = full_state[field]
                preserved_count += 1
        
        # Handle conditional fields - keep if they have meaningful data
        for field in cls.CONDITIONAL_FIELDS:
            if field in full_state:
                value = full_state[field]
                if cls._has_meaningful_data(value):
                    essential_state[field] = value
                    conditional_kept += 1
                else:
                    conditional_removed += 1
        
        # Preserve any field not explicitly marked as removable
        for field, value in full_state.items():
            if (field not in cls.CRITICAL_FIELDS and 
                field not in cls.CONDITIONAL_FIELDS and
                field not in cls.REMOVABLE_FIELDS):
                essential_state[field] = value
                preserved_count += 1
        
        # Count removed fields
        removed_count = len([f for f in full_state.keys() if f in cls.REMOVABLE_FIELDS])
        
        logger.info(f"State pruning complete: preserved {preserved_count} critical, "
                   f"kept {conditional_kept}/{conditional_kept + conditional_removed} conditional, "
                   f"removed {removed_count} unnecessary fields")
        
        return essential_state
    
    @classmethod
    def _has_meaningful_data(cls, value: Any) -> bool:
        """Check if a value contains meaningful data worth preserving."""
        if value is None:
            return False
        if isinstance(value, (list, dict, set)):
            return len(value) > 0
        if isinstance(value, str):
            return len(value.strip()) > 0
        return True
    
    @classmethod
    def validate_preservation(cls, original_state: Dict[str, Any], 
                            essential_state: Dict[str, Any]) -> List[str]:
        """
        Validate that essential data was preserved correctly.
        
        Args:
            original_state: Original full state
            essential_state: Pruned essential state
            
        Returns:
            List of validation errors (empty if all good)
        """
        errors = []
        
        # Check that all critical fields were preserved
        for field in cls.CRITICAL_FIELDS:
            if field in original_state and field not in essential_state:
                errors.append(f"Critical field '{field}' was lost during state pruning")
        
        # Check for data corruption in preserved fields
        for field in cls.CRITICAL_FIELDS:
            if field in original_state and field in essential_state:
                original_val = original_state[field]
                essential_val = essential_state[field]
                
                # Check for major structural changes
                if type(original_val) != type(essential_val):
                    errors.append(f"Field '{field}' type changed from {type(original_val)} to {type(essential_val)}")
                
                # Check list/dict sizes didn't shrink unexpectedly
                if isinstance(original_val, (list, dict)):
                    if len(original_val) > len(essential_val):
                        errors.append(f"Field '{field}' lost data: {len(original_val)} → {len(essential_val)} items")
        
        return errors
    
    @classmethod
    def get_field_classification(cls, field_name: str) -> str:
        """Get the classification of a field for debugging."""
        if field_name in cls.CRITICAL_FIELDS:
            return "CRITICAL"
        elif field_name in cls.CONDITIONAL_FIELDS:
            return "CONDITIONAL"
        elif field_name in cls.REMOVABLE_FIELDS:
            return "REMOVABLE"
        else:
            return "UNKNOWN"