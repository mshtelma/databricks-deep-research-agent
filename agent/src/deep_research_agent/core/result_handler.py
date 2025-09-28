"""
ResultHandler - Centralized handling of agent results.

This module provides the single source of truth for processing agent results,
eliminating code duplication and ensuring consistent behavior across all nodes.
"""

from typing import Dict, Any, Optional, List
import logging
import time

from .command_converter import CommandConverter
from .state_validator import StateValidator

logger = logging.getLogger(__name__)


class ResultHandler:
    """
    Centralized handling of agent results.
    
    This is the SINGLE source of truth for result processing,
    eliminating the 12+ duplicate hasattr() patterns in workflow_nodes.
    """
    
    # Critical fields that must always be preserved
    CRITICAL_FIELDS = [
        'messages',
        'research_topic',
        'observations', 
        'citations', 
        'search_results',
        'section_research_results',
        'completed_steps',
        'current_plan',
        'final_report'
    ]
    
    @staticmethod
    def process_agent_result(
        agent_name: str,
        result: Any,
        current_state: Dict[str, Any],
        logger: Optional[Any] = None,
        preserve_critical: bool = True
    ) -> Dict[str, Any]:
        """
        Process any agent result and return normalized state update.
        
        This is the SINGLE source of truth for result processing.
        
        Args:
            agent_name: Name of the agent that produced the result
            result: Agent result in any format (Command, dict, object, etc.)
            current_state: Current workflow state
            logger: Optional logger for debugging
            preserve_critical: Whether to preserve critical fields
            
        Returns:
            Dictionary containing state updates
            
        Note:
            This method guarantees a dict return and preserves critical data.
        """
        try:
            # Step 1: Convert result to dict
            update_dict = CommandConverter.to_dict(result, {})
            
            # Step 2: Extract goto if present
            goto = CommandConverter.extract_goto(result)
            
            # Step 3: Validate the update (if validator available)
            try:
                validated_update = StateValidator.validate_update(
                    agent_name, 
                    update_dict,
                    current_state
                )
            except Exception as e:
                # StateValidator might not be available or configured
                if logger:
                    logger.debug(f"State validation skipped: {e}")
                validated_update = update_dict
            
            # Step 4: Preserve critical fields if requested
            if preserve_critical:
                for field in ResultHandler.CRITICAL_FIELDS:
                    if field in current_state and field not in validated_update:
                        # Preserve the field from current state
                        validated_update[field] = current_state[field]
                    elif field in current_state and field in validated_update:
                        # Merge lists and dicts intelligently
                        ResultHandler._merge_field(
                            validated_update, 
                            current_state, 
                            field
                        )
            
            # Step 5: Add goto if present (use special key to avoid conflicts)
            if goto:
                validated_update['__next_node__'] = goto
            
            # Step 6: Add metadata
            validated_update['__last_agent__'] = agent_name
            validated_update['__update_timestamp__'] = time.time()
            
            if logger:
                logger.debug(
                    f"Processed {agent_name} result: "
                    f"{len(validated_update)} fields updated"
                )
            
            return validated_update
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to process {agent_name} result: {e}")
            
            # Critical: Return current state to prevent data loss
            return current_state.copy()
    
    @staticmethod
    def _merge_field(
        target_dict: Dict[str, Any],
        source_dict: Dict[str, Any],
        field: str
    ) -> None:
        """
        Intelligently merge a field from source into target.
        
        Args:
            target_dict: Target dictionary (will be modified)
            source_dict: Source dictionary
            field: Field name to merge
        """
        target_value = target_dict.get(field)
        source_value = source_dict.get(field)
        
        # Both are lists - extend
        if isinstance(target_value, list) and isinstance(source_value, list):
            # Avoid duplicates for certain fields
            if field in ['observations', 'citations', 'completed_steps']:
                # Merge without duplicates
                existing = set(str(item) for item in source_value)
                for item in target_value:
                    if str(item) not in existing:
                        source_value.append(item)
                target_dict[field] = source_value
            else:
                # Simple extend
                target_dict[field] = source_value + target_value
        
        # Both are dicts - update
        elif isinstance(target_value, dict) and isinstance(source_value, dict):
            merged = source_value.copy()
            merged.update(target_value)
            target_dict[field] = merged
        
        # Otherwise, target wins (it's the newer value)
    
    @staticmethod
    def extract_next_node(state: Dict[str, Any]) -> Optional[str]:
        """
        Extract the next node destination from state.
        
        Args:
            state: Workflow state
            
        Returns:
            Next node name or None
        """
        return state.get('__next_node__') or state.get('goto')
    
    @staticmethod
    def clean_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove internal metadata fields before returning to workflow.
        
        Args:
            state: State with metadata
            
        Returns:
            Cleaned state
        """
        cleaned = state.copy()
        
        # Remove internal metadata
        metadata_keys = [
            '__next_node__',
            '__last_agent__', 
            '__update_timestamp__'
        ]
        
        for key in metadata_keys:
            cleaned.pop(key, None)
        
        return cleaned