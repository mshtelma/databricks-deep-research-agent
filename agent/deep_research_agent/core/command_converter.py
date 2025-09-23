"""
CommandConverter - Consistent Command to dict conversion.

This module provides reliable conversion between LangGraph Command objects
and dictionaries, eliminating fragile hasattr() checks throughout the codebase.
"""

from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CommandConverter:
    """
    Consistent Command to dict conversion.
    
    Handles conversion from:
    - LangGraph Command objects
    - Dict responses
    - Objects with update/goto attributes
    - Dataclasses and custom objects
    
    This eliminates the fragile hasattr() pattern used throughout the codebase.
    """
    
    @staticmethod
    def to_dict(result: Any, fallback_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Convert any agent result to a dict.
        
        Args:
            result: Agent result in any format
            fallback_state: State to return if conversion fails
            
        Returns:
            Dictionary containing state updates
            
        Note:
            This method is defensive and will ALWAYS return a dict,
            preventing downstream crashes from type mismatches.
        """
        try:
            # Already a dict - return as is
            if isinstance(result, dict):
                return result
            
            # LangGraph Command object
            from langgraph.types import Command
            if isinstance(result, Command):
                if result.update is not None:
                    if isinstance(result.update, dict):
                        return result.update
                    elif hasattr(result.update, '__dict__'):
                        # Update is an object - convert to dict
                        return {k: v for k, v in result.update.__dict__.items() 
                               if not k.startswith('_')}
                return fallback_state or {}
            
            # ValidatedCommand wrapper
            if hasattr(result, 'as_command'):
                try:
                    command = result.as_command()
                    return CommandConverter.to_dict(command, fallback_state)
                except Exception as e:
                    logger.debug(f"Failed to extract command from wrapper: {e}")
            
            # Object with __dict__ (dataclasses, custom classes)
            if hasattr(result, '__dict__') and not callable(result):
                obj_dict = {}
                for k, v in result.__dict__.items():
                    if not k.startswith('_'):
                        # Skip private attributes
                        obj_dict[k] = v
                return obj_dict if obj_dict else fallback_state or {}
            
            # Command-like object with update attribute
            if hasattr(result, 'update'):
                update_val = getattr(result, 'update', None)
                
                # update is a method - call it
                if callable(update_val):
                    try:
                        update_result = update_val()
                        if isinstance(update_result, dict):
                            return update_result
                    except Exception as e:
                        logger.debug(f"Failed to call update method: {e}")
                
                # update is a property/attribute
                elif update_val is not None:
                    if isinstance(update_val, dict):
                        return update_val
                    elif hasattr(update_val, '__dict__'):
                        return {k: v for k, v in update_val.__dict__.items() 
                               if not k.startswith('_')}
            
            # Last resort - check for to_dict method
            if hasattr(result, 'to_dict') and callable(result.to_dict):
                try:
                    dict_result = result.to_dict()
                    if isinstance(dict_result, dict):
                        return dict_result
                except Exception as e:
                    logger.debug(f"Failed to call to_dict method: {e}")
            
            # Unable to convert - return fallback
            logger.debug(f"Unable to convert {type(result).__name__} to dict, using fallback")
            return fallback_state or {}
            
        except Exception as e:
            logger.error(f"Error converting result to dict: {e}")
            return fallback_state or {}
    
    @staticmethod
    def extract_goto(result: Any) -> Optional[str]:
        """
        Extract goto destination from any result type.
        
        Args:
            result: Agent result in any format
            
        Returns:
            Next node name or None
        """
        try:
            # LangGraph Command object
            from langgraph.types import Command
            if isinstance(result, Command):
                return result.goto
            
            # ValidatedCommand wrapper
            if hasattr(result, 'as_command'):
                try:
                    command = result.as_command()
                    return CommandConverter.extract_goto(command)
                except:
                    pass
            
            # Object with goto attribute
            if hasattr(result, 'goto'):
                goto_val = getattr(result, 'goto', None)
                if isinstance(goto_val, str):
                    return goto_val
                elif callable(goto_val):
                    try:
                        goto_result = goto_val()
                        if isinstance(goto_result, str):
                            return goto_result
                    except:
                        pass
            
            # Check for __goto__ in dict
            if isinstance(result, dict):
                return result.get('__goto__') or result.get('goto')
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting goto: {e}")
            return None
    
    @staticmethod
    def has_update(result: Any) -> bool:
        """
        Check if result has meaningful update data.
        
        Args:
            result: Agent result in any format
            
        Returns:
            True if result contains update data
        """
        update_dict = CommandConverter.to_dict(result)
        return bool(update_dict)
    
    @staticmethod
    def merge_with_state(
        result: Any, 
        current_state: Dict[str, Any],
        preserve_fields: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Merge result with current state, preserving critical fields.
        
        Args:
            result: Agent result in any format
            current_state: Current workflow state
            preserve_fields: Fields to preserve from current state
            
        Returns:
            Merged state dictionary
        """
        # Convert result to dict
        update_dict = CommandConverter.to_dict(result, {})
        
        # Start with current state
        merged = current_state.copy()
        
        # Apply updates
        merged.update(update_dict)
        
        # Preserve critical fields if requested
        if preserve_fields:
            for field in preserve_fields:
                if field in current_state and field not in update_dict:
                    merged[field] = current_state[field]
        
        return merged