"""
State validation and propagation ensuring data integrity between agents.

This module provides validation and ensures critical data is properly
propagated between agents without loss during transitions.
"""

import logging
import time
from typing import Dict, Any, Type, Optional, List
from copy import deepcopy

logger = logging.getLogger(__name__)


class StateValidator:
    """Validates and ensures proper state propagation between agents."""
    
    # Critical fields that MUST be preserved during transitions
    CRITICAL_FIELDS = {
        'observations',
        'citations', 
        'search_results',
        'current_plan',
        'research_topic'
    }
    
    # Required fields for each agent
    REQUIRED_FIELDS = {
        'coordinator': ['research_topic', 'messages'],
        'planner': ['research_topic'],
        'researcher': ['current_plan', 'research_topic'],
        'fact_checker': ['observations', 'current_plan', 'research_topic'],
        'reporter': ['observations', 'current_plan', 'research_topic']
    }
    
    # Fields that accumulate rather than replace
    ACCUMULATING_FIELDS = {
        'observations',
        'citations',
        'search_results', 
        'agent_handoffs',
        'errors',
        'warnings',
        'reflections'
    }
    
    @classmethod
    def validate_state_for_agent(cls, agent_name: str, state: Dict[str, Any]) -> bool:
        """
        Validate that state has required fields for an agent.
        
        Args:
            agent_name: Name of the agent
            state: Current state dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required fields are missing
        """
        required = cls.REQUIRED_FIELDS.get(agent_name, [])
        missing = []
        empty = []
        
        for field in required:
            if field not in state:
                missing.append(field)
            elif state[field] is None:
                empty.append(field)
        
        if missing:
            available_keys = list(state.keys())
            raise ValueError(
                f"Agent '{agent_name}' missing required fields: {missing}. "
                f"Available keys: {available_keys}"
            )
        
        if empty:
            logger.warning(f"Agent '{agent_name}' has empty required fields: {empty}")
        
        # Log state summary for debugging
        cls._log_state_summary(state, f"validated for {agent_name}")
        return True
    
    @classmethod
    def merge_command_update(cls, state: Dict[str, Any], command_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Properly merge command updates into state, preserving critical fields.
        
        Args:
            state: Current state
            command_update: Updates from Command
            
        Returns:
            Merged state with preserved critical data
        """
        if not command_update:
            logger.warning("Empty command update - no changes to apply")
            return state
        
        # Create a deep copy to avoid mutations
        new_state = deepcopy(state)
        
        # Track what we're updating
        logger.info(f"Merging command update: {list(command_update.keys())}")
        
        for key, value in command_update.items():
            if key in cls.ACCUMULATING_FIELDS and key in new_state:
                # Merge lists carefully
                merged = cls._merge_lists(new_state[key], value)
                new_state[key] = merged
                logger.info(f"Accumulated {key}: {len(new_state.get(key, []))} -> {len(merged)} items")
            else:
                # Direct update
                old_value = new_state.get(key)
                new_state[key] = value
                
                # Log critical updates
                if key in cls.CRITICAL_FIELDS:
                    old_count = len(old_value) if hasattr(old_value, '__len__') else 'N/A'
                    new_count = len(value) if hasattr(value, '__len__') else 'N/A'
                    logger.info(f"Updated critical field '{key}': {old_count} -> {new_count}")
        
        # Verify critical data wasn't lost
        cls._verify_critical_data_preservation(state, new_state)
        
        return new_state
    
    @classmethod 
    def _merge_lists(cls, existing: Any, new: Any) -> List[Any]:
        """Merge two lists, handling various input types."""
        # Convert to lists if not already
        existing_list = existing if isinstance(existing, list) else ([] if existing is None else [existing])
        new_list = new if isinstance(new, list) else ([] if new is None else [new])
        
        # Simple concatenation - could add deduplication if needed
        merged = existing_list + new_list
        
        # Limit size to prevent memory issues (keep last N items)
        MAX_ITEMS = 1000
        if len(merged) > MAX_ITEMS:
            merged = merged[-MAX_ITEMS:]
            logger.warning(f"Truncated list to {MAX_ITEMS} items to prevent memory issues")
        
        return merged
    
    @classmethod
    def _verify_critical_data_preservation(cls, old_state: Dict[str, Any], new_state: Dict[str, Any]):
        """Verify that critical data wasn't lost during merge."""
        for field in cls.CRITICAL_FIELDS:
            old_value = old_state.get(field)
            new_value = new_state.get(field)
            
            # Check for data loss
            if old_value and not new_value:
                logger.error(f"CRITICAL DATA LOSS: {field} was lost during state merge!")
                logger.error(f"  Old: {type(old_value)} with {len(old_value) if hasattr(old_value, '__len__') else 'N/A'} items")
                logger.error(f"  New: {new_value}")
                
                # Attempt recovery
                new_state[field] = old_value
                logger.info(f"Recovered {field} from old state")
            
            # Check for unexpected shrinkage (data partially lost)
            elif (hasattr(old_value, '__len__') and hasattr(new_value, '__len__') and 
                  len(old_value) > len(new_value) and len(old_value) > 0):
                logger.warning(f"Data shrinkage in {field}: {len(old_value)} -> {len(new_value)}")
    
    @classmethod
    def _log_state_summary(cls, state: Dict[str, Any], context: str = ""):
        """Log a summary of the current state for debugging."""
        logger.info(f"=== State Summary {context} ===")
        logger.info(f"Total keys: {len(state)}")
        logger.info(f"Keys present: {sorted(list(state.keys()))}")
        
        # Log critical field counts
        for field in cls.CRITICAL_FIELDS:
            value = state.get(field)
            if hasattr(value, '__len__'):
                logger.info(f"{field}: {len(value)} items")
            elif value is not None:
                logger.info(f"{field}: present ({type(value).__name__})")
            else:
                logger.info(f"{field}: missing")
        
        # Log workflow progress
        logger.info(f"Current agent: {state.get('current_agent', 'unknown')}")
        logger.info(f"Total steps: {state.get('total_workflow_steps', 0)}")
        logger.info("=" * 50)
    
    @classmethod
    def ensure_critical_fields_present(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure critical fields are present with default values."""
        defaults = {
            'observations': [],
            'citations': [],
            'search_results': [],
            'agent_handoffs': [],
            'errors': [],
            'warnings': [],
            'total_workflow_steps': 0,
            'research_loops': 0,
            'fact_check_loops': 0
        }
        
        updated = False
        for field, default_value in defaults.items():
            if field not in state or state[field] is None:
                state[field] = default_value
                updated = True
                logger.info(f"Initialized missing field '{field}' with default value")
        
        if updated:
            logger.info("Added missing critical fields to state")
        
        return state


class StatePropagationTracker:
    """
    Tracks data flow between agents for debugging.
    """
    
    def __init__(self):
        self.transitions = []
        self.data_loss_events = []
    
    def record_transition(self, from_agent: str, to_agent: str, state_snapshot: Dict[str, Any]):
        """Record a state transition between agents."""
        transition = {
            'from': from_agent,
            'to': to_agent,
            'timestamp': time.time(),
            'state_keys': sorted(list(state_snapshot.keys())),
            'observations_count': len(state_snapshot.get('observations', [])),
            'citations_count': len(state_snapshot.get('citations', [])),
            'search_results_count': len(state_snapshot.get('search_results', [])),
            'has_plan': 'current_plan' in state_snapshot and state_snapshot['current_plan'] is not None,
            'research_topic': state_snapshot.get('research_topic', 'missing')
        }
        
        self.transitions.append(transition)
        
        # Check for data loss compared to previous transition
        if len(self.transitions) > 1:
            self._detect_data_loss(self.transitions[-2], transition)
        
        logger.debug(f"Recorded transition: {from_agent} -> {to_agent}")
    
    def _detect_data_loss(self, prev_transition: Dict, curr_transition: Dict):
        """Detect potential data loss between transitions."""
        # Check for significant drops in data counts
        fields_to_check = ['observations_count', 'citations_count', 'search_results_count']
        
        for field in fields_to_check:
            prev_count = prev_transition.get(field, 0)
            curr_count = curr_transition.get(field, 0)
            
            # Flag significant drops (more than 50% loss)
            if prev_count > 0 and curr_count < prev_count * 0.5:
                data_loss_event = {
                    'field': field,
                    'from_agent': prev_transition['from'],
                    'to_agent': curr_transition['to'],
                    'prev_count': prev_count,
                    'curr_count': curr_count,
                    'loss_percentage': (prev_count - curr_count) / prev_count * 100,
                    'timestamp': curr_transition['timestamp']
                }
                
                self.data_loss_events.append(data_loss_event)
                logger.warning(f"Data loss detected: {field} dropped from {prev_count} to {curr_count} "
                             f"({data_loss_event['loss_percentage']:.1f}% loss) "
                             f"during {prev_transition['from']} -> {curr_transition['to']}")
    
    def diagnose_data_loss(self) -> Dict[str, Any]:
        """Generate a diagnostic report of data flow."""
        if not self.transitions:
            return {"status": "no_transitions_recorded"}
        
        report = {
            "total_transitions": len(self.transitions),
            "data_loss_events": len(self.data_loss_events),
            "transitions": self.transitions,
            "data_loss_details": self.data_loss_events
        }
        
        # Log summary
        logger.info(f"=== Data Flow Diagnostic Report ===")
        logger.info(f"Total transitions: {len(self.transitions)}")
        logger.info(f"Data loss events: {len(self.data_loss_events)}")
        
        for i, transition in enumerate(self.transitions):
            logger.info(f"Transition {i+1}: {transition['from']} -> {transition['to']}")
            logger.info(f"  Observations: {transition['observations_count']}")
            logger.info(f"  Citations: {transition['citations_count']}")
            logger.info(f"  Has plan: {transition['has_plan']}")
            logger.info(f"  Topic: {transition['research_topic']}")
        
        if self.data_loss_events:
            logger.warning("Data loss events detected:")
            for event in self.data_loss_events:
                logger.warning(f"  {event['field']}: {event['prev_count']} -> {event['curr_count']} "
                             f"({event['loss_percentage']:.1f}% loss)")
        
        logger.info("=" * 40)
        
        return report
    
    def get_last_transition(self) -> Optional[Dict[str, Any]]:
        """Get the most recent transition."""
        return self.transitions[-1] if self.transitions else None


# Global tracker instance for debugging
global_propagation_tracker = StatePropagationTracker()