"""
ValidatedCommand - A Command wrapper that preserves critical state data.

This module provides a Command wrapper that ensures critical research data
is not lost during state transitions between agents.
"""

import logging
import time
from typing import Dict, Any, Optional
from langgraph.types import Command

from .state_validator import StateValidator
from .agent_contracts import AgentDataContract

logger = logging.getLogger(__name__)


class ValidatedCommand:
    """
    Enhanced Command wrapper that validates and preserves critical state data.
    
    This wrapper ensures that agent outputs properly integrate with existing state
    while preserving critical fields like observations, citations, and search results.
    """
    
    def __init__(
        self,
        command: Command = None,
        update: Optional[Dict[str, Any]] = None,
        goto: Optional[str] = None,
        agent_name: Optional[str] = None,
        preserve_critical: bool = True
    ):
        """
        Initialize ValidatedCommand.
        
        Args:
            command: Base Command to wrap (optional)
            update: State updates to apply
            goto: Next node to visit
            agent_name: Name of the agent producing this command
            preserve_critical: Whether to preserve critical fields
        """
        # Create internal Command object
        if command:
            self._command = Command(update=command.update, goto=command.goto)
        else:
            self._command = Command(update=update, goto=goto)
        
        self.agent_name = agent_name
        self.preserve_critical = preserve_critical
        self._original_update = self._command.update.copy() if self._command.update else {}
        
        logger.info(f"Created ValidatedCommand for {agent_name} with {len(self._original_update)} update fields")
    
    @property
    def update(self) -> Optional[Dict[str, Any]]:
        """Get the update dictionary."""
        return self._command.update
    
    @property
    def goto(self) -> Optional[str]:
        """Get the goto destination."""
        return self._command.goto
    
    def as_command(self) -> Command:
        """Return the internal Command object."""
        return self._command
    
    @classmethod
    def from_agent_output(
        cls,
        agent_name: str,
        agent_output: Any,
        current_state: Dict[str, Any],
        next_node: Optional[str] = None
    ) -> 'ValidatedCommand':
        """
        Create ValidatedCommand from agent output.
        
        Args:
            agent_name: Name of the agent
            agent_output: Output from the agent (can be dict, object with to_dict, etc.)
            current_state: Current workflow state
            next_node: Optional next node to visit
            
        Returns:
            ValidatedCommand with properly merged state
        """
        logger.info(f"Creating ValidatedCommand from {agent_name} output")
        
        # Validate and convert agent output to dict - NO FALLBACKS
        output_dict = AgentDataContract.validate_output(agent_name, agent_output)
        logger.info(f"Agent output validation passed for {agent_name}")
        
        # Merge with state preservation
        merged_state = StateValidator.merge_command_update(current_state, output_dict)
        
        # Calculate actual updates (what changed)
        updates = {}
        for key, value in output_dict.items():
            updates[key] = value
        
        # Add agent tracking
        updates['current_agent'] = agent_name
        updates[f'last_{agent_name}_output'] = output_dict
        
        logger.info(f"ValidatedCommand created with {len(updates)} updates for {agent_name}")
        
        return cls(
            update=updates,
            goto=next_node,
            agent_name=agent_name,
            preserve_critical=True
        )
    
    @classmethod
    def create_routing_command(
        cls,
        next_node: str,
        current_state: Dict[str, Any],
        routing_reason: str = "standard_routing"
    ) -> 'ValidatedCommand':
        """
        Create a command for routing without data updates.
        
        Args:
            next_node: Next node to visit
            current_state: Current state (for validation)
            routing_reason: Reason for this routing decision
            
        Returns:
            ValidatedCommand for routing
        """
        logger.info(f"Creating routing command to {next_node}: {routing_reason}")
        
        # Ensure critical fields are present
        validated_state = StateValidator.ensure_critical_fields_present(current_state.copy())
        
        # Add routing metadata
        routing_update = {
            'last_routing_decision': next_node,
            'routing_reason': routing_reason,
            'routing_timestamp': time.time()
        }
        
        return cls(
            update=routing_update,
            goto=next_node,
            agent_name="router",
            preserve_critical=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'update': self.update,
            'goto': self.goto,
            'agent_name': self.agent_name,
            'preserve_critical': self.preserve_critical,
            'original_update': self._original_update
        }
    
    def validate_against_state(self, current_state: Dict[str, Any]) -> bool:
        """
        Validate this command against current state.
        
        Args:
            current_state: Current workflow state
            
        Returns:
            True if command is valid
            
        Raises:
            ValueError: If validation fails
        """
        if not self.update:
            return True
            
        # Check that we're not losing critical data
        merged_state = StateValidator.merge_command_update(current_state, self.update)
        
        # Verify critical data preservation
        StateValidator._verify_critical_data_preservation(current_state, merged_state)
        
        return True
    
    def __repr__(self) -> str:
        return f"ValidatedCommand(agent={self.agent_name}, updates={len(self.update or {})}, goto={self.goto})"