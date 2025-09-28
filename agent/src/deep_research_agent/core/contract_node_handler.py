"""
Contract Node Handler - Unified handling for contract-based agents in workflow nodes.

This module eliminates code duplication by providing a single function
that handles all contract validation and state merging for workflow nodes.
"""

import logging
from typing import Dict, Any, Callable

from .state_validator import StateValidator
from .agent_contracts import AgentDataContract

logger = logging.getLogger(__name__)


def execute_contract_agent(
    agent: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    agent_name: str,
    state: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute a contract-based agent with strict validation and state merging.
    
    This function replaces all the duplicate validation/merge code in workflow nodes.
    It enforces strict contract compliance with no fallbacks.
    
    Args:
        agent: Agent function/object to execute
        agent_name: Name of the agent (for logging/validation)
        state: Current workflow state
        config: Optional configuration dictionary
        
    Returns:
        Updated state dictionary with agent output merged
        
    Raises:
        ValueError: If any contract validation fails
    """
    logger.info(f"=== EXECUTING {agent_name.upper()} WITH CONTRACT ENFORCEMENT ===")
    
    # Step 1: Execute agent with strict contract enforcement
    try:
        agent_output = agent(state, config or {})
        logger.info(f"{agent_name} execution completed successfully")
    except Exception as e:
        logger.error(f"{agent_name} execution failed: {e}")
        raise ValueError(f"Agent {agent_name} execution failed: {e}")
    
    # Step 2: Validate agent output using contracts - NO FALLBACKS
    try:
        validated_output_dict = AgentDataContract.validate_output(agent_name.lower(), agent_output)
        logger.info(f"{agent_name} output contract validation passed")
    except Exception as e:
        logger.error(f"{agent_name} output validation FAILED: {e}")
        raise ValueError(f"Contract violation in {agent_name} output: {e}")
    
    # Step 3: Merge validated output with current state - NO FALLBACKS
    try:
        updated_state = StateValidator.merge_command_update(state, validated_output_dict)
        logger.info(f"{agent_name} state merge completed")
        
        # Record which agent last updated the state
        updated_state['current_agent'] = agent_name
        updated_state[f'last_{agent_name}_output'] = validated_output_dict
        
        return updated_state
        
    except Exception as e:
        logger.error(f"{agent_name} state merge FAILED: {e}")
        raise ValueError(f"State merge failed for {agent_name}: {e}")


def execute_contract_agent_with_circuit_breaker(
    agent: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    agent_name: str,
    state: Dict[str, Any],
    config: Dict[str, Any] = None,
    circuit_breaker_fn: Callable[[Dict[str, Any]], tuple] = None
) -> Dict[str, Any]:
    """
    Execute contract agent with circuit breaker support for workflow control.
    
    This extends the basic contract execution with circuit breaker logic
    that's common across workflow nodes.
    
    Args:
        agent: Agent function/object to execute
        agent_name: Name of the agent
        state: Current workflow state
        config: Optional configuration
        circuit_breaker_fn: Function that returns (should_terminate, reason, explanation)
        
    Returns:
        Updated state dictionary
        
    Raises:
        ValueError: If contract validation fails
    """
    logger.info(f"=== {agent_name.upper()} NODE WITH CIRCUIT BREAKER ===")
    
    # Step 1: Increment workflow step counter
    total_steps = state.get("total_workflow_steps", 0)
    state["total_workflow_steps"] = total_steps + 1
    
    # Step 2: Check circuit breaker if provided
    if circuit_breaker_fn:
        should_terminate, reason, explanation = circuit_breaker_fn(state)
        
        if should_terminate:
            logger.warning(
                f"Circuit breaker activated in {agent_name}_node - workflow terminating",
                extra={
                    "reason": reason.value if hasattr(reason, 'value') else str(reason),
                    "explanation": explanation,
                    "total_steps": total_steps
                }
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason}: {explanation}"
            )
            
            return state
    
    # Step 3: Execute agent with contract enforcement
    return execute_contract_agent(agent, agent_name, state, config)


def create_contract_workflow_node(
    agent: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    agent_name: str,
    circuit_breaker_fn: Callable[[Dict[str, Any]], tuple] = None,
    state_preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
    state_postprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] = None
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Factory function to create standardized workflow nodes for contract agents.
    
    This eliminates all the boilerplate code in workflow_nodes_enhanced.py by
    generating standardized node functions.
    
    Args:
        agent: Agent to execute
        agent_name: Name of the agent
        circuit_breaker_fn: Optional circuit breaker function
        state_preprocessor: Optional function to preprocess state before agent
        state_postprocessor: Optional function to postprocess state after agent
        
    Returns:
        Workflow node function that can be added to StateGraph
    """
    def workflow_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generated workflow node with contract enforcement."""
        logger.info(f"Executing {agent_name} workflow node")
        
        # Optional: Preprocess state
        if state_preprocessor:
            state = state_preprocessor(state)
        
        # Execute agent with contract enforcement and circuit breaker
        result_state = execute_contract_agent_with_circuit_breaker(
            agent=agent,
            agent_name=agent_name,
            state=state,
            circuit_breaker_fn=circuit_breaker_fn
        )
        
        # Optional: Postprocess state
        if state_postprocessor:
            result_state = state_postprocessor(result_state)
        
        logger.info(f"Completed {agent_name} workflow node")
        return result_state
    
    # Set function name for debugging
    workflow_node.__name__ = f"{agent_name}_node"
    return workflow_node