"""
ContractAgent - Base class for type-safe agent execution with contracts.

This module provides a base class that enforces strict input/output contracts
for all agents, eliminating code duplication and ensuring type safety.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic, Type

from .agent_contracts import AgentDataContract, AgentInputOutput

logger = logging.getLogger(__name__)

# Type variables for generic input/output contracts
InputContract = TypeVar('InputContract', bound=AgentInputOutput)
OutputContract = TypeVar('OutputContract', bound=AgentInputOutput)


class ContractAgent(ABC, Generic[InputContract, OutputContract]):
    """
    Base class for all agents that enforces strict input/output contracts.
    
    This eliminates code duplication across workflow nodes and ensures
    type safety at every agent boundary with no fallbacks.
    """
    
    def __init__(
        self,
        agent_name: str,
        input_contract_class: Type[InputContract],
        output_contract_class: Type[OutputContract],
        llm=None,
        config=None,
        event_emitter=None,
        **kwargs
    ):
        """
        Initialize the contract agent.
        
        Args:
            agent_name: Name of the agent (for logging and validation)
            input_contract_class: Class for input validation
            output_contract_class: Class for output validation
            llm: Language model instance
            config: Configuration dictionary
            event_emitter: Optional event emitter
            **kwargs: Additional agent-specific parameters
        """
        self.agent_name = agent_name
        self.input_contract_class = input_contract_class
        self.output_contract_class = output_contract_class
        self.llm = llm
        self.config = config or {}
        self.event_emitter = event_emitter
        
        # Store additional kwargs for agent-specific parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        logger.info(f"Initialized {agent_name} with strict contracts")
    
    def __call__(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> OutputContract:
        """
        Execute the agent with strict contract enforcement.
        
        Args:
            state: Current workflow state
            config: Optional configuration override
            
        Returns:
            Validated output contract object
            
        Raises:
            ValueError: If input or output validation fails
        """
        logger.info(f"=== {self.agent_name.upper()} AGENT STARTING ===")
        
        # Step 1: Strict input validation - NO FALLBACKS
        try:
            validated_input = AgentDataContract.validate_input(self.agent_name.lower(), state)
            logger.info(f"{self.agent_name} input validation passed")
        except Exception as e:
            logger.error(f"{self.agent_name} input validation FAILED: {e}")
            raise ValueError(f"Contract violation in {self.agent_name} input: {e}")
        
        # Step 2: Execute agent logic
        try:
            raw_output = self.execute(validated_input, config or self.config)
            logger.info(f"{self.agent_name} execution completed")
        except Exception as e:
            logger.error(f"{self.agent_name} execution FAILED: {e}")
            raise ValueError(f"Agent execution failed in {self.agent_name}: {e}")
        
        # Step 3: Strict output validation - NO FALLBACKS
        try:
            # Ensure output is a contract object
            if not isinstance(raw_output, self.output_contract_class):
                logger.error(f"{self.agent_name} returned {type(raw_output)}, expected {self.output_contract_class}")
                raise ValueError(f"Agent {self.agent_name} must return {self.output_contract_class.__name__}")
            
            # Validate the contract
            raw_output.validate()
            logger.info(f"{self.agent_name} output validation passed")
            
        except Exception as e:
            logger.error(f"{self.agent_name} output validation FAILED: {e}")
            raise ValueError(f"Contract violation in {self.agent_name} output: {e}")
        
        logger.info(f"=== {self.agent_name.upper()} AGENT COMPLETED ===")
        return raw_output
    
    @abstractmethod
    def execute(self, validated_input: InputContract, config: Dict[str, Any]) -> OutputContract:
        """
        Execute the agent's core logic.
        
        Args:
            validated_input: Pre-validated input contract
            config: Configuration dictionary
            
        Returns:
            Output contract object (will be validated after return)
        """
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent for debugging."""
        return {
            "name": self.agent_name,
            "input_contract": self.input_contract_class.__name__,
            "output_contract": self.output_contract_class.__name__,
            "has_llm": self.llm is not None,
            "has_event_emitter": self.event_emitter is not None
        }