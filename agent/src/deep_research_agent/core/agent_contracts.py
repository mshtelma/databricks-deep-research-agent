"""
Agent data contracts ensuring proper data flow between agents.

This module defines input/output schemas for each agent to prevent data loss
and ensure proper communication between agents in the multi-agent workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentInputOutput(Protocol):
    """Base protocol for agent I/O validation."""
    
    def validate(self) -> bool:
        """Validate the data structure."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state updates."""
        ...


@dataclass
class CoordinatorInput:
    """Input schema for Coordinator agent."""
    research_topic: str
    messages: List[Any]
    
    def validate(self) -> bool:
        if not self.research_topic:
            raise ValueError("Coordinator requires research_topic")
        if not self.messages:
            raise ValueError("Coordinator requires messages")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_topic": self.research_topic,
            "messages": self.messages
        }


@dataclass  
class CoordinatorOutput:
    """Output schema for Coordinator agent."""
    request_type: str  # "research", "greeting", "inappropriate"
    research_context: Optional[str] = None
    research_topic: str = ""  # Preserve the original topic
    
    def validate(self) -> bool:
        if not self.request_type:
            raise ValueError("Coordinator must output request_type")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_type": self.request_type,
            "research_context": self.research_context,
            "research_topic": self.research_topic
        }


@dataclass
class PlannerInput:
    """Input schema for Planner agent."""
    research_topic: str
    research_context: Optional[str] = None
    background_investigation_results: Optional[List[Dict]] = None
    observations: List[Any] = field(default_factory=list)  # Preserve existing
    
    def validate(self) -> bool:
        if not self.research_topic:
            raise ValueError("Planner requires research_topic")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_topic": self.research_topic,
            "research_context": self.research_context,
            "background_investigation_results": self.background_investigation_results,
            "observations": self.observations
        }


@dataclass
class PlannerOutput:
    """Output schema for Planner agent."""
    current_plan: Any  # ResearchPlan object
    suggested_report_structure: Optional[List[str]] = None
    section_specifications: Optional[List[Any]] = None
    observations: List[Any] = field(default_factory=list)  # Preserve
    
    def validate(self) -> bool:
        if not self.current_plan:
            raise ValueError("Planner must output current_plan")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_plan": self.current_plan,
            "suggested_report_structure": self.suggested_report_structure,
            "section_specifications": self.section_specifications,
            "observations": self.observations
        }


@dataclass
class ResearcherInput:
    """Input schema for Researcher agent."""
    current_plan: Any  # ResearchPlan
    research_topic: str
    current_step: Optional[Any] = None
    current_step_index: int = 0
    observations: List[Any] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    citations: List[Any] = field(default_factory=list)
    
    def validate(self) -> bool:
        if not self.current_plan:
            raise ValueError("Researcher requires current_plan")
        if not hasattr(self.current_plan, 'steps') and not isinstance(self.current_plan, dict):
            raise ValueError("current_plan must have steps attribute or be a dict")
        if not self.research_topic:
            raise ValueError("Researcher requires research_topic")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_plan": self.current_plan,
            "research_topic": self.research_topic,
            "current_step": self.current_step,
            "current_step_index": self.current_step_index,
            "observations": self.observations,
            "search_results": self.search_results,
            "citations": self.citations
        }


@dataclass
class ResearcherOutput:
    """Output schema for Researcher agent."""
    observations: List[Any]
    search_results: List[Dict]
    citations: List[Any]
    completed_steps: List[Any]
    current_plan: Any  # Updated plan
    research_topic: str  # Preserve
    current_step: Optional[Any] = None  # Critical for preventing infinite loops
    current_step_index: int = 0  # Track the step index
    
    def validate(self) -> bool:
        # Observations can be empty but should exist
        if self.observations is None:
            raise ValueError("Researcher must output observations (can be empty list)")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "search_results": self.search_results,
            "citations": self.citations,
            "completed_steps": self.completed_steps,
            "current_plan": self.current_plan,
            "research_topic": self.research_topic,
            "current_step": self.current_step,
            "current_step_index": self.current_step_index
        }


@dataclass
class FactCheckerInput:
    """Input schema for FactChecker agent."""
    observations: List[Any]
    current_plan: Any
    research_topic: str
    citations: List[Any] = field(default_factory=list)
    
    def validate(self) -> bool:
        if not self.research_topic:
            raise ValueError("FactChecker requires research_topic")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "current_plan": self.current_plan,
            "research_topic": self.research_topic,
            "citations": self.citations
        }


@dataclass
class FactCheckerOutput:
    """Output schema for FactChecker agent."""
    observations: List[Any]  # Pass through
    current_plan: Any  # Pass through
    research_topic: str  # Pass through
    citations: List[Any]
    factuality_score: float
    grounding_results: Optional[Dict] = None
    
    def validate(self) -> bool:
        if self.factuality_score < 0 or self.factuality_score > 1:
            raise ValueError("Factuality score must be between 0 and 1")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "current_plan": self.current_plan,
            "research_topic": self.research_topic,
            "citations": self.citations,
            "factuality_score": self.factuality_score,
            "grounding_results": self.grounding_results
        }


@dataclass
class ReporterInput:
    """Input schema for Reporter agent."""
    observations: List[Any]
    current_plan: Any
    research_topic: str
    citations: List[Any] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    factuality_score: Optional[float] = None
    grounding_results: Optional[Dict] = None
    
    def validate(self) -> bool:
        if not self.observations:
            logger.warning("Reporter received no observations - may produce limited report")
        if not self.current_plan:
            raise ValueError("Reporter requires current_plan")
        if not self.research_topic:
            raise ValueError("Reporter requires research_topic")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "current_plan": self.current_plan,
            "research_topic": self.research_topic,
            "citations": self.citations,
            "search_results": self.search_results,
            "factuality_score": self.factuality_score,
            "grounding_results": self.grounding_results
        }


@dataclass
class ReporterOutput:
    """Output schema for Reporter agent."""
    final_report: str
    report_sections: Dict[str, str]
    citations: List[Any]  # Final citations used
    
    def validate(self) -> bool:
        if not self.final_report:
            raise ValueError("Reporter must output final_report")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_report": self.final_report,
            "report_sections": self.report_sections,
            "citations": self.citations
        }


class AgentDataContract:
    """Helper class to validate agent inputs/outputs."""
    
    INPUT_SCHEMAS = {
        'coordinator': CoordinatorInput,
        'planner': PlannerInput,
        'researcher': ResearcherInput,
        'fact_checker': FactCheckerInput,
        'reporter': ReporterInput
    }
    
    OUTPUT_SCHEMAS = {
        'coordinator': CoordinatorOutput,
        'planner': PlannerOutput,
        'researcher': ResearcherOutput,
        'fact_checker': FactCheckerOutput,
        'reporter': ReporterOutput
    }
    
    @classmethod
    def validate_input(cls, agent_name: str, state: Dict[str, Any]) -> Any:
        """
        Validate and create typed input for an agent.
        
        Args:
            agent_name: Name of the agent
            state: Current state
            
        Returns:
            Typed input object
            
        Raises:
            ValueError: If validation fails
        """
        schema = cls.INPUT_SCHEMAS.get(agent_name)
        if not schema:
            logger.warning(f"No input schema defined for agent: {agent_name}")
            return state
        
        # Create input object from state
        try:
            # Extract relevant fields for the schema
            kwargs = {}
            for field_name in schema.__dataclass_fields__.keys():
                if field_name in state:
                    kwargs[field_name] = state[field_name]
            
            input_obj = schema(**kwargs)
            input_obj.validate()
            return input_obj
        except (TypeError, ValueError) as e:
            raise ValueError(f"Input validation failed for {agent_name}: {e}")
    
    @classmethod
    def validate_output(cls, agent_name: str, output: Any) -> Dict[str, Any]:
        """
        Validate agent output and convert to dict.
        
        Args:
            agent_name: Name of the agent
            output: Agent output
            
        Returns:
            Dictionary of output values
            
        Raises:
            ValueError: If validation fails
        """
        schema = cls.OUTPUT_SCHEMAS.get(agent_name)
        if not schema:
            logger.warning(f"No output schema defined for agent: {agent_name}")
            return output if isinstance(output, dict) else {}
        
        if hasattr(output, 'validate'):
            output.validate()
        
        if hasattr(output, 'to_dict'):
            return output.to_dict()
        
        return output if isinstance(output, dict) else {}