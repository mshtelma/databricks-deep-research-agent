"""
High-level response handling helpers for agents.

Provides convenient methods for agents to handle LLM responses
with automatic reasoning event emission and content extraction.
"""

from typing import Any, Optional, Dict, List
from . import get_logger
from .response_handlers import parse_structured_response, ParsedResponse

logger = get_logger(__name__)


class AgentResponseHandler:
    """High-level response handler for agent use cases."""
    
    def __init__(self, agent_name: str, event_emitter=None):
        self.agent_name = agent_name
        self.event_emitter = event_emitter
    
    def process_llm_response(
        self, 
        response: Any, 
        operation_name: str,
        emit_reasoning_events: bool = True,
        requested_entities: Optional[List[str]] = None
    ) -> str:
        """
        Process LLM response with automatic reasoning event emission.
        
        Args:
            response: Raw LLM response
            operation_name: Name of the operation (for logging/events)
            emit_reasoning_events: Whether to emit reasoning events
            requested_entities: List of entities for validation
            
        Returns:
            Extracted content string
        """
        # Parse the structured response
        parsed = parse_structured_response(response)
        
        # Log response details
        logger.info(f"🔍 LLM_RESPONSE [{self.agent_name}_{operation_name}]: "
                   f"{parsed.content[:500]}... "
                   f"(type: {parsed.response_type.value}, "
                   f"has_reasoning: {parsed.reasoning is not None})")
        
        # Emit reasoning event if we have reasoning and an event emitter
        if (emit_reasoning_events and 
            parsed.reasoning and 
            self.event_emitter and 
            hasattr(self.event_emitter, 'emit_reasoning_reflection')):
            
            try:
                self.event_emitter.emit_reasoning_reflection(
                    reasoning=parsed.reasoning[:500],  # Truncate for event
                    options=["content_generation"],
                    confidence=0.8,
                    stage_id=self.agent_name
                )
                logger.info(f"Emitted reasoning event for {operation_name}")
            except Exception as e:
                logger.warning(f"Failed to emit reasoning event for {operation_name}: {e}")
        
        # Entity validation if requested
        if requested_entities:
            self._validate_entities(parsed.content, requested_entities, operation_name)
        
        return parsed.content
    
    def extract_content_and_reasoning(self, response: Any) -> tuple[str, Optional[str]]:
        """
        Extract both content and reasoning without side effects.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (content, reasoning)
        """
        parsed = parse_structured_response(response)
        return parsed.content, parsed.reasoning
    
    def _validate_entities(self, content: str, requested_entities: List[str], operation_name: str):
        """Validate entities in response content."""
        try:
            from .entity_validation import EntityExtractor
            extractor = EntityExtractor()
            response_entities = extractor.extract_entities(content)
            hallucinated = response_entities - set(requested_entities)
            
            if hallucinated:
                logger.warning(f"🚨 ENTITY_HALLUCINATION [{self.agent_name}_{operation_name}]: "
                             f"LLM mentioned entities not in original query: {hallucinated}")
            else:
                logger.info(f"✅ ENTITY_VALIDATION [{self.agent_name}_{operation_name}]: "
                           f"Response only mentions requested entities: "
                           f"{response_entities & set(requested_entities)}")
        except Exception as e:
            logger.warning(f"Entity validation failed for {operation_name}: {e}")


def create_agent_response_handler(agent_name: str, event_emitter=None) -> AgentResponseHandler:
    """
    Factory function to create an agent response handler.
    
    Args:
        agent_name: Name of the agent (for logging)
        event_emitter: Event emitter instance for reasoning events
        
    Returns:
        Configured AgentResponseHandler
    """
    return AgentResponseHandler(agent_name, event_emitter)


# Convenient decorator for agent methods
def handle_llm_response(operation_name: str, emit_reasoning: bool = True):
    """
    Decorator to automatically handle LLM responses in agent methods.
    
    Usage:
        @handle_llm_response("synthesis")
        def synthesize_findings(self, response):
            # response is already processed
            return response
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get the LLM response (assume it's the first arg or 'response' kwarg)
            response = args[0] if args else kwargs.get('response')
            
            if hasattr(self, '_response_handler'):
                processed_content = self._response_handler.process_llm_response(
                    response, operation_name, emit_reasoning
                )
                # Replace the response with processed content
                if args:
                    args = (processed_content,) + args[1:]
                else:
                    kwargs['response'] = processed_content
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for quick usage
def extract_with_reasoning_events(
    response: Any, 
    agent_name: str, 
    operation_name: str,
    event_emitter=None,
    emit_reasoning: bool = True
) -> str:
    """
    Quick function to extract content with automatic reasoning event emission.
    
    Args:
        response: LLM response
        agent_name: Name of the calling agent
        operation_name: Name of the operation
        event_emitter: Event emitter instance
        emit_reasoning: Whether to emit reasoning events
        
    Returns:
        Extracted content
    """
    handler = AgentResponseHandler(agent_name, event_emitter)
    return handler.process_llm_response(response, operation_name, emit_reasoning)


def get_model_specific_handler(model_name: str, agent_name: str, event_emitter=None) -> AgentResponseHandler:
    """
    Get an optimized handler for specific model types.
    
    Args:
        model_name: Name/type of the model
        agent_name: Name of the calling agent  
        event_emitter: Event emitter instance
        
    Returns:
        Optimized AgentResponseHandler
    """
    handler = AgentResponseHandler(agent_name, event_emitter)
    
    # Could add model-specific optimizations here
    if "databricks" in model_name.lower():
        logger.debug(f"Optimizing handler for Databricks model: {model_name}")
        # Any Databricks-specific optimizations would go here
    
    return handler
