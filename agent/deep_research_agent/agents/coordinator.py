"""
Coordinator Agent: Entry point and traffic controller for research requests.

Based on deer-flow's coordinator pattern for request classification and routing.
"""

from typing import Dict, Any, Optional, Literal
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from deep_research_agent.core import get_logger
from deep_research_agent.core.multi_agent_state import EnhancedResearchState, StateManager


logger = get_logger(__name__)


class RequestType(str, Enum):
    """Types of requests the coordinator can handle."""
    GREETING = "greeting"
    RESEARCH = "research"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    INAPPROPRIATE = "inappropriate"
    UNKNOWN = "unknown"


class CoordinatorAgent:
    """
    Coordinator agent that classifies and routes incoming requests.
    
    Responsibilities:
    - Greet users and handle small talk
    - Classify incoming requests
    - Route complex queries to appropriate agents
    - Reject inappropriate requests
    - Handle clarifications
    """
    
    def __init__(self, llm=None, config=None):
        """Initialize the coordinator agent."""
        self.llm = llm
        self.config = config or {}
        self.name = "Coordinator"  # Capital for test compatibility
        self.description = "Routes research requests to appropriate agents"
    
    def __call__(
        self, 
        state: EnhancedResearchState, 
        config: Dict[str, Any]
    ) -> Command[Literal["planner", "background_investigator", "end"]]:
        """
        Process incoming request and route to appropriate agent.
        
        Args:
            state: Current research state
            config: Configuration dictionary
            
        Returns:
            Command directing to next agent or end
        """
        logger.info("Coordinator agent processing request")
        
        # Get the latest user message
        user_message = self._get_latest_user_message(state)
        if not user_message:
            logger.warning("No user message found in state")
            return Command(goto="end")
        
        # Classify the request
        request_type = self._classify_request(user_message)
        logger.info(f"Request classified as: {request_type}")
        
        # Route based on request type
        if request_type == RequestType.GREETING:
            response = self._handle_greeting(user_message)
            return self._respond_and_end(state, response)
        
        elif request_type == RequestType.INAPPROPRIATE:
            response = self._handle_inappropriate(user_message)
            return self._respond_and_end(state, response)
        
        elif request_type == RequestType.CLARIFICATION:
            response = self._request_clarification(user_message)
            return self._respond_and_end(state, response)
        
        elif request_type == RequestType.RESEARCH:
            # Extract research topic
            research_topic = self._extract_research_topic(user_message)
            
            # Record handoff
            state = StateManager.record_handoff(
                state,
                from_agent=self.name,
                to_agent="planner" if not state["enable_background_investigation"] else "background_investigator",
                reason="Research request identified",
                context={"research_topic": research_topic}
            )
            
            # Update state with research topic
            state["research_topic"] = research_topic
            
            # Route to background investigation or planner
            if state.get("enable_background_investigation", True):
                logger.info("Routing to background investigation")
                return Command(
                    goto="background_investigator",
                    update={"research_topic": research_topic}
                )
            else:
                logger.info("Routing directly to planner")
                return Command(
                    goto="planner",
                    update={"research_topic": research_topic}
                )
        
        else:
            # Unknown request type
            response = self._handle_unknown(user_message)
            return self._respond_and_end(state, response)
    
    def _get_latest_user_message(self, state: EnhancedResearchState) -> Optional[str]:
        """Extract the latest user message from state."""
        messages = state.get("messages", [])
        
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
            elif isinstance(message, dict) and message.get("role") == "user":
                return message.get("content")
        
        return None
    
    def _classify_request(self, message: str) -> RequestType:
        """
        Classify the type of request.
        
        Simple classification based on keywords and patterns.
        Could be enhanced with LLM-based classification.
        """
        message_lower = message.lower().strip()
        
        # Check for greetings
        greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "howdy", "what's up"
        ]
        if any(pattern in message_lower for pattern in greeting_patterns) and len(message_lower) < 20:
            return RequestType.GREETING
        
        # Check for inappropriate content
        inappropriate_patterns = [
            "hack", "illegal", "password", "credential", "malware",
            "exploit", "vulnerability attack", "ddos"
        ]
        if any(pattern in message_lower for pattern in inappropriate_patterns):
            return RequestType.INAPPROPRIATE
        
        # Check for clarification requests
        clarification_patterns = [
            "what do you mean", "can you explain", "i don't understand",
            "clarify", "confused", "not clear"
        ]
        if any(pattern in message_lower for pattern in clarification_patterns):
            return RequestType.CLARIFICATION
        
        # Check for research indicators
        research_patterns = [
            "research", "analyze", "investigate", "find out", "tell me about",
            "what is", "how does", "explain", "describe", "compare",
            "what are the", "give me information"
        ]
        if any(pattern in message_lower for pattern in research_patterns):
            return RequestType.RESEARCH
        
        # Check if it's a question
        if message.strip().endswith("?"):
            return RequestType.RESEARCH
        
        # Default to unknown
        return RequestType.UNKNOWN
    
    def _extract_research_topic(self, message: str) -> str:
        """Extract the research topic from the message."""
        # Remove common research request prefixes
        prefixes_to_remove = [
            "research", "analyze", "investigate", "find out about",
            "tell me about", "what is", "how does", "explain",
            "give me information about", "i want to know about"
        ]
        
        topic = message.lower()
        for prefix in prefixes_to_remove:
            if topic.startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        # Capitalize and clean up
        topic = topic.strip(" ?.,!").capitalize()
        
        # If topic is too short, use the full message
        if len(topic) < 10:
            topic = message.strip()
        
        return topic
    
    def _handle_greeting(self, message: str) -> str:
        """Handle greeting messages."""
        greetings = [
            "Hello! I'm your research assistant. How can I help you today?",
            "Hi there! I'm ready to help you with your research. What would you like to explore?",
            "Greetings! I can help you research and analyze various topics. What interests you?",
            "Hello! I specialize in comprehensive research and analysis. What topic shall we investigate?"
        ]
        
        # Simple rotation based on message length (for variety)
        index = len(message) % len(greetings)
        return greetings[index]
    
    def _handle_inappropriate(self, message: str) -> str:
        """Handle inappropriate requests."""
        return (
            "I'm designed to assist with legitimate research and analysis tasks. "
            "I cannot help with requests that involve illegal activities, "
            "security exploits, or potentially harmful content. "
            "Please provide a different research topic I can help you with."
        )
    
    def _request_clarification(self, message: str) -> str:
        """Request clarification from the user."""
        return (
            "I'd be happy to help, but I need a bit more clarity on your request. "
            "Could you please rephrase or provide more details about what you'd like me to research? "
            "For example, you might ask: 'Research the latest developments in renewable energy' "
            "or 'Analyze the impact of social media on mental health'."
        )
    
    def _handle_unknown(self, message: str) -> str:
        """Handle unknown request types."""
        return (
            "I'm not quite sure what you're asking for. "
            "I'm a research assistant that can help you investigate topics, "
            "analyze information, and provide comprehensive reports. "
            "Try asking me to research a specific topic or question. "
            "For example: 'What are the current trends in artificial intelligence?'"
        )
    
    def _respond_and_end(self, state: EnhancedResearchState, response: str) -> Command:
        """Add response to state and end the workflow."""
        return Command(
            goto="end",
            update={
                "messages": [AIMessage(content=response, name=self.name)],
                "final_report": response
            }
        )