"""
Coordinator Agent: Entry point and traffic controller for research requests.

Based on deer-flow's coordinator pattern for request classification and routing.
"""

from typing import Dict, Any, Optional, Literal
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..core import get_logger
from ..core.multi_agent_state import EnhancedResearchState, StateManager
from ..core.message_utils import get_last_user_message
from ..core.agent_contracts import CoordinatorInput, CoordinatorOutput
from ..core.contract_agent import ContractAgent


logger = get_logger(__name__)


class RequestType(str, Enum):
    """Types of requests the coordinator can handle."""
    GREETING = "greeting"
    RESEARCH = "research"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    INAPPROPRIATE = "inappropriate"
    UNKNOWN = "unknown"


class CoordinatorAgent(ContractAgent[CoordinatorInput, CoordinatorOutput]):
    """
    Coordinator agent that classifies and routes incoming requests.
    
    Responsibilities:
    - Greet users and handle small talk
    - Classify incoming requests
    - Route complex queries to appropriate agents
    - Reject inappropriate requests
    - Handle clarifications
    """
    
    def __init__(self, llm=None, config=None, event_emitter=None):
        """Initialize the coordinator agent."""
        super().__init__(
            agent_name="coordinator",
            input_contract_class=CoordinatorInput,
            output_contract_class=CoordinatorOutput,
            llm=llm,
            config=config,
            event_emitter=event_emitter
        )
        self.name = "Coordinator"  # Capital for test compatibility
        self.description = "Routes research requests to appropriate agents"
    
    def execute(
        self,
        validated_input: CoordinatorInput,
        config: Dict[str, Any]
    ) -> CoordinatorOutput:
        """
        Execute coordinator logic with validated input.
        
        Args:
            validated_input: Validated input contract
            config: Configuration dictionary
            
        Returns:
            CoordinatorOutput with classification and routing information
        """
        logger.info("Coordinator agent processing request")
        
        # Get the latest user message from validated input
        user_message = self._get_latest_user_message_from_input(validated_input)
        if not user_message:
            logger.warning("No user message found in validated input")
            # Return output indicating no message found
            return CoordinatorOutput(
                request_type="unknown",
                research_context=None,
                research_topic=""
            )
        
        # Classify the request
        request_type = self._classify_request(user_message)
        logger.info(f"Request classified as: {request_type}")
        
        # Route based on request type
        if request_type == RequestType.GREETING:
            response = self._handle_greeting(user_message)
            return CoordinatorOutput(
                request_type=request_type.value,
                research_context=None,
                research_topic=""
            )
        
        elif request_type == RequestType.INAPPROPRIATE:
            response = self._handle_inappropriate(user_message)
            return CoordinatorOutput(
                request_type=request_type.value,
                research_context=None,
                research_topic=""
            )
        
        elif request_type == RequestType.CLARIFICATION:
            response = self._request_clarification(user_message)
            return CoordinatorOutput(
                request_type=request_type.value,
                research_context=None,
                research_topic=""
            )
        
        elif request_type == RequestType.RESEARCH:
            # Extract research topic
            research_topic = self._extract_research_topic(user_message)
            
            # Return contract output for research request
            logger.info("Research request processed - router will determine next step")
            return CoordinatorOutput(
                request_type=request_type.value,
                research_context=f"User wants to research: {research_topic}",
                research_topic=research_topic
            )
        
        else:
            # Unknown request type
            response = self._handle_unknown(user_message)
            return CoordinatorOutput(
                request_type=RequestType.UNKNOWN.value,
                research_context=None,
                research_topic=""
            )
    
    def _get_latest_user_message_from_input(self, validated_input: CoordinatorInput) -> Optional[str]:
        """Extract the latest user message from validated input."""
        return get_last_user_message(validated_input.messages)
    
    def _get_latest_user_message(self, state: EnhancedResearchState) -> Optional[str]:
        """Extract the latest user message from state (legacy method)."""
        messages = state.get("messages", [])
        return get_last_user_message(messages)
    
    def _classify_request(self, message: str) -> RequestType:
        """
        Classify the type of request using tiered approach:
        1. High-confidence pattern matching (fast, ~99% of cases)
        2. LLM classification with timeout (intelligent but protected)
        3. Safe default fallback

        This approach avoids unnecessary LLM calls for obvious research requests,
        preventing hangs when LLM endpoints are slow or unresponsive.
        """
        import time
        import re

        start_time = time.time()
        message_lower = message.lower().strip()

        # TIER 1: HIGH-CONFIDENCE pattern matching (executes BEFORE LLM check)

        # Check for simple greetings (short messages with greeting words at word boundaries)
        greeting_patterns = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood afternoon\b",
            r"\bgood evening\b", r"\bgreetings\b", r"\bhowdy\b", r"\bwhat\'?s up\b"
        ]
        if any(re.search(pattern, message_lower) for pattern in greeting_patterns) and len(message_lower) < 30:
            # Additional check: ensure it's just a greeting, not part of a larger research request
            if not any(word in message_lower for word in ["compare", "analyze", "research", "help me", "tell me", "what is"]):
                logger.info(f"Classified as GREETING via pattern matching (took {time.time() - start_time:.3f}s)")
                return RequestType.GREETING

        # Check for inappropriate content (security patterns)
        inappropriate_patterns = [
            "hack", "illegal", "password", "credential", "malware",
            "exploit", "vulnerability attack", "ddos", "break into",
            "bypass security", "crack"
        ]
        if any(pattern in message_lower for pattern in inappropriate_patterns):
            logger.info(f"Classified as INAPPROPRIATE via pattern matching (took {time.time() - start_time:.3f}s)")
            return RequestType.INAPPROPRIATE

        # *** KEY FIX: Check research patterns BEFORE LLM ***
        # This prevents unnecessary LLM calls for obvious research requests (99% of cases)
        # and avoids hangs when LLM endpoints are slow/unresponsive
        research_patterns = [
            "research", "analyze", "investigate", "find out", "tell me about",
            "what is", "how does", "explain", "describe", "compare",
            "what are the", "give me information", "help me", "i want to know",
            "i need to", "should i", "which is better", "pros and cons",
            "difference between", "versus", "vs", "what should i",
            "recommend", "suggest", "advice", "opinion", "i want to", "i want a"
        ]
        if any(pattern in message_lower for pattern in research_patterns):
            logger.info(f"Classified as RESEARCH via pattern matching (took {time.time() - start_time:.3f}s)")
            return RequestType.RESEARCH

        # Check for clarification requests
        clarification_patterns = [
            "what do you mean", "can you explain", "i don't understand",
            "clarify", "confused", "not clear"
        ]
        if any(pattern in message_lower for pattern in clarification_patterns):
            logger.info(f"Classified as CLARIFICATION via pattern matching (took {time.time() - start_time:.3f}s)")
            return RequestType.CLARIFICATION

        # Check if it's a question (likely research)
        if message.strip().endswith("?"):
            logger.info(f"Classified as RESEARCH via question mark heuristic (took {time.time() - start_time:.3f}s)")
            return RequestType.RESEARCH

        # For longer messages that don't match patterns, likely RESEARCH
        if len(message.strip()) > 50:
            logger.info(f"Classified as RESEARCH via length heuristic (took {time.time() - start_time:.3f}s)")
            return RequestType.RESEARCH

        # TIER 2: LLM classification for AMBIGUOUS cases (with timeout protection)
        # Only reach here for short messages that don't match any patterns
        if self.llm:
            try:
                logger.info(f"No high-confidence pattern match, trying LLM classification...")
                result = self._llm_classify_with_timeout(message, timeout=5.0)
                logger.info(f"Classified as {result.value} via LLM (took {time.time() - start_time:.3f}s)")
                return result
            except Exception as e:
                logger.warning(f"LLM classification failed or timed out: {e}. Falling back to default.")
                # Fall through to default

        # TIER 3: Safe default fallback
        # If no patterns matched and LLM failed/unavailable, default to RESEARCH
        logger.info(f"Defaulting to RESEARCH classification (took {time.time() - start_time:.3f}s)")
        return RequestType.RESEARCH

    def _llm_classify_request(self, message: str) -> RequestType:
        """
        Use LLM to intelligently classify the request type.
        
        This handles nuanced cases that pattern matching misses, like:
        - "I'm buying something. Should I get iPhone or Android?"
        - "Help me decide between different universities"
        - "I want a rigorous comparison of tax systems"
        """
        classification_prompt = f"""Classify this user request into exactly one of these categories:

RESEARCH - Any request for information, analysis, comparison, or help making decisions. This includes:
- Comparing options ("iPhone vs Android", "which is better")
- Asking for explanations ("what is machine learning")
- Seeking advice ("should I", "help me decide")
- Requesting analysis ("analyze the pros and cons")
- Decision support ("I'm buying X, what should I choose")
- Any question seeking factual information

CLARIFICATION - User is confused about a previous response and needs clarification

GREETING - Simple social greetings with no informational request

INAPPROPRIATE - Requests for illegal activities, hacking, security exploits

User request: "{message}"

Important: Only reply with the category name (RESEARCH, CLARIFICATION, GREETING, or INAPPROPRIATE). No explanation needed."""

        try:
            # Use the LLM to classify
            response = self.llm.invoke(classification_prompt)
            
            # Extract the classification from response using helper
            from ..core.llm_response_parser import extract_text_from_response
            classification = extract_text_from_response(response).upper()
            
            # Map to our RequestType enum
            classification_map = {
                'RESEARCH': RequestType.RESEARCH,
                'CLARIFICATION': RequestType.CLARIFICATION, 
                'GREETING': RequestType.GREETING,
                'INAPPROPRIATE': RequestType.INAPPROPRIATE
            }
            
            if classification in classification_map:
                logger.info(f"LLM classified '{message[:50]}...' as {classification}")
                return classification_map[classification]
            else:
                logger.warning(f"LLM returned unexpected classification: {classification}")
                # Default to RESEARCH for unexpected classifications
                return RequestType.RESEARCH
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            raise  # Re-raise to trigger fallback logic

    def _llm_classify_with_timeout(self, message: str, timeout: float = 5.0) -> RequestType:
        """
        Use LLM to classify request with timeout protection.

        This prevents hangs when LLM endpoints are slow or unresponsive by wrapping
        the LLM call in a thread with a timeout. If the timeout expires, the call
        is aborted and an exception is raised, allowing fallback to pattern matching.

        Args:
            message: User message to classify
            timeout: Maximum seconds to wait for LLM response (default: 5.0)

        Returns:
            RequestType classification from LLM

        Raises:
            TimeoutError: If LLM call exceeds timeout
            Exception: If LLM call fails for other reasons
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        def _classify():
            """Inner function to run in thread."""
            return self._llm_classify_request(message)

        # Use single-use ThreadPoolExecutor to avoid shared pool issues
        # Each timeout-wrapped call gets its own executor that's properly cleaned up
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_classify)
            try:
                result = future.result(timeout=timeout)
                return result
            except FuturesTimeoutError as e:
                logger.error(f"LLM classification timed out after {timeout}s for message: {message[:100]}...")
                raise TimeoutError(f"LLM classification timed out after {timeout}s") from e
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                raise

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
    
    # Removed _respond_and_end method - no longer needed with contract approach
    # State updates are handled by the contract system