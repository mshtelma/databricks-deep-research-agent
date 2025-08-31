"""
Message conversion utilities for the research agent.

This module handles conversion between different message formats used by
LangChain, MLflow ResponsesAgent, and OpenAI APIs.
"""

import json
from typing import Dict, List, Any, Union, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
# ChatMessage removed - using dictionary format as per Databricks standards
from mlflow.types.responses import ResponsesAgentRequest

from deep_research_agent.core import (
    get_logger, 
    MessageConversionError, 
    AgentRole,
    ContentType,
    ResponseContent
)

logger = get_logger(__name__)


class MessageConverter:
    """Handles conversion between different message formats."""
    
    def __init__(self):
        """Initialize message converter."""
        pass
    
    def extract_text_content(self, content: ContentType) -> str:
        """
        Extract text content from various formats.
        
        Args:
            content: Content in various formats (string, list, dict)
            
        Returns:
            Extracted text content
            
        Raises:
            MessageConversionError: If content extraction fails
        """
        try:
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(item['text'])
                        elif 'content' in item:
                            text_parts.append(str(item['content']))
                        else:
                            # Try to extract any string values
                            for value in item.values():
                                if isinstance(value, str):
                                    text_parts.append(value)
                return ''.join(text_parts)
            elif isinstance(content, dict):
                if 'text' in content:
                    return content['text']
                elif 'content' in content:
                    return str(content['content'])
                else:
                    # Try to find text content in dict values
                    for value in content.values():
                        if isinstance(value, str) and len(value) > 0:
                            return value
                    return str(content)
            else:
                return str(content)
        except Exception as e:
            logger.error("Failed to extract text content", error=e, content_type=type(content).__name__)
            raise MessageConversionError(f"Failed to extract text content: {e}")
    
    def responses_to_langchain(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Convert ResponsesAgent messages to LangChain format.
        
        Args:
            messages: List of ResponsesAgent message dictionaries
            
        Returns:
            List of LangChain BaseMessage objects
            
        Raises:
            MessageConversionError: If conversion fails
        """
        try:
            langchain_messages = []
            
            for msg in messages:
                role = msg.get('role', '').lower()
                content = msg.get('content', '')
                
                # Extract text content if it's structured
                if isinstance(content, (list, dict)):
                    content = self.extract_text_content(content)
                
                if role == AgentRole.USER:
                    langchain_messages.append(HumanMessage(content=content))
                elif role == AgentRole.ASSISTANT:
                    langchain_messages.append(AIMessage(content=content))
                elif role == AgentRole.SYSTEM:
                    langchain_messages.append(SystemMessage(content=content))
                else:
                    logger.warning(f"Unknown message role: {role}", role=role)
                    # Default to HumanMessage for unknown roles
                    langchain_messages.append(HumanMessage(content=content))
            
            return langchain_messages
        except Exception as e:
            logger.error("Failed to convert ResponsesAgent messages to LangChain", error=e)
            raise MessageConversionError(f"Failed to convert messages: {e}")
    
    def langchain_to_responses(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Convert LangChain messages to ResponsesAgent format.
        
        Args:
            messages: List of LangChain BaseMessage objects
            
        Returns:
            List of ResponsesAgent message dictionaries
            
        Raises:
            MessageConversionError: If conversion fails
        """
        try:
            responses_messages = []
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    responses_messages.append({
                        "role": AgentRole.USER,
                        "content": self.extract_text_content(msg.content)
                    })
                elif isinstance(msg, AIMessage):
                    # Handle tool calls if present
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        responses_messages.extend(
                            self._convert_tool_calls(msg)
                        )
                    else:
                        responses_messages.append({
                            "role": AgentRole.ASSISTANT,
                            "content": self.extract_text_content(msg.content)
                        })
                elif isinstance(msg, SystemMessage):
                    responses_messages.append({
                        "role": AgentRole.SYSTEM,
                        "content": self.extract_text_content(msg.content)
                    })
                else:
                    logger.warning(f"Unknown message type: {type(msg)}", message_type=type(msg).__name__)
                    responses_messages.append({
                        "role": AgentRole.ASSISTANT,
                        "content": str(msg.content) if hasattr(msg, 'content') else str(msg)
                    })
            
            return responses_messages
        except Exception as e:
            logger.error("Failed to convert LangChain messages to ResponsesAgent", error=e)
            raise MessageConversionError(f"Failed to convert messages: {e}")
    
    def _convert_tool_calls(self, ai_message: AIMessage) -> List[Dict[str, Any]]:
        """
        Convert AI message with tool calls to ResponsesAgent format.
        
        Args:
            ai_message: AIMessage with tool calls
            
        Returns:
            List of ResponsesAgent message dictionaries
        """
        messages = []
        
        for tool_call in ai_message.tool_calls:
            messages.append({
                "type": "function_call",
                "call_id": tool_call.get("id", str(uuid4())),
                "name": tool_call.get("name", "unknown_function"),
                "arguments": json.dumps(tool_call.get("args", {})),
            })
        
        return messages
    
    def openai_to_langchain(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Convert OpenAI format messages to LangChain format.
        
        Args:
            messages: List of OpenAI message dictionaries
            
        Returns:
            List of LangChain BaseMessage objects
        """
        try:
            langchain_messages = []
            
            for msg in messages:
                role = msg.get('role', '').lower()
                content = msg.get('content', '')
                
                if role == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
                elif role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                else:
                    logger.warning(f"Unknown OpenAI message role: {role}", role=role)
                    langchain_messages.append(HumanMessage(content=content))
            
            return langchain_messages
        except Exception as e:
            logger.error("Failed to convert OpenAI messages to LangChain", error=e)
            raise MessageConversionError(f"Failed to convert OpenAI messages: {e}")
    
    def request_to_langchain(self, request: ResponsesAgentRequest) -> List[BaseMessage]:
        """
        Convert ResponsesAgentRequest to LangChain messages.
        
        Args:
            request: ResponsesAgentRequest object
            
        Returns:
            List of LangChain BaseMessage objects
        """
        try:
            messages = []
            
            for msg in request.input:
                # Handle both dictionary format (Databricks standard) and Message objects
                if isinstance(msg, dict):
                    # Dictionary format: {"role": "user", "content": "..."}
                    role = msg.get('role', 'user')
                    content = self.extract_text_content(msg.get('content', ''))
                else:
                    # Message object format
                    role = getattr(msg, 'role', 'user')
                    content = self.extract_text_content(getattr(msg, 'content', ''))
                
                # Convert role string to AgentRole for comparison
                if role == AgentRole.USER or role == 'user':
                    messages.append(HumanMessage(content=content))
                elif role == AgentRole.ASSISTANT or role == 'assistant':
                    messages.append(AIMessage(content=content))
                elif role == AgentRole.SYSTEM or role == 'system':
                    messages.append(SystemMessage(content=content))
                else:
                    logger.warning(f"Unknown request message role: {role}", role=role)
                    messages.append(HumanMessage(content=content))
            
            return messages
        except Exception as e:
            logger.error("Failed to convert ResponsesAgentRequest to LangChain", error=e)
            raise MessageConversionError(f"Failed to convert request: {e}")
    
    def validate_message_format(self, message: Dict[str, Any]) -> bool:
        """
        Validate message format.
        
        Args:
            message: Message dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['role', 'content']
        
        for field in required_fields:
            if field not in message:
                logger.warning(f"Missing required field in message: {field}", message=message)
                return False
        
        valid_roles = [role.value for role in AgentRole]
        if message['role'] not in valid_roles:
            logger.warning(f"Invalid message role: {message['role']}", role=message['role'])
            return False
        
        return True
    
    def normalize_message_content(self, content: Any) -> str:
        """
        Normalize message content to string format.
        
        Args:
            content: Content in any format
            
        Returns:
            Normalized string content
        """
        try:
            return self.extract_text_content(content)
        except Exception as e:
            logger.warning("Failed to normalize message content", error=e)
            return str(content)


# Create singleton instance
message_converter = MessageConverter()