"""
Polymorphic message handling utilities for the research agent.

This module implements the "Accept Anything, Process Correctly" principle,
handling all message formats (LangChain Messages, dict format, MLflow ChatMessage)
without requiring format conversions at boundaries.
"""

import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from functools import lru_cache
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class MessageFormat(Enum):
    """Detected message format types."""
    LANGCHAIN_MESSAGE = "langchain_message"
    DICT_FORMAT = "dict_format"
    MLFLOW_CHAT_MESSAGE = "mlflow_chat_message"
    STRING_CONTENT = "string_content"
    UNKNOWN = "unknown"


class MessageAccessor:
    """
    Polymorphic message accessor that handles any message format.
    
    Implements format detection, caching, and consistent access patterns
    for all message types used across the codebase.
    """
    
    def __init__(self):
        """Initialize message accessor with format cache."""
        self._format_cache = {}
    
    def detect_format(self, message: Any) -> MessageFormat:
        """
        Detect the format of a message object.
        
        Args:
            message: Message in any format
            
        Returns:
            Detected MessageFormat enum value
        """
        # Use object id for caching to avoid expensive format detection
        message_id = id(message)
        
        if message_id in self._format_cache:
            return self._format_cache[message_id]
        
        detected_format = self._detect_format_uncached(message)
        self._format_cache[message_id] = detected_format
        
        return detected_format
    
    def _detect_format_uncached(self, message: Any) -> MessageFormat:
        """Internal format detection without caching."""
        try:
            # LangChain BaseMessage objects
            if isinstance(message, BaseMessage):
                return MessageFormat.LANGCHAIN_MESSAGE
            
            # MLflow ChatMessage objects (have role and content attributes)
            if hasattr(message, 'role') and hasattr(message, 'content'):
                return MessageFormat.MLFLOW_CHAT_MESSAGE
            
            # Dictionary format
            if isinstance(message, dict):
                if 'role' in message and 'content' in message:
                    return MessageFormat.DICT_FORMAT
                elif 'content' in message:
                    return MessageFormat.DICT_FORMAT
            
            # String content
            if isinstance(message, str):
                return MessageFormat.STRING_CONTENT
            
            return MessageFormat.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Error detecting message format: {e}", message_type=type(message).__name__)
            return MessageFormat.UNKNOWN
    
    def extract_content(self, message: Any) -> str:
        """
        Extract text content from any message format.
        
        Args:
            message: Message in any format
            
        Returns:
            Extracted text content as string
        """
        try:
            format_type = self.detect_format(message)
            
            if format_type == MessageFormat.LANGCHAIN_MESSAGE:
                return self._extract_from_langchain(message)
            elif format_type == MessageFormat.MLFLOW_CHAT_MESSAGE:
                return self._extract_from_mlflow_message(message)
            elif format_type == MessageFormat.DICT_FORMAT:
                return self._extract_from_dict(message)
            elif format_type == MessageFormat.STRING_CONTENT:
                return str(message)
            else:
                logger.warning(f"Unknown message format for content extraction: {type(message)}")
                return str(message)
                
        except Exception as e:
            logger.error(f"Failed to extract content from message: {e}", message_type=type(message).__name__)
            return str(message)
    
    def extract_role(self, message: Any) -> str:
        """
        Extract role from any message format.
        
        Args:
            message: Message in any format
            
        Returns:
            Role string (user, assistant, system)
        """
        try:
            format_type = self.detect_format(message)
            
            if format_type == MessageFormat.LANGCHAIN_MESSAGE:
                return self._extract_role_from_langchain(message)
            elif format_type == MessageFormat.MLFLOW_CHAT_MESSAGE:
                return str(message.role) if hasattr(message, 'role') else 'user'
            elif format_type == MessageFormat.DICT_FORMAT:
                return message.get('role', 'user')
            else:
                return 'user'  # Default role for string content
                
        except Exception as e:
            logger.error(f"Failed to extract role from message: {e}", message_type=type(message).__name__)
            return 'user'
    
    def is_user_message(self, message: Any) -> bool:
        """Check if message is from user role."""
        try:
            role = self.extract_role(message)
            return role.lower() in ['user', 'human']
        except Exception:
            return False
    
    def is_assistant_message(self, message: Any) -> bool:
        """Check if message is from assistant role."""
        try:
            role = self.extract_role(message)
            return role.lower() in ['assistant', 'ai']
        except Exception:
            return False
    
    def get_message_info(self, message: Any) -> Dict[str, Any]:
        """
        Get comprehensive message information.
        
        Args:
            message: Message in any format
            
        Returns:
            Dictionary with role, content, format, and metadata
        """
        try:
            format_type = self.detect_format(message)
            role = self.extract_role(message)
            content = self.extract_content(message)
            
            info = {
                'role': role,
                'content': content,
                'format': format_type.value,
                'length': len(content),
                'is_user': role.lower() in ['user', 'human'],
                'is_assistant': role.lower() in ['assistant', 'ai'],
                'is_system': role.lower() == 'system'
            }
            
            # Add format-specific metadata
            if format_type == MessageFormat.LANGCHAIN_MESSAGE:
                info['message_type'] = type(message).__name__
                if hasattr(message, 'name'):
                    info['name'] = message.name
            elif format_type == MessageFormat.MLFLOW_CHAT_MESSAGE:
                info['message_type'] = 'ChatMessage'
            elif format_type == MessageFormat.DICT_FORMAT:
                info['message_type'] = 'dict'
                info['dict_keys'] = list(message.keys())
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get message info: {e}", message_type=type(message).__name__)
            return {
                'role': 'user',
                'content': str(message),
                'format': 'error',
                'length': len(str(message)),
                'is_user': True,
                'is_assistant': False,
                'is_system': False
            }
    
    def _extract_from_langchain(self, message: BaseMessage) -> str:
        """Extract content from LangChain BaseMessage."""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle structured content (list of dicts)
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return ''.join(text_parts)
            else:
                return str(content)
        return ""
    
    def _extract_from_mlflow_message(self, message: Any) -> str:
        """Extract content from MLflow ChatMessage."""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle structured content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return ''.join(text_parts)
            else:
                return str(content)
        return ""
    
    def _extract_from_dict(self, message: Dict[str, Any]) -> str:
        """Extract content from dictionary format."""
        content = message.get('content', '')
        
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle structured content
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return ''.join(text_parts)
        else:
            return str(content)
    
    def _extract_role_from_langchain(self, message: BaseMessage) -> str:
        """Extract role from LangChain BaseMessage."""
        if isinstance(message, HumanMessage):
            return 'user'
        elif isinstance(message, AIMessage):
            return 'assistant'
        elif isinstance(message, SystemMessage):
            return 'system'
        else:
            return 'user'


# Global message accessor instance
message_accessor = MessageAccessor()


def extract_content(message: Any) -> str:
    """
    Extract text content from any message format.
    
    Convenience function that uses the global message accessor.
    
    Args:
        message: Message in any format
        
    Returns:
        Extracted text content as string
    """
    return message_accessor.extract_content(message)


def extract_role(message: Any) -> str:
    """
    Extract role from any message format.
    
    Convenience function that uses the global message accessor.
    
    Args:
        message: Message in any format
        
    Returns:
        Role string (user, assistant, system)
    """
    return message_accessor.extract_role(message)


def get_last_user_message(messages: List[Any]) -> Optional[str]:
    """
    Get the last user message content from a list of messages.
    
    Args:
        messages: List of messages in any format
        
    Returns:
        Content of the last user message, or None if not found
    """
    try:
        if not messages:
            return None
        
        for message in reversed(messages):
            if message_accessor.is_user_message(message):
                return message_accessor.extract_content(message)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get last user message: {e}")
        return None


def get_last_assistant_message(messages: List[Any]) -> Optional[str]:
    """
    Get the last assistant message content from a list of messages.
    
    Args:
        messages: List of messages in any format
        
    Returns:
        Content of the last assistant message, or None if not found
    """
    try:
        if not messages:
            return None
        
        for message in reversed(messages):
            if message_accessor.is_assistant_message(message):
                return message_accessor.extract_content(message)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get last assistant message: {e}")
        return None


def safe_message_get(message: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a property from a message, handling all formats.
    
    This replaces the problematic `.get()` calls on message objects
    that caused AttributeError when used with LangChain Messages.
    
    Args:
        message: Message in any format
        key: Property key to access
        default: Default value if key not found
        
    Returns:
        Property value or default
    """
    try:
        format_type = message_accessor.detect_format(message)
        
        if format_type == MessageFormat.DICT_FORMAT:
            # Dictionary - use .get() safely
            return message.get(key, default)
        
        elif format_type in [MessageFormat.LANGCHAIN_MESSAGE, MessageFormat.MLFLOW_CHAT_MESSAGE]:
            # Object with attributes - use getattr
            return getattr(message, key, default)
        
        else:
            # For other formats, return default
            return default
            
    except Exception as e:
        logger.warning(f"Error in safe_message_get for key '{key}': {e}")
        return default


def is_valid_message(message: Any) -> bool:
    """
    Check if a message is valid and has required properties.
    
    Args:
        message: Message to validate
        
    Returns:
        True if message is valid, False otherwise
    """
    try:
        # Try to extract basic properties
        content = message_accessor.extract_content(message)
        role = message_accessor.extract_role(message)
        
        # Valid if we can extract content and role
        return len(content) > 0 and len(role) > 0
        
    except Exception:
        return False


def normalize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize a list of messages to consistent dict format.
    
    Args:
        messages: List of messages in any format
        
    Returns:
        List of normalized message dictionaries
    """
    normalized = []
    
    try:
        for message in messages:
            info = message_accessor.get_message_info(message)
            normalized.append({
                'role': info['role'],
                'content': info['content']
            })
        
        return normalized
        
    except Exception as e:
        logger.error(f"Failed to normalize messages: {e}")
        return []


def filter_messages_by_role(messages: List[Any], role: str) -> List[Any]:
    """
    Filter messages by role.
    
    Args:
        messages: List of messages in any format
        role: Role to filter by (user, assistant, system)
        
    Returns:
        List of messages matching the role
    """
    try:
        filtered = []
        role_lower = role.lower()
        
        for message in messages:
            message_role = message_accessor.extract_role(message).lower()
            if message_role == role_lower:
                filtered.append(message)
        
        return filtered
        
    except Exception as e:
        logger.error(f"Failed to filter messages by role '{role}': {e}")
        return []


def get_conversation_summary(messages: List[Any]) -> Dict[str, Any]:
    """
    Get a summary of the conversation.
    
    Args:
        messages: List of messages in any format
        
    Returns:
        Dictionary with conversation statistics and info
    """
    try:
        summary = {
            'total_messages': len(messages),
            'user_messages': 0,
            'assistant_messages': 0,
            'system_messages': 0,
            'total_content_length': 0,
            'last_user_message': None,
            'last_assistant_message': None,
            'message_formats': {}
        }
        
        for message in messages:
            info = message_accessor.get_message_info(message)
            
            # Count by role
            if info['is_user']:
                summary['user_messages'] += 1
                summary['last_user_message'] = info['content']
            elif info['is_assistant']:
                summary['assistant_messages'] += 1
                summary['last_assistant_message'] = info['content']
            elif info['is_system']:
                summary['system_messages'] += 1
            
            # Track content length
            summary['total_content_length'] += info['length']
            
            # Track formats
            format_name = info['format']
            summary['message_formats'][format_name] = summary['message_formats'].get(format_name, 0) + 1
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        return {'error': str(e)}