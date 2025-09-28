"""
Redaction utilities for sanitizing PII and secrets from agent events.

This module provides utilities to safely redact sensitive information
from intermediate events before they are emitted to the UI.
"""

import re
from typing import Any, Dict, List, Optional, Union

from . import get_logger

logger = get_logger(__name__)


class RedactionUtils:
    """Utilities for redacting sensitive information from agent events."""
    
    DEFAULT_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
        r'\bsk-[A-Za-z0-9]{32,}\b',  # OpenAI-style API keys (more specific, check first)
        r'(?<=Bearer\s)token:\s*[A-Za-z0-9._:/\-]+(?:\s+[A-Za-z0-9._:/\-]+)*',  # Bearer token: xyz (lookbehind for Bearer, specifically token: pattern) - check first
        r'\bBearer\s+[A-Za-z0-9._/-]+\b',  # Bearer tokens (direct replacement, like "Bearer abc123def456")
        r'\bkey[_-]?[0-9a-zA-Z]{8,}\b',  # Generic key patterns
        r'\btoken[_-]?[0-9a-zA-Z]{8,}\b',  # Generic token patterns
        r'\bpassword[_-]?[^\s]{4,}\b',  # Password patterns
        r'\bsecret[_-]?[0-9a-zA-Z]{8,}\b',  # Secret patterns
        r'\b[A-Za-z0-9]{20,}\b',  # potential API keys (20+ alphanumeric chars) - less specific, check last
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card patterns
    ]
    
    REDACTION_PLACEHOLDER = "[REDACTED]"
    
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """Initialize with custom redaction patterns."""
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        # Compile regex patterns for efficiency
        self._compiled_patterns = []
        for pattern in self.patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    
    def redact_text(self, text: str) -> str:
        """Redact sensitive information from text."""
        if not isinstance(text, str):
            return str(text)
        
        redacted_text = text
        redaction_count = 0
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(redacted_text)
            if matches:
                redacted_text = pattern.sub(self.REDACTION_PLACEHOLDER, redacted_text)
                redaction_count += len(matches)
        
        if redaction_count > 0:
            logger.debug(f"Redacted {redaction_count} sensitive items from text")
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive information from dictionary."""
        if not isinstance(data, dict):
            return data
        
        redacted_data = {}
        
        for key, value in data.items():
            # Redact key names that might contain sensitive info
            redacted_key = self.redact_text(key)
            
            # Recursively redact values
            if isinstance(value, str):
                redacted_data[redacted_key] = self.redact_text(value)
            elif isinstance(value, dict):
                redacted_data[redacted_key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted_data[redacted_key] = self.redact_list(value)
            else:
                redacted_data[redacted_key] = value
        
        return redacted_data
    
    def redact_list(self, data: List[Any]) -> List[Any]:
        """Recursively redact sensitive information from list."""
        if not isinstance(data, list):
            return data
        
        redacted_list = []
        
        for item in data:
            if isinstance(item, str):
                redacted_list.append(self.redact_text(item))
            elif isinstance(item, dict):
                redacted_list.append(self.redact_dict(item))
            elif isinstance(item, list):
                redacted_list.append(self.redact_list(item))
            else:
                redacted_list.append(item)
        
        return redacted_list
    
    def redact_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from event data."""
        return self.redact_dict(event_data)
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length with ellipsis."""
        if not isinstance(text, str) or len(text) <= max_length:
            return text
        
        if max_length < 4:
            return text[:max_length]
        
        return text[:max_length - 3].rstrip() + "..."
    
    def sanitize_for_ui(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        preserve_structure: bool = True
    ) -> str:
        """
        Sanitize text for UI display by redacting sensitive info and truncating.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length to truncate to
            preserve_structure: Whether to preserve JSON/structured format
        
        Returns:
            Sanitized text safe for UI display
        """
        if not isinstance(text, str):
            text = str(text)
        
        # First redact sensitive information
        sanitized = self.redact_text(text)
        has_redactions = "[REDACTED]" in sanitized
        
        # Then truncate if needed
        if max_length and len(sanitized) > max_length:
            truncated = self.truncate_text(sanitized, max_length)
            
            # If we had redactions but lost them during truncation, 
            # ensure at least one [REDACTED] marker is preserved
            if has_redactions and "[REDACTED]" not in truncated:
                # Try to fit at least one [REDACTED] marker
                redacted_marker = "[REDACTED]"
                if max_length >= len(redacted_marker) + 3:  # +3 for "..."
                    # Replace end of truncated text with redaction marker
                    available_space = max_length - len(redacted_marker) - 3
                    if available_space > 0:
                        truncated = sanitized[:available_space].rstrip() + redacted_marker + "..."
                    else:
                        truncated = redacted_marker
                else:
                    # If not enough space, just use the redacted marker
                    truncated = redacted_marker[:max_length]
            
            sanitized = truncated
        
        return sanitized


# Global instance for easy access
_global_redactor = None


def get_redactor(custom_patterns: Optional[List[str]] = None) -> RedactionUtils:
    """Get a global redactor instance or create one with custom patterns."""
    global _global_redactor
    
    if _global_redactor is None or custom_patterns:
        _global_redactor = RedactionUtils(custom_patterns)
    
    return _global_redactor


def redact_text(text: str, custom_patterns: Optional[List[str]] = None) -> str:
    """Convenience function to redact text using global redactor."""
    redactor = get_redactor(custom_patterns)
    return redactor.redact_text(text)


def redact_dict(data: Dict[str, Any], custom_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function to redact dict using global redactor."""
    redactor = get_redactor(custom_patterns)
    return redactor.redact_dict(data)


def sanitize_for_ui(
    text: str, 
    max_length: Optional[int] = None, 
    custom_patterns: Optional[List[str]] = None
) -> str:
    """Convenience function to sanitize text for UI using global redactor."""
    redactor = get_redactor(custom_patterns)
    return redactor.sanitize_for_ui(text, max_length)
