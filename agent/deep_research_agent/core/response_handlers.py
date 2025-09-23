"""
Abstract response handling utilities for various LLM providers.

Provides reusable, model-agnostic utilities for parsing structured LLM responses.
"""

from typing import Any, Dict, List, Tuple, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class ResponseType(Enum):
    """Types of content in structured responses."""
    TEXT = "text"
    REASONING = "reasoning" 
    THINKING = "thinking"
    ANSWER = "answer"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


@dataclass
class ParsedResponse:
    """Container for parsed LLM response components."""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    response_type: ResponseType = ResponseType.TEXT
    metadata: Optional[Dict[str, Any]] = None


class ResponseParser(Protocol):
    """Protocol for response parsers."""
    
    def can_handle(self, response: Any) -> bool:
        """Check if this parser can handle the given response format."""
        ...
    
    def parse(self, response: Any) -> ParsedResponse:
        """Parse the response into structured components."""
        ...


class DatabricksResponseParser:
    """Parser for Databricks structured responses."""
    
    def can_handle(self, response: Any) -> bool:
        """Check if response is a Databricks structured format."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
            
        # Check for Databricks list format
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') in ['reasoning', 'text']:
                    return True
        
        # Check for single Databricks dict format
        if isinstance(content, dict) and content.get('type') in ['reasoning', 'text']:
            return True
            
        return False
    
    def parse(self, response: Any) -> ParsedResponse:
        """Parse Databricks structured response."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        reasoning_text = ""
        actual_content = ""
        metadata = {}
        
        if isinstance(content, list):
            # Handle list format: [{"type": "reasoning", ...}, {"type": "text", ...}]
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'reasoning':
                        reasoning_text = self._extract_reasoning(item)
                        metadata['has_reasoning'] = True
                    elif item.get('type') == 'text':
                        actual_content = item.get('text', '')
                        
        elif isinstance(content, dict):
            # Handle single dict format
            if content.get('type') == 'reasoning':
                reasoning_text = self._extract_reasoning(content)
                metadata['has_reasoning'] = True
            elif content.get('type') == 'text':
                actual_content = content.get('text', '')
        
        # Enhanced content extraction with validation
        logger.debug(f"DatabricksResponseParser: actual_content length={len(actual_content)}, reasoning_text length={len(reasoning_text)}")
        
        # Log samples for debugging missing content (debug level)
        if actual_content:
            logger.debug(f"Sample actual_content: {actual_content[:200]}...")
        if reasoning_text:
            logger.debug(f"Sample reasoning_text: {reasoning_text[:200]}...")
        
        # Determine final content with stricter validation
        if actual_content and actual_content.strip():
            # We have actual content - clean it and use it
            final_content = self._clean_json_artifacts(actual_content)
            response_type = ResponseType.TEXT
            metadata['content_source'] = 'text_block'
            logger.debug("Using actual text content")
        elif reasoning_text and reasoning_text.strip():
            # Only fall back to reasoning if it's substantial enough and likely not reasoning
            reasoning_stripped = reasoning_text.strip()
            
            # Check if reasoning looks like actual content (not reasoning)
            # Reasoning typically contains phrases like "I need to", "Let me think", etc.
            reasoning_indicators = [
                "i need to", "let me", "i should", "i will", "first i", 
                "thinking about", "considering", "my approach", "i'll",
                "step 1:", "step 2:", "next, i", "based on this"
            ]
            
            is_likely_reasoning = any(indicator in reasoning_stripped.lower() for indicator in reasoning_indicators)
            
            # Check if this looks like structured content (even if short)
            has_structure_markers = any(marker in reasoning_stripped for marker in [
                "# ", "## ", "### ",  # Headers
                "Research Report", "Executive Summary", "Key Findings", "References",  # Report sections
                "- ", "• ", "* ",  # List items
                "|", "```",  # Tables or code blocks
            ])
            
            # More lenient criteria for content extraction
            min_length = 30 if has_structure_markers else 100
            
            if not is_likely_reasoning and len(reasoning_stripped) > min_length:
                # This might be actual content mislabeled as reasoning - clean it
                final_content = self._clean_json_artifacts(reasoning_text)
                response_type = ResponseType.TEXT
                metadata['content_source'] = 'reasoning_block_repurposed'
                logger.warning("Using reasoning text as content - may be mislabeled content")
            else:
                # This is likely actual reasoning - don't use as content
                final_content = ""
                response_type = ResponseType.REASONING
                metadata['content_source'] = 'empty_fallback'
                logger.warning("Both content and reasoning are empty or reasoning is actual reasoning - returning empty")
        else:
            # Both are empty
            final_content = ""
            response_type = ResponseType.UNKNOWN
            metadata['content_source'] = 'empty'
            logger.warning("No content or reasoning found in response")
        
        return ParsedResponse(
            content=final_content.strip(),
            reasoning=reasoning_text.strip() if reasoning_text else None,
            response_type=response_type,
            metadata=metadata
        )
    
    def _extract_reasoning(self, reasoning_block: Dict[str, Any]) -> str:
        """Extract reasoning text from reasoning block."""
        summary = reasoning_block.get('summary', [])
        if isinstance(summary, list):
            for summary_item in summary:
                if isinstance(summary_item, dict) and summary_item.get('type') == 'summary_text':
                    return summary_item.get('text', '')
        return ""

    def _clean_json_artifacts(self, content: str) -> str:
        """Clean JSON artifacts from extracted content."""
        if not content:
            return content
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip lines that are pure JSON artifacts
            if not stripped or stripped in ['| |', '| | },', '| | }, ]', '},', '}]', '], ]', '], ', '}, ]', '} ]', ']', '[, ]', '[,]']:
                continue
            
            # Skip lines starting with JSON artifacts
            if stripped.startswith('}, ]') or stripped.startswith('}]') or stripped.startswith('], ]') or stripped.startswith('[, ]') or stripped.startswith('[,]'):
                # Try to salvage any content after the artifact
                salvaged = stripped
                for artifact in ['}, ]', '}]', '], ]', '} ]', '}, ', '}', ']', '[, ]', '[,]']:
                    if salvaged.startswith(artifact):
                        salvaged = salvaged[len(artifact):].strip()
                        break
                if salvaged and len(salvaged) > 3:  # Only keep if there's meaningful content
                    cleaned_lines.append(salvaged)
                continue
                
            # Clean inline JSON artifacts at the beginning of lines
            cleaned_line = stripped
            for artifact in ['}, ]', '}]', '], ]', '} ]', '},', '}', ']', '[, ]', '[,]']:
                if cleaned_line.startswith(artifact):
                    cleaned_line = cleaned_line[len(artifact):].strip()
                    break
            
            # Skip if the line became empty or is just punctuation
            if cleaned_line and len(cleaned_line) > 2:
                cleaned_lines.append(line.replace(stripped, cleaned_line) if cleaned_line != stripped else line)
        
        result = '\n'.join(cleaned_lines)
        
        # Log if we cleaned anything
        if len(result) != len(content):
            logger.info(f"Cleaned JSON artifacts: {len(content)} -> {len(result)} chars")
            
            # Log significant content reduction as warning
            reduction_ratio = (len(content) - len(result)) / len(content) if len(content) > 0 else 0
            if reduction_ratio > 0.5:
                logger.warning(f"SIGNIFICANT CONTENT REDUCTION: {reduction_ratio:.1%} removed during JSON cleaning")
                logger.warning(f"Original sample: {content[:300]}")
                logger.warning(f"Cleaned sample: {result[:300]}")
        
        return result


class PlainTextResponseParser:
    """Parser for plain text responses."""
    
    def can_handle(self, response: Any) -> bool:
        """Check if response is plain text."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        return isinstance(content, str)
    
    def parse(self, response: Any) -> ParsedResponse:
        """Parse plain text response."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        return ParsedResponse(
            content=content.strip(),
            response_type=ResponseType.TEXT
        )


class GenericStructuredResponseParser:
    """Parser for generic structured responses."""
    
    def can_handle(self, response: Any) -> bool:
        """Check if response is some other structured format."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        return isinstance(content, (list, dict))
    
    def parse(self, response: Any) -> ParsedResponse:
        """Parse generic structured response."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        text_parts = []
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Look for common text fields
                    for field in ['text', 'content', 'message', 'response']:
                        if field in item and item[field]:
                            text_parts.append(str(item[field]))
                            break
                elif isinstance(item, str):
                    text_parts.append(item)
        
        elif isinstance(content, dict):
            # Look for common text fields
            for field in ['text', 'content', 'message', 'response']:
                if field in content and content[field]:
                    text_parts.append(str(content[field]))
                    break
        
        final_content = '\n'.join(text_parts) if text_parts else str(content)
        
        return ParsedResponse(
            content=final_content.strip(),
            response_type=ResponseType.UNKNOWN
        )


class UniversalResponseHandler:
    """Universal handler that tries multiple parsers in order."""
    
    def __init__(self):
        self.parsers = [
            DatabricksResponseParser(),
            PlainTextResponseParser(),
            GenericStructuredResponseParser(),
        ]
    
    def parse_response(self, response: Any) -> ParsedResponse:
        """Parse response using the first compatible parser."""
        for parser in self.parsers:
            if parser.can_handle(response):
                logger.debug(f"Using {parser.__class__.__name__} for response parsing")
                return parser.parse(response)
        
        # Fallback
        logger.warning("No compatible parser found, using string conversion")
        return ParsedResponse(
            content=str(response),
            response_type=ResponseType.UNKNOWN
        )
    
    def extract_text(self, response: Any) -> str:
        """Extract just the text content (backward compatible)."""
        parsed = self.parse_response(response)
        return parsed.content
    
    def extract_with_reasoning(self, response: Any) -> Tuple[str, Optional[str]]:
        """Extract both content and reasoning."""
        parsed = self.parse_response(response)
        return parsed.content, parsed.reasoning


# Global instance for backward compatibility and easy usage
_universal_handler = UniversalResponseHandler()


def extract_text_from_response(response: Any) -> str:
    """
    Extract text content from any LLM response format.
    
    This is the main entry point for backward compatibility.
    """
    return _universal_handler.extract_text(response)


def parse_structured_response(response: Any) -> ParsedResponse:
    """
    Parse response into structured components.
    
    Returns:
        ParsedResponse with content, reasoning, and metadata
    """
    return _universal_handler.parse_response(response)


def extract_content_and_reasoning(response: Any) -> Tuple[str, Optional[str]]:
    """
    Extract both content and reasoning from response.
    
    Returns:
        Tuple of (content, reasoning) where reasoning may be None
    """
    return _universal_handler.extract_with_reasoning(response)


# Model-specific convenience functions
def register_custom_parser(parser: ResponseParser, priority: int = 0) -> None:
    """Register a custom parser with the universal handler."""
    if priority == 0:
        _universal_handler.parsers.append(parser)
    else:
        _universal_handler.parsers.insert(priority, parser)


def create_model_specific_handler(model_name: str) -> UniversalResponseHandler:
    """Create a handler optimized for specific model."""
    handler = UniversalResponseHandler()
    
    if "databricks" in model_name.lower():
        # Prioritize Databricks parser
        handler.parsers.insert(0, DatabricksResponseParser())
    elif "openai" in model_name.lower() or "gpt" in model_name.lower():
        # Could add OpenAI-specific parser here
        pass
    elif "claude" in model_name.lower():
        # Could add Claude-specific parser here
        pass
    
    return handler
