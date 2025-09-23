#!/usr/bin/env python3
"""
Content Sanitizer for Deep Research Agent

Provides comprehensive content filtering and sanitization to prevent JSON/structured data
from contaminating markdown report content. This is the critical foundation layer that
ensures clean separation between reasoning blocks and actual report content.

Key Features:
- JSON detection and removal from markdown content
- Structured data extraction and isolation
- Content type classification and validation
- Performance-optimized with minimal overhead
- Comprehensive logging for debugging
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class ContentType(Enum):
    """Content type classification."""
    PURE_MARKDOWN = "pure_markdown"
    MIXED_CONTENT = "mixed_content"
    PURE_JSON = "pure_json"
    STRUCTURED_DATA = "structured_data"
    INVALID = "invalid"


@dataclass
class SanitizationResult:
    """Result of content sanitization process."""
    clean_content: str
    extracted_reasoning: List[Dict[str, Any]]
    extracted_metadata: Dict[str, Any]
    content_type: ContentType
    warnings: List[str]
    sanitization_applied: bool


class ContentSanitizer:
    """
    Advanced content sanitizer for separating reasoning from report content.
    
    This class implements a multi-layer approach to content sanitization:
    1. JSON Detection: Identifies JSON structures within text
    2. Content Extraction: Separates structured data from markdown
    3. Content Validation: Ensures output is pure markdown
    4. Metadata Preservation: Maintains reasoning/metadata separately
    """
    
    def __init__(self, strict_mode: bool = True, preserve_metadata: bool = True):
        """
        Initialize content sanitizer.
        
        Args:
            strict_mode: If True, removes all non-markdown content aggressively
            preserve_metadata: If True, extracts and preserves reasoning/metadata
        """
        self.strict_mode = strict_mode
        self.preserve_metadata = preserve_metadata
        
        # Compile regex patterns for performance
        self._json_patterns = [
            # Array patterns - detect JSON arrays like [{"type": "reasoning", ...}]
            re.compile(r'\[\s*\{\s*["\']type["\']\s*:\s*["\'](?:reasoning|summary|metadata)["\'].*?\}\s*\]', 
                      re.DOTALL | re.IGNORECASE),
            
            # Object patterns - detect JSON objects like {"type": "reasoning", ...}
            re.compile(r'\{\s*["\']type["\']\s*:\s*["\'](?:reasoning|summary|metadata)["\'].*?\}', 
                      re.DOTALL | re.IGNORECASE),
            
            # Generic JSON array detection
            re.compile(r'\[\s*\{.*?\}\s*\]', re.DOTALL),
            
            # Generic JSON object detection (more conservative)
            re.compile(r'\{\s*["\'][^"\']+["\']\s*:\s*["\'][^"\']*["\'].*?\}', re.DOTALL)
        ]
        
        # Markdown table patterns
        self._table_patterns = [
            re.compile(r'\|.*?\|.*?\n\|[-\s|:]+\|.*?\n(?:\|.*?\|.*?\n)*', re.MULTILINE),
            re.compile(r'^[\s]*\|.*\|[\s]*$', re.MULTILINE)
        ]
        
        # Code block patterns
        self._code_patterns = [
            re.compile(r'```[\w]*\n.*?\n```', re.DOTALL),
            re.compile(r'`[^`]+`', re.DOTALL)
        ]
    
    def sanitize_content(self, content: str) -> SanitizationResult:
        """
        Main sanitization method that processes content and separates concerns.
        
        Args:
            content: Raw content that may contain mixed JSON and markdown
            
        Returns:
            SanitizationResult with clean content and extracted metadata
        """
        if not content or not isinstance(content, str):
            return SanitizationResult(
                clean_content="",
                extracted_reasoning=[],
                extracted_metadata={},
                content_type=ContentType.INVALID,
                warnings=["Empty or invalid content provided"],
                sanitization_applied=False
            )
        
        original_length = len(content)
        warnings = []
        extracted_reasoning = []
        extracted_metadata = {}
        
        # Step 1: Classify content type
        content_type = self._classify_content(content)
        logger.debug(f"Content classified as: {content_type.value}")
        
        # Step 2: Handle different content types
        if content_type == ContentType.PURE_MARKDOWN:
            # Already clean, minimal processing needed
            clean_content = self._normalize_markdown(content)
            sanitization_applied = False
            
        elif content_type == ContentType.PURE_JSON:
            # Extract all content as structured data
            extracted_data = self._extract_json_content(content)
            extracted_reasoning.extend(extracted_data.get("reasoning", []))
            extracted_metadata.update(extracted_data.get("metadata", {}))
            
            # Try to find actual content within the JSON
            clean_content = self._extract_content_from_json(extracted_data)
            if not clean_content:
                clean_content = ""
                warnings.append("No markdown content found in JSON structure")
            sanitization_applied = True
            
        elif content_type == ContentType.MIXED_CONTENT:
            # Most complex case - separate JSON from markdown
            clean_content, extracted_data = self._separate_mixed_content(content)
            extracted_reasoning.extend(extracted_data.get("reasoning", []))
            extracted_metadata.update(extracted_data.get("metadata", {}))
            sanitization_applied = True
            
        else:  # STRUCTURED_DATA or fallback
            # Conservative approach - extract what we can
            clean_content, extracted_data = self._conservative_extraction(content)
            extracted_reasoning.extend(extracted_data.get("reasoning", []))
            extracted_metadata.update(extracted_data.get("metadata", {}))
            sanitization_applied = True
        
        # Step 3: Final validation and cleanup
        clean_content = self._final_cleanup(clean_content)
        
        # Step 4: Validation
        if len(clean_content) < original_length * 0.1 and original_length > 100:
            warnings.append(f"Significant content reduction: {original_length} -> {len(clean_content)} characters")
        
        # Step 5: Quality checks
        if self._contains_json_artifacts(clean_content):
            warnings.append("Potential JSON artifacts remain in clean content")
            if self.strict_mode:
                clean_content = self._aggressive_json_removal(clean_content)
        
        logger.info(
            "Content sanitization completed",
            original_length=original_length,
            clean_length=len(clean_content),
            content_type=content_type.value,
            reasoning_blocks=len(extracted_reasoning),
            sanitization_applied=sanitization_applied,
            warnings_count=len(warnings)
        )
        
        return SanitizationResult(
            clean_content=clean_content,
            extracted_reasoning=extracted_reasoning,
            extracted_metadata=extracted_metadata,
            content_type=content_type,
            warnings=warnings,
            sanitization_applied=sanitization_applied
        )
    
    def _classify_content(self, content: str) -> ContentType:
        """Classify the type of content to determine processing strategy."""
        content_stripped = content.strip()
        
        # Check if it's pure JSON
        if self._is_pure_json(content_stripped):
            return ContentType.PURE_JSON
        
        # Check for JSON patterns
        json_matches = 0
        total_json_length = 0
        for pattern in self._json_patterns:
            matches = pattern.findall(content)
            json_matches += len(matches)
            total_json_length += sum(len(match) for match in matches)
        
        # Calculate ratios
        json_ratio = total_json_length / len(content) if content else 0
        
        # Classification logic
        if json_matches == 0 and json_ratio < 0.1:
            return ContentType.PURE_MARKDOWN
        elif json_ratio > 0.7:
            return ContentType.STRUCTURED_DATA
        elif json_matches > 0:
            return ContentType.MIXED_CONTENT
        else:
            return ContentType.PURE_MARKDOWN
    
    def _is_pure_json(self, content: str) -> bool:
        """Check if content is pure JSON."""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _extract_json_content(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON content."""
        try:
            data = json.loads(content)
            return self._normalize_extracted_data(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON content: {e}")
            return {"reasoning": [], "metadata": {}}
    
    def _separate_mixed_content(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Separate JSON structures from markdown content."""
        clean_content = content
        extracted_data = {"reasoning": [], "metadata": {}}
        
        # Process each JSON pattern
        for pattern in self._json_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                json_text = match.group(0)
                try:
                    # Try to parse and extract the JSON
                    parsed = json.loads(json_text)
                    normalized = self._normalize_extracted_data(parsed)
                    
                    # Merge extracted data
                    extracted_data["reasoning"].extend(normalized.get("reasoning", []))
                    extracted_data["metadata"].update(normalized.get("metadata", {}))
                    
                    # Remove from clean content
                    clean_content = clean_content.replace(json_text, "")
                    
                except (json.JSONDecodeError, ValueError):
                    # If it's not valid JSON, remove it anyway in strict mode
                    if self.strict_mode:
                        clean_content = clean_content.replace(json_text, "")
        
        return clean_content, extracted_data
    
    def _conservative_extraction(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Conservative extraction for unclear content types."""
        # Start with original content
        clean_content = content
        extracted_data = {"reasoning": [], "metadata": {}}
        
        # Only remove obvious JSON patterns
        for pattern in self._json_patterns[:2]:  # Only use the most specific patterns
            matches = pattern.finditer(content)
            for match in matches:
                json_text = match.group(0)
                try:
                    parsed = json.loads(json_text)
                    normalized = self._normalize_extracted_data(parsed)
                    extracted_data["reasoning"].extend(normalized.get("reasoning", []))
                    extracted_data["metadata"].update(normalized.get("metadata", {}))
                    clean_content = clean_content.replace(json_text, "")
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return clean_content, extracted_data
    
    def _normalize_extracted_data(self, data: Any) -> Dict[str, Any]:
        """Normalize extracted JSON data into standard format."""
        normalized = {"reasoning": [], "metadata": {}}
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item_type = item.get("type", "").lower()
                    if item_type in ["reasoning", "summary"]:
                        normalized["reasoning"].append(item)
                    else:
                        normalized["metadata"].update(item)
        elif isinstance(data, dict):
            data_type = data.get("type", "").lower()
            if data_type in ["reasoning", "summary"]:
                normalized["reasoning"].append(data)
            else:
                normalized["metadata"].update(data)
        
        return normalized
    
    def _extract_content_from_json(self, data: Dict[str, Any]) -> str:
        """Try to extract actual content from JSON structures."""
        content_candidates = []
        
        # Look for common content fields
        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in ["content", "text", "report", "summary", "output"]:
                        if isinstance(value, str) and len(value) > 50:
                            content_candidates.append((len(value), value))
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, f"{path}[{i}]")
        
        extract_recursive(data)
        
        # Return the longest content candidate
        if content_candidates:
            content_candidates.sort(reverse=True)
            return content_candidates[0][1]
        
        return ""
    
    def _normalize_markdown(self, content: str) -> str:
        """Normalize markdown content without major changes."""
        # Remove excessive whitespace while preserving structure
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Preserve code blocks and tables
            if line.strip().startswith('```') or '|' in line:
                normalized_lines.append(line)
            else:
                normalized_lines.append(line.rstrip())
        
        return '\n'.join(normalized_lines).strip()
    
    def _final_cleanup(self, content: str) -> str:
        """Final cleanup and validation of content."""
        if not content:
            return ""
        
        # Remove multiple consecutive newlines (max 2)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove any remaining obvious JSON fragments
        if self.strict_mode:
            content = self._aggressive_json_removal(content)
        
        return content.strip()
    
    def _contains_json_artifacts(self, content: str) -> bool:
        """Check if content still contains JSON artifacts."""
        json_indicators = [
            '{"type":',
            '"summary":',
            '"reasoning":',
            '[{"',
            '}]',
            '"},',
        ]
        
        return any(indicator in content for indicator in json_indicators)
    
    def _aggressive_json_removal(self, content: str) -> str:
        """Aggressively remove any remaining JSON-like structures."""
        # Remove JSON-like patterns - enhanced for complex structures
        patterns_to_remove = [
            # Complex nested reasoning blocks
            r'\[\s*\{\s*["\']type["\']\s*:\s*["\']reasoning["\'].*?\}\s*\]',
            r'\[\s*\{\s*["\']type["\']\s*:\s*["\']summary["\'].*?\}\s*\]',
            r'\[\s*\{\s*["\']type["\']\s*:\s*["\']text["\'].*?\}\s*\]',
            
            # Simple JSON objects with type field
            r'\{\s*["\']type["\']\s*:\s*["\'][^"\']*["\'].*?\}',
            
            # Markdown table markers followed by JSON (specific issue pattern)
            r'\|\s*\|\s*\[\s*\{.*?\}\s*\]',
            
            # Any remaining array-like JSON structures  
            r'\[\s*\{[^{}]*["\'][^"\']*["\']\s*:[^}]*\}\s*\]',
            
            # Enhanced patterns for the specific test case
            r'\[\s*\{\s*"type":\s*"reasoning"[^]]*\]\s*,?\s*"thinking":[^}]*\}[^]]*\]',
            r'"summary":\s*\[\s*\{[^]]*\]\s*,?\s*"thinking":[^}]*',
            r'"\s*,?\s*"thinking":\s*"[^"]*"[^}]*\}[^]]*\]',
            
            # Remove reasoning block prefixes
            r'reasoning["\']?\s*:\s*\[',
            r'summary["\']?\s*:\s*\[',
            
            # Remove standalone JSON fragments
            r'\{[^{}]*"summary"[^{}]*\}',
            r'\{[^{}]*"reasoning"[^{}]*\}',
            r'\{[^{}]*"text"[^{}]*\}',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Additional cleanup for the specific issue pattern
        # Remove any line that starts with | | and contains JSON
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            
            # Skip lines that are clearly JSON artifacts in table format
            if stripped.startswith('| |') and ('{' in line or '[' in line or '"' in line):
                continue
                
            # Skip lines that are pure JSON
            if (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']')):
                continue
                
            # Skip lines with only JSON fragments or punctuation
            if not stripped or stripped in ['| |', '| | },', '| | }, ]', '},', '}]', ']', ',', '"}]']:
                continue
                
            # Skip lines that contain fragments of the specific problematic pattern
            if any(fragment in stripped for fragment in [
                ', "thinking": "', '"thinking": "', '"}]', '"}, "thinking"',
                'type": "reasoning"', 'type": "summary_text"'
            ]):
                continue
                
            # Skip lines that are mostly JSON punctuation with little meaningful text
            if len(stripped) < 15:
                json_chars = sum(1 for char in stripped if char in '{}[]",:')
                alpha_chars = sum(1 for char in stripped if char.isalpha())
                if json_chars > alpha_chars and json_chars > 3:
                    continue
                    
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)


# Global instance for convenient access
global_content_sanitizer = ContentSanitizer(strict_mode=True, preserve_metadata=True)


def sanitize_agent_content(content: str) -> SanitizationResult:
    """
    Convenience function for agent content sanitization.
    
    This is the main entry point for agent components to clean their output.
    
    Args:
        content: Raw content from agent that may contain JSON structures
        
    Returns:
        SanitizationResult with clean markdown content and extracted metadata
    """
    return global_content_sanitizer.sanitize_content(content)


def extract_clean_markdown(content: str) -> str:
    """
    Simple helper to extract only clean markdown content.
    
    Args:
        content: Raw content that may contain JSON structures
        
    Returns:
        Clean markdown content only
    """
    result = global_content_sanitizer.sanitize_content(content)
    return result.clean_content


def has_mixed_content(content: str) -> bool:
    """
    Check if content contains mixed JSON and markdown.
    
    Args:
        content: Content to check
        
    Returns:
        True if content contains both JSON and markdown
    """
    result = global_content_sanitizer.sanitize_content(content)
    return result.content_type == ContentType.MIXED_CONTENT