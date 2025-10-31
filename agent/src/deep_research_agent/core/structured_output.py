"""
Single source of truth for all LLM structured output handling.

This module provides a unified API for parsing LLM responses across all formats:
- Databricks structured responses (list format with reasoning/text blocks)
- Plain text responses
- Markdown-wrapped JSON
- JSON with syntax errors (with repair)
- Pydantic model validation

This replaces duplicated logic across:
- response_handlers.py (text extraction)
- json_utils.py (JSON parsing with repair)
- llm_response_parser.py (compatibility wrapper)
"""

from typing import Any, Optional, Type, TypeVar, Union, Dict, List
from pydantic import BaseModel
import json
from . import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredOutputParser:
    """
    Unified parser for all LLM responses.

    This class handles the complete pipeline:
    1. Extract text from structured responses (Databricks list format)
    2. Clean markdown artifacts (code fences, etc.)
    3. Parse JSON with optional repair
    4. Validate against Pydantic schema (if provided)
    """

    @staticmethod
    def parse(
        response: Any,
        schema: Optional[Type[T]] = None,
        repair_json: bool = True,
        return_text_if_not_json: bool = False,
    ) -> Union[str, T, dict, list]:
        """
        Parse LLM response and optionally validate against schema.

        Args:
            response: Raw LLM response (ChatModel output, string, list, dict)
            schema: Optional Pydantic model to validate against
            repair_json: Whether to attempt JSON repair on parse errors
            return_text_if_not_json: If True, return text when JSON parsing fails

        Returns:
            - If schema provided: Validated Pydantic instance
            - If response contains JSON: Parsed dict/list
            - If return_text_if_not_json: Extracted text string
            - Otherwise: Raises ValueError

        Examples:
            # Extract and parse JSON
            data = StructuredOutputParser.parse(response)

            # Extract, parse, and validate
            result = StructuredOutputParser.parse(
                response,
                schema=ResearchSynthesis
            )

            # Get text if not JSON
            text = StructuredOutputParser.parse(
                response,
                return_text_if_not_json=True
            )
        """
        # Step 1: Extract text from structured responses
        text = _extract_text(response)

        # Step 2: Clean markdown artifacts
        text = _clean_markdown(text)

        # If no schema requested and not forcing JSON, try to parse as JSON
        if schema is None and not return_text_if_not_json:
            # Try to parse as JSON (user might want dict/list)
            parsed = _parse_json(text, repair=repair_json)
            return parsed if parsed is not None else text

        # Step 3: Parse JSON
        parsed = _parse_json(text, repair=repair_json)

        # Handle parse failure
        if parsed is None:
            if return_text_if_not_json:
                logger.debug("JSON parsing failed, returning text")
                return text

            error_msg = (
                f"LLM did not return valid JSON. "
                f"Received text (first 500 chars): {text[:500]}"
            )
            if schema:
                error_msg = f"Expected schema: {schema.__name__}. " + error_msg
            raise ValueError(error_msg)

        # If no schema, return parsed JSON
        if schema is None:
            return parsed

        # Step 4: Validate parsed data is a dict (required for Pydantic schemas)
        # NO NORMALIZATION - if model returns wrong type, that's an error!
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected dict for schema {schema.__name__}, got {type(parsed).__name__}. "
                f"This indicates the model did not follow the schema. "
                f"Parsed data (first 500 chars): {str(parsed)[:500]}"
            )

        # ✅ NEW: Log parsed data structure before validation (DEBUG level for diagnostic)
        logger.debug(
            f"🔍 STRUCTURED VALIDATION | Schema: {schema.__name__}\n"
            f"   - Parsed type: {type(parsed).__name__}\n"
            f"   - Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}\n"
            f"   - Parsed preview: {str(parsed)[:300]}..."
        )

        # Step 5: Validate against schema (strict validation, no normalization)
        try:
            validated = schema(**parsed)
            logger.info(f"✅ VALIDATION SUCCESS | Schema: {schema.__name__}")

            # ✅ NEW: Log EntityMetricsOutput specifics
            if hasattr(validated, 'extracted_values'):
                extracted = getattr(validated, 'extracted_values', {})
                entity = getattr(validated, 'entity', 'unknown')
                logger.info(
                    f"   -> EntityMetricsOutput validated: entity={entity}, "
                    f"metrics={len(extracted)}, keys={list(extracted.keys())[:10]}"
                )

            return validated
        except Exception as e:
            logger.error(
                f"❌ VALIDATION FAILED | Schema: {schema.__name__}\n"
                f"   - Error: {type(e).__name__}: {str(e)[:200]}\n"
                f"   - Parsed data type: {type(parsed).__name__}\n"
                f"   - Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}"
            )
            raise ValueError(
                f"Schema validation failed for {schema.__name__}: {e}\n"
                f"Parsed data type: {type(parsed)}\n"
                f"Parsed data (first 500 chars): {str(parsed)[:500]}"
            )

    @staticmethod
    def extract_text(response: Any) -> str:
        """
        Extract plain text only, no JSON parsing.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned text string
        """
        text = _extract_text(response)
        return _clean_markdown(text)

    @staticmethod
    def extract_json(response: Any, repair: bool = True) -> Optional[Union[dict, list]]:
        """
        Extract and parse JSON, returning None if not valid JSON.

        Args:
            response: Raw LLM response
            repair: Whether to attempt JSON repair

        Returns:
            Parsed dict/list or None
        """
        text = _extract_text(response)
        text = _clean_markdown(text)
        return _parse_json(text, repair=repair)


def _extract_text(response: Any) -> str:
    """
    Extract text from Databricks structured responses or plain strings.

    Handles:
    - Databricks list format: [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]
    - Plain strings
    - Objects with .content attribute
    """
    # Import here to avoid circular dependency
    from .response_handlers import extract_text_from_response
    return extract_text_from_response(response)


def _clean_markdown(text: str) -> str:
    """
    Remove markdown code fences and clean formatting.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - Generic code blocks
    """
    if not text:
        return text

    text = text.strip()

    # Remove markdown code blocks
    if "```json" in text:
        # Extract content between ```json and ```
        parts = text.split("```json", 1)
        if len(parts) > 1:
            rest = parts[1].split("```", 1)
            if len(rest) > 0:
                text = rest[0]
    elif text.startswith("```") and "```" in text[3:]:
        # Generic code block ```language\n...\n```
        parts = text.split("```")
        if len(parts) >= 3:
            # Get content between first and second ```
            content = parts[1]
            # Skip language identifier line if present
            if "\n" in content:
                lines = content.split("\n", 1)
                # If first line is short (likely language id), skip it
                if len(lines[0].strip()) < 20:
                    text = lines[1] if len(lines) > 1 else lines[0]
                else:
                    text = content
            else:
                text = content

    return text.strip()


# ===== Schema-Aware Response Normalization =====
# These functions provide abstract, reusable conversion logic
# for ANY Pydantic model, not model-specific hacks.

def _get_type_origin(annotation):
    """Get the origin of a type annotation (e.g., list from List[str])."""
    import typing
    if hasattr(typing, 'get_origin'):
        return typing.get_origin(annotation)
    # Fallback for older Python
    return getattr(annotation, '__origin__', None)


def _get_type_args(annotation):
    """Get the args of a type annotation (e.g., [str] from List[str])."""
    import typing
    if hasattr(typing, 'get_args'):
        return typing.get_args(annotation)
    # Fallback for older Python
    return getattr(annotation, '__args__', ())


# ==============================================================================
# DEPRECATED NORMALIZATION FUNCTIONS (NO LONGER USED)
# ==============================================================================
# These functions attempted to "fix" LLM responses that didn't follow the schema.
# This approach was removed because:
# 1. It masked the real problem (model not following schema)
# 2. Made debugging harder (unclear why validation failed)
# 3. Added complexity without solving root cause
#
# New approach: Fail fast when schema not followed, use comprehensive logging
# to diagnose the issue, then fix the prompt or schema.
#
# Functions kept for reference only. Not called by parse() anymore.
# ==============================================================================


def _is_table_schema(schema: Type[BaseModel]) -> bool:
    """
    Check if schema represents a table structure.

    A table schema has:
    - headers: List[str] field
    - rows: List[List[str]] or List[List[Any]] field

    Returns:
        True if this is a table schema, False otherwise

    Examples:
        TableBlock → True (has headers and rows)
        CoordinatorDecision → False (no table structure)
    """
    if not hasattr(schema, '__fields__'):
        return False

    fields = schema.__fields__
    has_headers = False
    has_rows = False

    for field_name, field_info in fields.items():
        annotation = field_info.annotation
        origin = _get_type_origin(annotation)
        args = _get_type_args(annotation)

        # Check for headers: List[str]
        if field_name == 'headers' and origin is list and args and args[0] is str:
            has_headers = True
            logger.debug(f"Found headers field: List[str]")

        # Check for rows: List[List[...]]
        if field_name == 'rows' and origin is list and args:
            inner_origin = _get_type_origin(args[0])
            if inner_origin is list:
                has_rows = True
                logger.debug(f"Found rows field: List[List[...]]")

    is_table = has_headers and has_rows
    if is_table:
        logger.info(f"✅ Detected table schema: {schema.__name__}")
    return is_table


def _infer_column_count(strings: list) -> Optional[int]:
    """
    Infer number of table columns from flat string list.

    Args:
        strings: Flat list of string values (all table cells in sequence)

    Returns:
        Number of columns, or None if cannot be inferred

    Algorithm:
        - Try column counts 2-20
        - Find count where total_strings % cols == 0 and creates ≥2 rows
        - Fallback: allow small remainder for metadata

    Examples:
        62 strings → try 6 cols: 62 % 6 != 0, reject
        60 strings → try 6 cols: 60 % 6 == 0, 10 rows ✓
    """
    total = len(strings)
    if total < 2:  # Need at least headers
        logger.warning(f"Cannot infer columns from {total} strings (need ≥2)")
        return None

    # Try column counts from 2 to 20 (exact division)
    for cols in range(2, min(21, total + 1)):
        if total % cols == 0:
            rows = total // cols
            if rows >= 2:  # At least header + 1 data row
                logger.info(f"✅ Inferred {cols} columns × {rows} rows (1 header + {rows-1} data rows)")
                return cols

    # Fallback: try to find best fit even with small remainder
    # (Some LLMs add extra metadata items)
    for cols in range(2, min(21, total + 1)):
        remainder = total % cols
        if remainder <= 2:  # Allow 1-2 extra items
            rows = total // cols
            if rows >= 2:
                logger.info(f"⚠️  Inferred {cols} columns with {remainder} extra items (will be ignored)")
                return cols

    logger.error(f"❌ Cannot infer column count from {total} strings")
    return None


def _restructure_flat_table(data: list, schema: Type[BaseModel]) -> dict:
    """
    Convert flat list of table data into headers + rows structure.

    Handles LLM responses that return table cells as a flat list:
    ['Col1', 'Col2', 'Col3',     # Headers (first N items)
     'Val1', 'Val2', 'Val3',     # Row 1 (next N items)
     'Val4', 'Val5', 'Val6',     # Row 2 (next N items)
     ...
     {'caption': '...'}]         # Optional metadata dict at end

    Args:
        data: Flat list from LLM (mix of strings and optional dicts)
        schema: Target Pydantic model (must be table schema)

    Returns:
        Dict with 'headers' and 'rows' keys, plus any dict fields

    Examples:
        Input:  ['A', 'B', 'C', 'x', 'y', 'z', {'caption': 'Table 1'}]
        Output: {'headers': ['A', 'B', 'C'],
                 'rows': [['x', 'y', 'z']],
                 'caption': 'Table 1'}
    """
    if not data:
        logger.warning("Empty data list for table restructuring")
        return {}

    result = {}

    # Separate strings from dicts
    strings = [x for x in data if isinstance(x, str)]
    dicts = [x for x in data if isinstance(x, dict)]

    logger.info(f"📊 Restructuring flat table: {len(strings)} strings, {len(dicts)} dicts")

    # Infer column count from string data
    num_cols = _infer_column_count(strings)
    if not num_cols:
        logger.error(f"❌ Cannot restructure table: failed to infer column count")
        # Fallback: return as-is and let validation fail with clear error
        return {'headers': [], 'rows': strings}

    # Split into headers (first row) and data rows (remaining)
    headers = strings[:num_cols]
    remaining = strings[num_cols:]

    logger.debug(f"Headers: {headers}")

    # Chunk remaining strings into rows of num_cols
    rows = []
    for i in range(0, len(remaining), num_cols):
        row = remaining[i:i+num_cols]
        if len(row) == num_cols:  # Only add complete rows
            rows.append(row)
            logger.debug(f"Row {len(rows)}: {len(row)} cells")
        else:
            logger.warning(f"⚠️  Incomplete row with {len(row)} cells (expected {num_cols}), skipping")

    result['headers'] = headers
    result['rows'] = rows

    logger.info(f"✅ Restructured: {len(headers)} headers, {len(rows)} data rows")

    # Merge any dict fields (like caption) into result
    for d in dicts:
        for k, v in d.items():
            if k in schema.__fields__ and k not in result:
                result[k] = v
                logger.debug(f"Added field '{k}' from metadata dict")

    return result


def normalize_for_schema(parsed: Any, schema: Type[BaseModel]) -> dict:
    """
    Intelligently convert parsed JSON to match Pydantic schema structure.

    This is a **generic, reusable** function that works for ANY BaseModel:
    - TableBlock, CoordinatorDecision, or any future model
    - Handles list→dict, missing fields, type mismatches
    - Uses schema introspection, not hardcoded logic

    Args:
        parsed: Parsed JSON (can be dict, list, or primitive)
        schema: Target Pydantic BaseModel class

    Returns:
        Dictionary that can be unpacked into schema(**result)

    Examples:
        # List with mixed types → dict
        ['header1', 'header2', {'rows': [...]}] → {'headers': [...], 'rows': [...]}

        # List of dicts → merged dict
        [{'action': 'SUFFICIENT'}, {'reasoning': '...'}] → {'action': 'SUFFICIENT', 'reasoning': '...'}
    """
    if isinstance(parsed, dict):
        return parsed  # Already correct format

    if isinstance(parsed, list):
        logger.info(f"Normalizing list response for schema {schema.__name__}")
        return _convert_list_to_dict(parsed, schema)

    # Fallback: wrap primitive in dict (may fail validation, but that's expected)
    logger.warning(f"Cannot normalize {type(parsed).__name__} for schema {schema.__name__}, wrapping in dict")
    return {"value": parsed}


def _convert_list_to_dict(data: list, schema: Type[BaseModel]) -> dict:
    """
    Generic list-to-dict converter using schema introspection.

    Implements multiple conversion strategies:
    1. All dicts → merge them
    2. Mixed types → match by field type annotations
    3. Homogeneous types → assign to first matching field

    Args:
        data: List to convert
        schema: Target Pydantic model

    Returns:
        Dict matching schema structure
    """
    if not data:
        return {}

    # CRITICAL: Check if this is a table schema (headers + rows)
    # If so, restructure flat list before type-based matching
    # This handles the common case where LLMs return table cells as a flat list
    if _is_table_schema(schema):
        logger.info(f"🔧 Detected table schema, attempting flat list restructuring")
        return _restructure_flat_table(data, schema)

    fields = schema.__fields__ if hasattr(schema, '__fields__') else {}
    result = {}

    # Strategy 1: List of dicts → merge them
    # Example: [{'rows': [...]}, {'caption': '...'}] → {'rows': [...], 'caption': '...'}
    if all(isinstance(item, dict) for item in data):
        for item in data:
            result.update(item)
        logger.debug(f"Merged {len(data)} dicts into single dict")
        return result

    # Strategy 2: Mixed types → type-based matching
    # Example: ['header1', 'header2', {'rows': [...]}] for TableBlock
    #   → {'headers': ['header1', 'header2'], 'rows': [...]}

    # Collect items by type
    strings = [x for x in data if isinstance(x, str)]
    dicts = [x for x in data if isinstance(x, dict)]
    lists = [x for x in data if isinstance(x, list)]
    numbers = [x for x in data if isinstance(x, (int, float))]

    # Match to schema fields by type annotation
    for field_name, field_info in fields.items():
        annotation = field_info.annotation
        origin = _get_type_origin(annotation)
        args = _get_type_args(annotation)

        # List[str] fields get string items
        if origin is list and args and args[0] is str and strings:
            result[field_name] = strings.copy()
            logger.debug(f"Assigned {len(strings)} strings to {field_name}")
            strings = []  # Consume

        # List[List[str]] fields (nested lists like 'rows') - check dicts for nested data
        elif origin is list and args and _get_type_origin(args[0]) is list:
            # Look for dict with matching key or 'rows' key
            for d in dicts[:]:
                if field_name in d:
                    result[field_name] = d[field_name]
                    dicts.remove(d)
                    logger.debug(f"Extracted {field_name} from dict")
                    break
                elif 'rows' in d and field_name == 'rows':
                    result[field_name] = d['rows']
                    dicts.remove(d)
                    logger.debug(f"Extracted rows from dict")
                    break

        # Dict or BaseModel fields get remaining dicts
        elif (origin is dict or (isinstance(annotation, type) and issubclass(annotation, BaseModel))) and dicts:
            result[field_name] = dicts[0]
            dicts.pop(0)
            logger.debug(f"Assigned dict to {field_name}")

        # str fields without default - try to extract from remaining strings
        elif annotation is str and strings and field_info.is_required():
            result[field_name] = strings[0]
            strings.pop(0)
            logger.debug(f"Assigned string to required field {field_name}")

    # If we have leftover dicts with known keys, merge them
    for d in dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                if k in fields and k not in result:
                    result[k] = v

    logger.info(f"Converted list to dict with {len(result)} fields for {schema.__name__}")
    return result


def _parse_json(text: str, repair: bool = True) -> Optional[Union[dict, list]]:
    """
    Parse JSON with optional repair.

    Args:
        text: Text to parse as JSON
        repair: Whether to attempt JSON repair on failure

    Returns:
        Parsed dict/list or None if parsing fails
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        logger.debug(f"✅ JSON parsed successfully: type={type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed: {e}")

        if not repair:
            return None

        # Try repair
        try:
            from json_repair import repair_json
            logger.debug("Attempting JSON repair...")
            repaired = repair_json(text)
            parsed = json.loads(repaired)
            logger.info(f"✅ JSON repaired and parsed successfully")
            return parsed
        except ImportError:
            logger.debug("json_repair not available, cannot repair")
            return None
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None


# Global instance for convenience
_parser = StructuredOutputParser()


# Convenience functions (backward compatible API)

def parse_llm_response(
    response: Any,
    schema: Optional[Type[T]] = None,
    repair_json: bool = True,
) -> Union[str, T, dict, list]:
    """
    Parse LLM response with optional schema validation.

    This is the main entry point for most use cases.

    Args:
        response: Raw LLM response
        schema: Optional Pydantic model to validate against
        repair_json: Whether to attempt JSON repair

    Returns:
        Validated Pydantic instance, dict/list, or string

    Examples:
        # Parse JSON response
        data = parse_llm_response(response)

        # Parse and validate
        result = parse_llm_response(response, schema=ResearchSynthesis)
    """
    return _parser.parse(response, schema=schema, repair_json=repair_json)


def extract_text(response: Any) -> str:
    """
    Extract plain text only, no JSON parsing.

    Args:
        response: Raw LLM response

    Returns:
        Cleaned text string
    """
    return _parser.extract_text(response)


def extract_json(response: Any, repair: bool = True) -> Optional[Union[dict, list]]:
    """
    Extract and parse JSON, returning None if not valid JSON.

    Args:
        response: Raw LLM response
        repair: Whether to attempt JSON repair

    Returns:
        Parsed dict/list or None
    """
    return _parser.extract_json(response, repair=repair)


# Re-export for backward compatibility
def parse_structured_response(response: Any, schema: Optional[Type[T]] = None) -> Union[str, T, dict]:
    """Backward compatible alias for parse_llm_response."""
    return parse_llm_response(response, schema=schema)
