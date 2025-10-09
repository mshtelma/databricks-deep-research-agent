"""
TRUE structured report generation with programmatic table rendering.

This model guarantees correct markdown table syntax by having the LLM generate
structured data (headers, rows, cells) and rendering programmatically.

Key features:
- Explicit separation of paragraphs and tables
- Programmatic table rendering (separator rows guaranteed!)
- Column count validation and padding
- No markdown syntax errors possible
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Dict, Any, Type
import json


# ===== ENHANCED TABLE MODELS WITH CELL-LEVEL DERIVATION TRACKING =====
# These models force LLM to justify values before generating them


class TableCell(BaseModel):
    """
    A single table cell with mandatory derivation tracking.

    CRITICAL: derivation field comes FIRST in schema to force LLM reasoning.
    This creates a "chain of thought" pattern that reduces hallucinations.
    """
    derivation: str = Field(
        ...,
        description=(
            "REQUIRED: Explain how this value was obtained. Options:\n"
            "- 'extracted: [source observation text or number]' for direct data from research\n"
            "- 'calculated: [formula with source values]' for derived values (e.g., '$8,050 calculated from $23,000 × 0.35')\n"
            "- 'not_available' if data is genuinely missing\n"
            "- 'estimated: [basis and assumptions]' for approximations (use sparingly)"
        )
    )
    value: str = Field(
        ...,
        description="The actual cell content to display in the table"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence level in this value based on source quality"
    )

    class Config:
        extra = "forbid"


class TableRow(BaseModel):
    """
    A table row consisting of multiple cells with derivation tracking.
    """
    cells: List[TableCell] = Field(
        ...,
        description="List of cells in this row (must match header count)"
    )

    class Config:
        extra = "forbid"


class TableBlockWithTrackedCells(BaseModel):
    """
    Enhanced table with cell-level derivation tracking (NEW - v2).

    This model ensures every single cell value has a documented source/reasoning,
    dramatically reducing hallucinations by forcing "proof of work" for each value.
    """
    type: Literal["table"] = "table"
    headers: List[str] = Field(..., description="Column headers")
    rows: List[TableRow] = Field(..., description="Table rows with tracked cells")
    caption: str = Field("", description="Optional table caption")

    class Config:
        extra = "forbid"

    def render_markdown(self) -> str:
        """
        Render table with smart footnote markers for derived/estimated values.

        Returns:
            Markdown table with:
            - † markers for calculated values
            - ‡ markers for estimated values
            - * markers for low confidence
            - Footnote legend at bottom
        """
        if not self.headers or not self.rows:
            return ""

        num_cols = len(self.headers)
        lines = []
        footnote_markers = {}

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")

        # Separator row (GUARANTEED!)
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        # Data rows with smart footnote markers
        for row in self.rows:
            rendered_cells = []
            for cell in row.cells[:num_cols]:  # Ensure column count
                value = cell.value

                # Add footnote markers for non-extracted values
                if cell.derivation.startswith("calculated:"):
                    if "†" not in footnote_markers:
                        footnote_markers["†"] = "Calculated from source data"
                    value = f"{value}†"
                elif cell.derivation.startswith("estimated:"):
                    if "‡" not in footnote_markers:
                        footnote_markers["‡"] = "Estimated value (use with caution)"
                    value = f"{value}‡"
                elif cell.confidence == "low":
                    if "*" not in footnote_markers:
                        footnote_markers["*"] = "Low confidence"
                    value = f"{value}*"

                rendered_cells.append(value)

            # Pad if needed to match column count
            while len(rendered_cells) < num_cols:
                rendered_cells.append("")

            lines.append("| " + " | ".join(rendered_cells) + " |")

        # Add footnotes if any markers were used
        if footnote_markers:
            lines.append("")
            for marker, description in sorted(footnote_markers.items()):
                lines.append(f"_{marker} {description}_")

        # Add caption if present
        if self.caption:
            lines.append("")
            lines.append(f"*{self.caption}*")

        return "\n".join(lines)


# ===== SIMPLIFIED BLOCK MODELS FOR MULTI-CALL GENERATION =====
# These separate models avoid the validation issues with combined ContentBlock


class ParagraphBlock(BaseModel):
    """
    A text paragraph block (no default values, no unused fields).

    Used for generating text content separately from tables.
    """
    type: Literal["paragraph"] = "paragraph"
    text: str = Field(..., description="Paragraph text with markdown formatting (bold, italic, etc.)")

    class Config:
        extra = "forbid"


class TableBlock(BaseModel):
    """
    A table block with programmatic markdown rendering (GUARANTEED separator rows).

    Used for generating structured table data that renders to perfect markdown.
    LLM generates simple list-of-lists structure, we handle rendering.
    """
    type: Literal["table"] = "table"
    headers: List[str] = Field(..., description="Column headers")
    rows: List[List[str]] = Field(..., description="Rows as list of string lists (simplified from cell objects)")
    caption: str = Field("", description="Optional table caption (empty string if none)")

    class Config:
        extra = "forbid"

    @field_validator('rows')
    @classmethod
    def validate_row_lengths(cls, rows: List[List[str]], info) -> List[List[str]]:
        """Ensure all rows have the same number of cells as headers."""
        if not rows:
            return rows

        headers = info.data.get('headers', [])
        expected_cols = len(headers) if headers else 0

        if expected_cols == 0:
            return rows

        # Pad or trim each row to match header count
        validated_rows = []
        for row in rows:
            # Trim excess cells
            trimmed = row[:expected_cols]

            # Pad with empty strings if needed
            while len(trimmed) < expected_cols:
                trimmed.append("")

            validated_rows.append(trimmed)

        return validated_rows

    @field_validator('headers', 'rows', 'caption', mode='before')
    @classmethod
    def clean_unicode_padding(cls, v, info):
        """
        Clean unicode padding when TableBlock is created (defense in depth).

        This provides an additional layer of protection against unicode bloat,
        cleaning the data before it even reaches render_markdown().
        """
        def clean_text(text):
            if not isinstance(text, str):
                return str(text) if text is not None else ""

            # Remove main culprit unicode spaces
            for char in ['\u202f', '\u2009', '\u200a', '\u00a0', '\u200b']:
                text = text.replace(char, ' ')

            # Normalize whitespace
            return ' '.join(text.split())

        field_name = info.field_name

        if field_name == 'rows' and isinstance(v, list):
            if v and isinstance(v[0], list):  # List of lists (rows)
                return [[clean_text(cell) for cell in row] for row in v]
        elif field_name == 'headers' and isinstance(v, list):
            return [clean_text(item) for item in v]
        elif field_name == 'caption' and isinstance(v, str):
            return clean_text(v)

        return v

    def _clean_cell_content(self, text: str) -> str:
        """
        Remove unicode padding and normalize whitespace in table cells.

        Fixes the critical bug where LLMs generate massive invisible unicode
        padding (e.g., \\u202f) to "visually align" cells, creating bloated reports.

        Args:
            text: Raw cell content from LLM

        Returns:
            Cleaned cell content with normalized whitespace
        """
        if not text:
            return ""

        # Remove ALL problematic unicode spaces that LLMs use for padding
        unicode_spaces = [
            '\u202f',  # Narrow no-break space (main culprit - 3.7M occurrences!)
            '\u2009',  # Thin space
            '\u200a',  # Hair space
            '\u00a0',  # Non-breaking space
            '\u2000', '\u2001', '\u2002', '\u2003', '\u2004',
            '\u2005', '\u2006', '\u2007', '\u2008',  # Various width spaces
            '\u200b',  # Zero-width space
        ]

        cleaned = text
        for space in unicode_spaces:
            cleaned = cleaned.replace(space, ' ')

        # Normalize multiple spaces to single space
        cleaned = ' '.join(cleaned.split())

        # Safety limit on cell size to prevent bloat
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + '...'

        return cleaned

    def render_markdown(self) -> str:
        """
        Programmatically generate guaranteed-correct markdown table.

        The separator row is GUARANTEED to be present (no LLM can mess this up!).
        All cell content is cleaned to remove unicode padding bloat.

        Returns:
            Perfect markdown table string with separator row and clean cells
        """
        if not self.headers or not self.rows:
            return ""

        num_cols = len(self.headers)
        lines = []

        # Clean and render header row
        cleaned_headers = [self._clean_cell_content(h) for h in self.headers]
        header_line = "| " + " | ".join(cleaned_headers) + " |"
        lines.append(header_line)

        # Separator row (GUARANTEED!)
        separator_line = "| " + " | ".join(["---"] * num_cols) + " |"
        lines.append(separator_line)

        # Clean and render data rows
        for row in self.rows:
            # Ensure consistent column count (should already be validated)
            padded_row = (row + [""] * num_cols)[:num_cols]
            cleaned_row = [self._clean_cell_content(cell) for cell in padded_row]
            row_line = "| " + " | ".join(cleaned_row) + " |"
            lines.append(row_line)

        md = "\n".join(lines)

        # Add caption if present (also clean it)
        if self.caption:
            cleaned_caption = self._clean_cell_content(self.caption)
            md += f"\n\n*{cleaned_caption}*"

        return md


# Note: ReportSection is no longer used in multi-call generation approach.
# Each block (ParagraphBlock or TableBlock) is generated and rendered separately.


# ===== Schema Generation Helpers for Databricks Compatibility =====

def strip_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove all 'additionalProperties' keys from JSON schema.

    Databricks Foundation Models reject schemas containing 'additionalProperties',
    even when set to False. This is a known API limitation.

    Args:
        schema: JSON schema dict from Pydantic model_json_schema()

    Returns:
        Cleaned schema without any additionalProperties keys
    """
    if isinstance(schema, dict):
        # Remove additionalProperties at current level
        schema.pop('additionalProperties', None)

        # Recursively process nested objects
        for key, value in schema.items():
            if isinstance(value, dict):
                strip_additional_properties(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        strip_additional_properties(item)

    return schema


def inline_definitions(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inline $defs references directly into schema body.

    Databricks may reject schemas with $ref pointers to $defs.
    This function flattens the schema by replacing all $ref with actual definitions.

    Args:
        schema: JSON schema potentially containing $defs and $ref

    Returns:
        Flattened schema with all references inlined
    """
    if '$defs' not in schema:
        return schema

    # Extract definitions
    defs = schema.pop('$defs')

    def replace_refs(obj: Any) -> Any:
        """Recursively replace $ref with actual definition."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                # Extract definition name from reference path
                ref_path = obj['$ref']  # e.g., "#/$defs/TableCell"
                def_name = ref_path.split('/')[-1]

                if def_name in defs:
                    # Replace reference with actual definition (recursively process it too)
                    return replace_refs(defs[def_name].copy())
                else:
                    # Definition not found, return original
                    return obj
            else:
                # Recursively process nested dictionaries
                return {k: replace_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [replace_refs(item) for item in obj]
        else:
            return obj

    return replace_refs(schema)


def get_databricks_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate Databricks-compatible JSON schema from Pydantic model.

    Applies all necessary transformations for Databricks API compatibility:
    1. Removes 'additionalProperties' keywords (causes rejection)
    2. Inlines $defs to avoid $ref resolution issues
    3. Validates resulting schema structure

    This function uses the proven pattern from researcher.py which successfully
    generates ResearchSynthesis schemas for Databricks endpoints.

    Args:
        model_class: Pydantic BaseModel class to generate schema for

    Returns:
        Clean schema dict ready for use with Databricks with_structured_output

    Example:
        >>> schema = get_databricks_schema(StructuredReport)
        >>> structured_llm = llm.with_structured_output(
        ...     StructuredReport,
        ...     method="json_mode"
        ... )

    Raises:
        AssertionError: If generated schema is missing required fields
    """
    # Use Pydantic v2's model_json_schema() for clean schema generation
    raw_schema = model_class.model_json_schema()

    # Apply Databricks compatibility transformations
    cleaned = strip_additional_properties(raw_schema)
    flattened = inline_definitions(cleaned)

    # Validate schema has required structure
    assert 'type' in flattened, f"Schema for {model_class.__name__} must have 'type' field"
    assert 'properties' in flattened, f"Schema for {model_class.__name__} must have 'properties' field"

    # Log schema size for debugging
    schema_size = len(json.dumps(flattened))

    return flattened


class StructuredReport(BaseModel):
    """
    TRUE structured report with programmatic table rendering.

    All fields required (no Optional to avoid anyOf in JSON schema).
    Tables are guaranteed to have correct markdown syntax.
    """

    title: str = Field(..., description="Report title")

    key_points: List[str] = Field(..., description="3-5 main takeaways as bullet points")

    overview: str = Field(..., description="Opening paragraph framing the research question")

    sections: List[ReportSection] = Field(
        ...,
        description="Array of report sections (minimum 3, maximum 8). Each section has structured content blocks."
    )

    references: List[str] = Field(
        ...,
        description="Citations as markdown list items: '- [Title](URL)' (use empty list if none)"
    )

    appendix: str = Field(..., description="Supplementary material (use empty string if none)")

    class Config:
        extra = "forbid"
        validate_assignment = True

    @classmethod
    def get_databricks_schema(cls) -> Dict[str, Any]:
        """
        Get Databricks-compatible JSON schema for this model.

        Uses the global get_databricks_schema() helper to generate a clean schema
        without additionalProperties or $defs, ready for use with LangChain's
        with_structured_output method.

        This is the PROVEN PATTERN from researcher.py which successfully uses
        structured output for ResearchSynthesis models.

        Returns:
            Dict[str, Any]: Clean JSON schema for Databricks endpoints

        Example:
            >>> schema = StructuredReport.get_databricks_schema()
            >>> structured_llm = llm.with_structured_output(
            ...     StructuredReport,
            ...     method="json_mode"
            ... )
            >>> result = structured_llm.invoke(messages)
            >>> report = StructuredReport(**result) if isinstance(result, dict) else result
            >>> markdown = render_structured_report(report)
        """
        return get_databricks_schema(cls)


def render_structured_report(report: StructuredReport) -> str:
    """
    Render StructuredReport to markdown with programmatic table generation.

    Tables are guaranteed to have correct markdown syntax because we
    render them programmatically, not by trusting LLM-generated strings.
    """
    parts = []

    # Title
    parts.append(f"# {report.title}\n")

    # Key Points
    if report.key_points:
        parts.append("## Key Points\n")
        for point in report.key_points:
            parts.append(f"- {point}")
        parts.append("")

    # Overview
    if report.overview:
        parts.append("## Overview\n")
        parts.append(report.overview)
        parts.append("")

    # Sections with structured content blocks
    for section in report.sections:
        if not section.title:
            continue

        parts.append(f"## {section.title}\n")

        # Render each content block
        for block in section.content_blocks:
            block_md = block.render_markdown()
            if block_md:
                parts.append(block_md)
                parts.append("")  # Add spacing between blocks

    # References
    if report.references:
        parts.append("## References\n")
        for ref in report.references:
            if not ref.startswith("-"):
                ref = f"- {ref}"
            parts.append(ref)
        parts.append("")

    # Appendix
    if report.appendix:
        parts.append("## Appendix\n")
        parts.append(report.appendix)
        parts.append("")

    return "\n".join(parts)