"""
Generic Pydantic models for structured report generation.

These models provide abstract, reusable structures for any type of research report,
not tied to specific query types (comparisons, analyses, etc.).

The key insight: LLM generates structured data, we render perfect markdown.
This guarantees 100% correct formatting with no validation loops needed.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class TableCell(BaseModel):
    """
    Generic table cell - works for any content type.

    Supports footnote markers for academic-style references.
    """
    content: str = Field(..., description="Cell content (text, numbers, etc.)")
    footnote_marker: Optional[str] = Field(
        None,
        description="Optional footnote reference like '*¹', '*²', etc."
    )
    alignment: Optional[Literal["left", "center", "right"]] = Field(
        "left",
        description="Text alignment within cell (default: left)"
    )


class TableRow(BaseModel):
    """
    Generic table row containing cells.

    Supports different row types to handle headers, data, and footers.
    """
    cells: List[TableCell] = Field(..., description="List of cells in this row")
    row_type: Optional[Literal["header", "data", "footer"]] = Field(
        "data",
        description="Type of row: header (with separator), data (normal), or footer"
    )


class Table(BaseModel):
    """
    Generic table structure - NOT tied to comparisons or specific formats.

    Works for:
    - Comparison tables (Country vs Tax Rate)
    - Feature matrices (Product vs Features)
    - Timeline tables (Date vs Event)
    - Ranking tables (Rank vs Company vs Score)
    - Any other tabular data

    The renderer automatically creates perfect markdown with:
    - Proper pipe delimiters
    - Separator rows after headers
    - Footnote placement
    """
    caption: Optional[str] = Field(
        None,
        description="Optional table title/caption rendered as ### heading"
    )
    rows: List[TableRow] = Field(
        ...,
        description="Table rows - first row typically contains headers"
    )
    footnotes: Optional[List[str]] = Field(
        None,
        description="List of footnote definitions like '*¹ Madrid center'"
    )

    @property
    def column_count(self) -> int:
        """Infer column count from first row."""
        return len(self.rows[0].cells) if self.rows else 0

    def model_post_init(self, __context):
        """Validate table structure after initialization."""
        if not self.rows:
            return

        # Ensure all rows have same number of cells
        expected_cols = self.column_count
        for i, row in enumerate(self.rows):
            if len(row.cells) != expected_cols:
                raise ValueError(
                    f"Row {i} has {len(row.cells)} cells, expected {expected_cols}"
                )


class ContentBlock(BaseModel):
    """
    Generic content block - can be text, table, or list.

    Allows mixing different content types within a section:
    - Paragraph → narrative text
    - Table → structured data
    - Bullet list → key points
    - Numbered list → sequential steps
    """
    block_type: Literal["paragraph", "table", "bullet_list", "numbered_list"] = Field(
        ...,
        description="Type of content block"
    )

    # Only ONE of these should be populated based on block_type
    text: Optional[str] = Field(
        None,
        description="Text content for paragraph block_type"
    )
    table: Optional[Table] = Field(
        None,
        description="Table data for table block_type"
    )
    list_items: Optional[List[str]] = Field(
        None,
        description="List items for bullet_list or numbered_list block_type"
    )

    def model_post_init(self, __context):
        """Validate that correct field is populated for block_type."""
        if self.block_type == "paragraph" and not self.text:
            raise ValueError("paragraph block_type requires text field")
        elif self.block_type == "table" and not self.table:
            raise ValueError("table block_type requires table field")
        elif self.block_type in ["bullet_list", "numbered_list"] and not self.list_items:
            raise ValueError(f"{self.block_type} requires list_items field")


class ReportSection(BaseModel):
    """
    Generic report section - works for ANY report style.

    Supports:
    - Flat sections (just title + content blocks)
    - Nested sections (subsections for hierarchical organization)
    - Mixed content (paragraphs, tables, lists in any order)
    """
    title: str = Field(..., description="Section heading")
    content_blocks: List[ContentBlock] = Field(
        ...,
        description="Ordered list of content blocks (paragraphs, tables, lists)"
    )
    subsections: Optional[List[ReportSection]] = Field(
        None,
        description="Optional nested subsections for hierarchical structure"
    )


class StructuredReport(BaseModel):
    """
    Complete structured report - generic format for ALL query types.

    Standard structure:
    1. Title (# heading)
    2. Key Points (executive summary bullets)
    3. Overview (opening paragraph)
    4. Sections (main content with mixed blocks)
    5. References (citations list)
    6. Appendix (optional supplementary material)

    This structure works for:
    - Comparison reports (tax, products, services)
    - Analysis reports (market, technology, industry)
    - Summary reports (literature reviews, research summaries)
    - Technical reports (documentation, specifications)
    - Any other research report type
    """
    title: str = Field(..., description="Report title (rendered as # heading)")

    key_points: List[str] = Field(
        ...,
        description="3-5 main takeaways for executive summary"
        # Note: min_items/max_items not supported by Databricks endpoint
    )

    overview: str = Field(
        ...,
        description="Opening paragraph that frames the research question"
    )

    sections: List[ReportSection] = Field(
        ...,
        description="Main report sections with mixed content blocks"
        # Note: min_items not supported by Databricks endpoint
    )

    references: List[str] = Field(
        default_factory=list,
        description="Citations in markdown format: '- [Title](URL)'"
    )

    appendix: Optional[str] = Field(
        None,
        description="Optional supplementary material, raw data, or methodology notes"
    )

    class Config:
        """Pydantic model configuration."""
        # Databricks endpoint doesn't support additionalProperties in JSON Schema
        extra = "forbid"  # Changed from "allow" to avoid additionalProperties keyword
        # Validate on assignment for runtime safety
        validate_assignment = True