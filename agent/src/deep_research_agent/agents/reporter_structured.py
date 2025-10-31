"""
Phase 2 & 3: Structured Report Generation with Programmatic Table Building

This module implements structured report generation that NEVER allows LLMs to
generate markdown tables directly. Tables are built programmatically from
structured data.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..core.report_generation.models import (
    CalculationContext,
    ComparisonEntry,
    Calculation,
    TableSpec,
    SectionDraft,
    TableRenderMetadata
)
from ..core.report_models import (
    Table,
    TableRow,
    TableCell,
    ContentBlock,
    ReportSection
)
from ..core.report_models_structured import StructuredReport
from ..core.multi_agent_state import EnhancedResearchState
from .reporter_instrumentation import ReporterInstrumentation

logger = logging.getLogger(__name__)


def _safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute from object or dict.

    Args:
        obj: Object or dict
        attr: Attribute name
        default: Default value if not found

    Returns:
        Attribute value or default
    """
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)


class SectionDraftResponse(BaseModel):
    """Response model for structured section generation."""
    sections: List[SectionDraft] = Field(
        ...,
        description="List of report sections with narrative and table references"
    )

    class Config:
        extra = "forbid"


class StructuredReportGenerator:
    """
    Generates reports using structured generation to prevent markdown table issues.

    Key principles:
    1. LLMs NEVER generate markdown table syntax
    2. Tables built programmatically from ComparisonEntry data
    3. Narrative and tables are separate
    4. All data is traceable to sources
    """

    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the structured report generator.

        Args:
            llm: Language model for narrative generation
            config: Optional configuration overrides
        """
        self.llm = llm
        self.config = config or {}
        self.instrumentation = ReporterInstrumentation()

    async def generate_holistic_report_structured(
        self,
        findings: Dict[str, Any],
        calc_context: CalculationContext,
        dynamic_sections: Optional[List[Dict[str, Any]]] = None
    ) -> SectionDraftResponse:
        """
        Generate narrative skeleton without inline tables (Phase 2).

        Tables are referenced by ID, not generated inline.

        Args:
            findings: Research findings
            calc_context: Calculation context with data and comparisons
            dynamic_sections: Optional sections from state plan

        Returns:
            Structured sections with table references
        """
        # Build prompt that emphasizes NO TABLES
        prompt = self._build_structured_prompt(
            findings,
            calc_context,
            dynamic_sections
        )

        # Use structured output to force JSON response
        # ✅ CRITICAL FIX: Use json_schema for strict schema enforcement (was json_mode)
        structured_llm = self.llm.with_structured_output(
            SectionDraftResponse,
            method="json_schema"
        )

        messages = [
            SystemMessage(content="""You are a research report writer.

CRITICAL RULES:
1. Return valid JSON matching SectionDraftResponse schema EXACTLY
2. NEVER include markdown tables or pipe characters (|) in paragraphs
3. Reference tables using table_refs field with IDs from Available Table Specifications
4. Write clear, concise narrative paragraphs
5. Each section should flow logically into the next"""),
            HumanMessage(content=prompt)
        ]

        try:
            result = await structured_llm.ainvoke(messages)

            # Ensure we have a SectionDraftResponse object
            if isinstance(result, dict):
                draft = SectionDraftResponse(**result)
            else:
                draft = result

            logger.info(f"Generated {len(draft.sections)} structured sections")

            # Validate no inline tables in paragraphs
            for section in draft.sections:
                for para in section.paragraphs:
                    if '|' in para and para.count('|') > 2:
                        logger.warning(
                            f"Potential inline table detected in section '{section.title}'"
                        )

            return draft

        except Exception as e:
            logger.error(f"Failed to generate structured sections: {e}")
            # Return minimal fallback structure
            return SectionDraftResponse(sections=[
                SectionDraft(
                    title="Executive Summary",
                    paragraphs=[
                        "Unable to generate full report due to processing error.",
                        f"Research topic: {findings.get('research_topic', 'Unknown')}"
                    ],
                    table_refs=[]
                )
            ])

    def _build_structured_prompt(
        self,
        findings: Dict[str, Any],
        calc_context: CalculationContext,
        dynamic_sections: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Build prompt for structured generation."""

        # Extract table IDs
        table_specs = _safe_get_attr(calc_context, 'table_specifications', [])
        table_ids = [
            spec.table_id if hasattr(spec, 'table_id') else spec.get('table_id', 'unknown')
            for spec in table_specs
        ]

        # Extract key insights
        summary_insights = _safe_get_attr(calc_context, 'summary_insights', [])
        insights = summary_insights[:5] if summary_insights else []

        # Build section plan
        if dynamic_sections:
            section_plan = json.dumps(dynamic_sections, indent=2)
        else:
            section_plan = """[
    {"title": "Executive Summary", "purpose": "Key findings and recommendations"},
    {"title": "Comparative Analysis", "purpose": "Detailed comparison with data"},
    {"title": "Conclusions", "purpose": "Final insights and next steps"}
]"""

        prompt = f"""Generate a research report following this EXACT structure.

RESEARCH TOPIC: {findings.get('research_topic', 'Research Analysis')}

KEY INSIGHTS:
{chr(10).join(f'- {insight}' for insight in insights)}

PLANNED SECTIONS:
{section_plan}

AVAILABLE TABLE SPECIFICATIONS (reference these by ID in table_refs):
{', '.join(table_ids) if table_ids else 'No tables available'}

AVAILABLE DATA POINTS: {len(_safe_get_attr(calc_context, 'extracted_data', []))}
CALCULATIONS PERFORMED: {len(_safe_get_attr(calc_context, 'calculations', []))}
COMPARISONS AVAILABLE: {len(_safe_get_attr(calc_context, 'key_comparisons', []))}

CRITICAL INSTRUCTIONS:
1. Generate JSON matching SectionDraftResponse schema
2. DO NOT create markdown tables with | characters
3. Reference tables using table_refs field (e.g., "table_refs": ["country_comparison"])
4. Write 2-4 paragraphs per section
5. Ensure narrative flows logically

Example response format:
{{
    "sections": [
        {{
            "title": "Executive Summary",
            "paragraphs": [
                "This research examines...",
                "Key findings include..."
            ],
            "table_refs": []
        }},
        {{
            "title": "Comparative Analysis",
            "paragraphs": [
                "The data reveals significant differences...",
                "When comparing the countries..."
            ],
            "table_refs": ["country_comparison"]
        }}
    ]
}}"""

        return prompt

    def build_table_from_comparisons(
        self,
        table_spec: TableSpec,
        comparisons: List[ComparisonEntry],
        calculations: List[Calculation]
    ) -> Tuple[Table, TableRenderMetadata]:
        """
        Build table programmatically from structured data (Phase 3).

        NO LLM CALLS - pure deterministic data transformation.

        Args:
            table_spec: Table specification from Phase 1
            comparisons: Comparison entries with data
            calculations: Calculations for transparency

        Returns:
            Tuple of (Table object, metadata)
        """
        # 🔍 DEBUG: Log what data is available at table building time
        logger.info(
            f"[DEBUG_NA_BUG] 🏗️ build_table_from_data called: "
            f"table_id={table_spec.table_id}"
        )
        logger.info(
            f"[DEBUG_NA_BUG]   📦 Received {len(comparisons)} comparisons"
        )
        logger.info(
            f"[DEBUG_NA_BUG]   🧮 Received {len(calculations)} calculations"
        )

        # Log sample comparison structure
        if comparisons:
            sample = comparisons[0]
            logger.info(
                f"[DEBUG_NA_BUG]   📊 Sample comparison[0]: "
                f"primary_key='{sample.primary_key}', "
                f"metrics keys={list(sample.metrics.keys())[:5] if sample.metrics else 'EMPTY'}..."
            )
        else:
            logger.warning("[DEBUG_NA_BUG]   ⚠️ NO COMPARISONS PROVIDED!")

        # Log sample calculation
        if calculations:
            calc_sample = calculations[0]
            logger.info(
                f"[DEBUG_NA_BUG]   🧮 Sample calculation[0]: "
                f"description='{calc_sample.description}', "
                f"result={calc_sample.result}, "
                f"formula={getattr(calc_sample, 'formula', 'N/A')}"
            )
        else:
            logger.warning("[DEBUG_NA_BUG]   ⚠️ NO CALCULATIONS PROVIDED!")

        # Determine which entities to include
        if table_spec.row_entities:
            target_entities = table_spec.row_entities
        else:
            # Fallback to all entities in comparisons
            target_entities = list(set(c.primary_key for c in comparisons))
            logger.info(
                f"No row_entities in spec '{table_spec.table_id}', "
                f"using {len(target_entities)} entities from comparisons"
            )

        # Filter relevant comparisons
        relevant_comparisons = [
            c for c in comparisons
            if c.primary_key in target_entities
        ]

        if not relevant_comparisons and comparisons:
            # Enhanced fallback: Try fuzzy matching before giving up
            logger.warning(
                f"No exact entity matches for table '{table_spec.table_id}', "
                f"trying fuzzy matching. Target entities: {target_entities[:5]}..."
            )

            # Try to find comparisons that might match with fuzzy logic
            fuzzy_matches = []
            for comp in comparisons:
                comp_key_lower = comp.primary_key.lower()
                for target in target_entities:
                    target_lower = target.lower()
                    # Check various matching patterns
                    if (target_lower in comp_key_lower or
                        comp_key_lower in target_lower or
                        target_lower.replace(' ', '') == comp_key_lower.replace(' ', '') or
                        # Check if comparison key is a metric that should be transposed
                        any(word in comp_key_lower for word in ['rate', 'tax', 'contribution', 'benefit'])):
                        # Found potential match
                        fuzzy_matches.append(comp)
                        logger.debug(f"Fuzzy matched '{comp.primary_key}' to target '{target}'")
                        break

            if fuzzy_matches:
                relevant_comparisons = fuzzy_matches
                logger.info(
                    f"Found {len(fuzzy_matches)} fuzzy matches for table '{table_spec.table_id}'"
                )
            else:
                # Last resort: Check if comparisons use metric names as keys (common error)
                # In this case, we should try to restructure the data
                metric_like_keys = [
                    c for c in comparisons
                    if any(word in c.primary_key.lower()
                          for word in ['rate', 'tax', 'marginal', 'contribution', 'benefit', 'threshold'])
                ]

                if metric_like_keys:
                    logger.warning(
                        f"Detected metric-based comparison structure for table '{table_spec.table_id}'. "
                        f"Found {len(metric_like_keys)} metric-like keys. This suggests Stage 1B "
                        f"misinterpreted the structure. Using all comparisons as fallback."
                    )
                    relevant_comparisons = comparisons
                else:
                    # Final fallback: use all comparisons
                    logger.warning(
                        f"No matching entities for table '{table_spec.table_id}', "
                        "using all comparisons as last resort"
                    )
                    relevant_comparisons = comparisons

        # Determine columns
        if table_spec.column_metrics:
            columns = table_spec.column_metrics
            logger.info(
                f"[Table Builder] Using column_metrics from spec: {columns[:5]}..."
                if len(columns) > 5 else f"[Table Builder] Using column_metrics from spec: {columns}"
            )
        else:
            # Extract all unique metrics from comparisons
            all_metrics = set()
            for comp in relevant_comparisons:
                all_metrics.update(comp.metrics.keys())
            columns = sorted(list(all_metrics))
            logger.info(
                f"No column_metrics in spec '{table_spec.table_id}', "
                f"using {len(columns)} metrics from data"
            )

        # Debug: Show what metrics comparisons actually have
        if relevant_comparisons:
            sample_comp = relevant_comparisons[0]
            logger.info(
                f"[Table Builder] Sample comparison '{sample_comp.primary_key}' has metrics: "
                f"{list(sample_comp.metrics.keys())[:5]}..."
                if len(sample_comp.metrics) > 5 else
                f"{list(sample_comp.metrics.keys())}"
            )

        # Build header row
        headers = ["Entity"] + columns
        header_cells = [
            TableCell(content=header, alignment="left" if i == 0 else "right")
            for i, header in enumerate(headers)
        ]
        header_row = TableRow(cells=header_cells, row_type="header")

        # Build data rows with calculation transparency
        data_rows = []
        referenced_calcs = []
        
        # FIX #5: Track N/A statistics for debugging
        total_cells = 0
        na_cells = 0

        for comp in relevant_comparisons:
            cells = [TableCell(content=comp.primary_key, alignment="left")]

            # 🔍 DEBUG: Log comparison structure
            logger.info(
                f"[DEBUG_NA_BUG] 📊 Processing entity: '{comp.primary_key}', "
                f"available metrics: {list(comp.metrics.keys())[:5]}... ({len(comp.metrics)} total)"
            )

            for metric in columns:
                total_cells += 1
                value = comp.metrics.get(metric)

                # 🔍 DEBUG: Log each lookup attempt
                logger.info(
                    f"[DEBUG_NA_BUG] 🔎 Looking up: entity='{comp.primary_key}', metric='{metric}'"
                )
                logger.info(
                    f"[DEBUG_NA_BUG]   → Direct lookup result: {value}"
                )
                if value is None:
                    logger.info(
                        f"[DEBUG_NA_BUG]   ❌ NOT FOUND in comp.metrics - will try fuzzy matching"
                    )
                else:
                    logger.info(
                        f"[DEBUG_NA_BUG]   ✅ FOUND: {value}"
                    )

                # If exact match fails, try fuzzy matching by stripping scenario prefixes
                # This handles cases where table columns have scenario prefixes
                # (e.g., "single_net_take_home") but data points have simple names ("net_take_home")
                if value is None and "_" in metric:
                    # Try matching without common scenario prefixes
                    scenario_prefixes = [
                        "single_", "married_", "married_no_child_", "married_one_child_",
                        "married_two_child_", "family_", "individual_", "couple_"
                    ]
                    for prefix in scenario_prefixes:
                        if metric.startswith(prefix):
                            base_metric = metric[len(prefix):]
                            value = comp.metrics.get(base_metric)
                            if value is not None:
                                logger.debug(
                                    f"[Table Builder] Fuzzy matched '{metric}' to '{base_metric}' "
                                    f"for entity '{comp.primary_key}'"
                                )
                                break

                if value is None or value == "N/A":
                    cell_content = "N/A"
                    na_cells += 1  # FIX #5: Track N/A count
                else:
                    # Find related calculation for transparency
                    related_calc = self._find_related_calculation(
                        metric,
                        comp.primary_key,
                        calculations
                    )

                    if related_calc:
                        # Show value with formula if available (Phase 4: Formula Transparency)
                        cell_content = self._render_cell_with_formula(
                            value,
                            related_calc,
                            show_inline_formula=True  # Can be config-driven
                        )
                        if related_calc not in referenced_calcs:
                            referenced_calcs.append(related_calc)
                    else:
                        cell_content = str(value)

                cells.append(
                    TableCell(content=cell_content, alignment="right")
                )

            data_rows.append(TableRow(cells=cells, row_type="data"))

        # Build footnotes with calculation details
        footnotes = []
        for i, calc in enumerate(referenced_calcs[:5], 1):  # Limit to 5 footnotes
            footnotes.append(
                f"[{i}] {calc.description}: {calc.formula} = {calc.result} {calc.unit}"
            )

        # Create Table object
        table = Table(
            caption=table_spec.purpose,
            rows=[header_row] + data_rows,
            footnotes=footnotes if footnotes else None
        )

        # Create metadata
        metadata = TableRenderMetadata(
            source_spec_id=table_spec.table_id,
            title=table_spec.purpose,
            alignments=["left"] + ["right"] * len(columns),
            footnotes=footnotes,
            referenced_calculation_ids=[calc.description for calc in referenced_calcs]
        )

        # FIX #5: Log N/A statistics for debugging
        valid_cells = total_cells - na_cells
        na_percentage = (na_cells / total_cells * 100) if total_cells > 0 else 0
        
        logger.info(
            f"Built table '{table_spec.table_id}' with "
            f"{len(data_rows)} rows × {len(columns) + 1} columns. "
            f"Data quality: {valid_cells}/{total_cells} cells populated ({na_cells} N/A, {na_percentage:.1f}%)"
        )
        
        if na_cells == total_cells and total_cells > 0:
            logger.error(
                f"[TABLE BUILDER] CRITICAL: All {total_cells} data cells in table '{table_spec.table_id}' are N/A! "
                "This means no data was successfully extracted/calculated. "
                "Check: 1) extraction_path hints, 2) observation_id references, 3) calculation inputs"
            )
        elif na_percentage > 50:
            logger.warning(
                f"[TABLE BUILDER] Over 50% of cells ({na_percentage:.1f}%) in table '{table_spec.table_id}' are N/A. "
                "Table quality is poor. Consider reviewing data extraction logic."
            )

        return table, metadata

    def _find_related_calculation(
        self,
        metric: str,
        entity: str,
        calculations: List[Calculation]
    ) -> Optional[Calculation]:
        """Find calculation related to a specific metric and entity."""
        metric_lower = metric.lower().replace('_', ' ')
        entity_lower = entity.lower()

        for calc in calculations:
            desc_lower = calc.description.lower()
            # Check if both metric and entity are mentioned
            if (metric_lower in desc_lower or
                any(part in desc_lower for part in metric_lower.split())) and \
               entity_lower in desc_lower:
                return calc

        # Fallback: just metric match
        for calc in calculations:
            desc_lower = calc.description.lower()
            if metric_lower in desc_lower or \
               any(part in desc_lower for part in metric_lower.split()):
                return calc

        return None
    
    def _render_cell_with_formula(
        self,
        value: Any,
        calculation: Calculation,
        show_inline_formula: bool = True
    ) -> str:
        """Render table cell with optional inline formula display.
        
        Args:
            value: Cell value
            calculation: Related calculation object
            show_inline_formula: Whether to show formula inline
        
        Returns:
            Formatted cell content string
        """
        # Start with the value
        cell_content = str(value)
        
        # Add inline formula if enabled and available
        if show_inline_formula and hasattr(calculation, 'formula') and calculation.formula:
            # Simplify formula for inline display (max 50 chars)
            formula = calculation.formula
            if len(formula) > 50:
                formula = formula[:47] + "..."
            
            # Format: "value (formula)"
            # Example: "45,000 (net_income - rent)"
            cell_content = f"{value} <sup title='{formula}'>*</sup>"
        
        return cell_content

    def plan_sections_from_state(
        self,
        state: EnhancedResearchState
    ) -> List[Dict[str, Any]]:
        """
        Convert state plan to section structure (Phase 2.5).

        Args:
            state: Research state with plan

        Returns:
            List of section definitions
        """
        sections = []

        # Handle both dict and object access patterns for state
        current_plan = state.get('current_plan') if isinstance(state, dict) else getattr(state, 'current_plan', None)

        if current_plan and isinstance(current_plan, dict):
            # Try to extract suggested report structure
            if "suggested_report_structure" in current_plan:
                structure = current_plan["suggested_report_structure"]
                if isinstance(structure, list):
                    for item in structure:
                        if isinstance(item, str):
                            sections.append({
                                "title": item,
                                "purpose": f"Analysis of {item.lower()}"
                            })
                        elif isinstance(item, dict) and "title" in item:
                            sections.append(item)

            # If no explicit structure, infer from steps
            if not sections and "steps" in current_plan:
                # Always start with Executive Summary
                sections.append({
                    "title": "Executive Summary",
                    "purpose": "Overview and key findings"
                })

                # Add sections based on step patterns
                steps = current_plan.get("steps", [])
                has_comparison = any(
                    "compar" in str(step).lower()
                    for step in steps
                )
                has_analysis = any(
                    "analyz" in str(step).lower() or "analy" in str(step).lower()
                    for step in steps
                )

                if has_comparison:
                    sections.append({
                        "title": "Comparative Analysis",
                        "purpose": "Detailed comparison of key metrics"
                    })

                if has_analysis and not has_comparison:
                    sections.append({
                        "title": "Detailed Analysis",
                        "purpose": "In-depth analysis of findings"
                    })

                # Always end with conclusions
                sections.append({
                    "title": "Conclusions and Recommendations",
                    "purpose": "Summary and next steps"
                })

        # Fallback structure if no plan available
        if not sections:
            logger.info("No plan structure found, using default sections")
            sections = [
                {"title": "Executive Summary", "purpose": "Key findings overview"},
                {"title": "Analysis", "purpose": "Detailed analysis"},
                {"title": "Conclusions", "purpose": "Final insights"}
            ]

        logger.info(f"Planned {len(sections)} sections from state")
        return sections

    def render_table_to_markdown(self, table: Table) -> str:
        """
        Render Table object to markdown string.

        Guarantees correct markdown table syntax.

        Args:
            table: Table object to render

        Returns:
            Markdown string with perfect table formatting
        """
        if not table.rows:
            return ""

        lines = []

        # Determine column count from header
        header_row = table.rows[0]
        num_cols = len(header_row.cells)

        # Render header
        header_line = "| " + " | ".join(
            cell.content for cell in header_row.cells
        ) + " |"
        lines.append(header_line)

        # Separator row (GUARANTEED!)
        separator_line = "| " + " | ".join(["---"] * num_cols) + " |"
        lines.append(separator_line)

        # Data rows
        for row in table.rows[1:]:  # Skip header
            # Ensure correct column count
            cells = row.cells[:num_cols]
            while len(cells) < num_cols:
                cells.append(TableCell(content=""))

            row_line = "| " + " | ".join(
                cell.content for cell in cells
            ) + " |"
            lines.append(row_line)

        # Add caption if present
        if table.caption:
            lines.append("")
            lines.append(f"*Table: {table.caption}*")

        # Add footnotes if present
        if table.footnotes:
            lines.append("")
            for footnote in table.footnotes:
                lines.append(f"{footnote}")

        return "\n".join(lines)