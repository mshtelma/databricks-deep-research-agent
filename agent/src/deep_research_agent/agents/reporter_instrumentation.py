"""
Phase 0: Instrumentation and Debug Logging for Reporter

This module provides instrumentation to capture metrics and validate data completeness
before and after report generation. It helps identify missing entities, calculation coverage,
and table specification alignment with the state plan.
"""

import json
import logging
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
from pathlib import Path

from ..core.report_generation.models import (
    CalculationContext,
    ComparisonEntry,
    DataPoint,
    TableSpec
)
from ..core.multi_agent_state import EnhancedResearchState

logger = logging.getLogger(__name__)


class ReporterInstrumentation:
    """Instrumentation for capturing metrics and validating data completeness."""

    def __init__(self, debug_dir: Optional[str] = None):
        """
        Initialize instrumentation.

        Args:
            debug_dir: Directory for saving debug outputs (defaults to ./debug_output/)
        """
        self.debug_dir = Path(debug_dir or "./debug_output/")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}

    def instrument_state(
        self,
        state: EnhancedResearchState,
        calc_context: CalculationContext
    ) -> Dict[str, Any]:
        """
        Capture metrics from state and calculation context.

        Args:
            state: The research state containing plan and observations
            calc_context: The extracted calculation context

        Returns:
            Dictionary of captured metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "state_analysis": self._analyze_state(state),
            "calc_context_analysis": self._analyze_calc_context(calc_context),
            "validation_results": self._validate_completeness(state, calc_context),
            "golden_sample": self._capture_golden_sample(calc_context)
        }

        # Log critical findings
        for issue in metrics["validation_results"]["critical_issues"]:
            logger.warning(f"INSTRUMENTATION: {issue}")

        # Save to file for analysis
        self._save_metrics(metrics)

        self.metrics = metrics
        return metrics

    def _analyze_state(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Analyze the research state structure."""
        # Handle both dict and object access patterns for state
        current_plan = state.get('current_plan') if isinstance(state, dict) else getattr(state, 'current_plan', None)
        observations = state.get('observations') if isinstance(state, dict) else getattr(state, 'observations', None)
        factuality_report = state.get('factuality_report') if isinstance(state, dict) else getattr(state, 'factuality_report', None)
        factuality_score = state.get('factuality_score') if isinstance(state, dict) else getattr(state, 'factuality_score', None)
        report_style = state.get('report_style') if isinstance(state, dict) else getattr(state, 'report_style', None)

        analysis = {
            "has_plan": bool(current_plan),
            "plan_entities": [],
            "plan_sections": [],
            "observation_count": len(observations) if observations else 0,
            "has_factuality_report": bool(factuality_report),
            "factuality_score": factuality_score or 0.0,
            "report_style": str(report_style) if report_style else "DEFAULT"
        }

        # Extract entities and sections from plan
        if current_plan:
            if isinstance(current_plan, dict):
                # Look for entities mentioned in plan steps
                if "steps" in current_plan:
                    for step in current_plan.get("steps", []):
                        if isinstance(step, dict):
                            desc = step.get("description", "")
                            # Extract country names (simple heuristic)
                            for country in ["Spain", "France", "Germany", "Switzerland", "UK", "Poland", "Bulgaria"]:
                                if country in desc:
                                    if country not in analysis["plan_entities"]:
                                        analysis["plan_entities"].append(country)

                # Extract suggested sections
                if "suggested_report_structure" in current_plan:
                    structure = current_plan["suggested_report_structure"]
                    if isinstance(structure, list):
                        analysis["plan_sections"] = structure
                    elif isinstance(structure, dict) and "sections" in structure:
                        analysis["plan_sections"] = structure["sections"]

        return analysis

    def _analyze_calc_context(self, calc_context: CalculationContext) -> Dict[str, Any]:
        """Analyze the calculation context structure.

        FIXED: Handle both CalculationContext objects and dicts.
        """
        # FIXED: Handle both object and dict formats
        if isinstance(calc_context, dict):
            # Dict format: use .get() with defaults
            extracted_data = calc_context.get("extracted_data", [])
            calculations = calc_context.get("calculations", [])
            key_comparisons = calc_context.get("key_comparisons", [])
            table_specifications = calc_context.get("table_specifications", [])
        else:
            # Object format: use attributes
            extracted_data = calc_context.extracted_data
            calculations = calc_context.calculations
            key_comparisons = calc_context.key_comparisons
            table_specifications = calc_context.table_specifications

        analysis = {
            "data_points_count": len(extracted_data),
            "calculations_count": len(calculations),
            "comparisons_count": len(key_comparisons),
            "table_specs_count": len(table_specifications),
            "entities_in_data": set(),
            "metrics_in_data": set(),
            "entities_in_comparisons": set(),
            "table_spec_ids": []
        }

        # Extract unique entities and metrics
        for dp in extracted_data:
            # Handle both object and dict formats for data points
            entity = dp.entity if hasattr(dp, 'entity') else dp.get('entity')
            metric = dp.metric if hasattr(dp, 'metric') else dp.get('metric')
            if entity:
                analysis["entities_in_data"].add(entity)
            if metric:
                analysis["metrics_in_data"].add(metric)

        for comp in key_comparisons:
            # Handle both object and dict formats for comparisons
            primary_key = comp.primary_key if hasattr(comp, 'primary_key') else comp.get('primary_key')
            if primary_key:
                analysis["entities_in_comparisons"].add(primary_key)

        for spec in table_specifications:
            # Handle both object and dict formats for specs
            table_id = spec.table_id if hasattr(spec, 'table_id') else spec.get('table_id')
            if table_id:
                analysis["table_spec_ids"].append(table_id)

        # Convert sets to lists for JSON serialization
        analysis["entities_in_data"] = list(analysis["entities_in_data"])
        analysis["metrics_in_data"] = list(analysis["metrics_in_data"])
        analysis["entities_in_comparisons"] = list(analysis["entities_in_comparisons"])

        return analysis

    def _validate_completeness(
        self,
        state: EnhancedResearchState,
        calc_context: CalculationContext
    ) -> Dict[str, Any]:
        """
        Validate that calculation context contains entities from state plan.

        Returns:
            Validation results with critical issues and warnings
        """
        results = {
            "critical_issues": [],
            "warnings": [],
            "coverage": {}
        }

        state_analysis = self._analyze_state(state)
        calc_analysis = self._analyze_calc_context(calc_context)

        # Check entity coverage
        plan_entities = set(state_analysis["plan_entities"])
        data_entities = set(calc_analysis["entities_in_data"])
        comp_entities = set(calc_analysis["entities_in_comparisons"])

        missing_from_data = plan_entities - data_entities
        missing_from_comparisons = plan_entities - comp_entities

        if missing_from_data:
            results["critical_issues"].append(
                f"Plan entities missing from extracted data: {list(missing_from_data)}"
            )

        if missing_from_comparisons:
            results["warnings"].append(
                f"Plan entities missing from comparisons: {list(missing_from_comparisons)}"
            )

        # Check table specifications alignment
        # FIXED: Handle both dict and object formats for calc_context
        if isinstance(calc_context, dict):
            table_specs = calc_context.get("table_specifications", [])
        else:
            table_specs = calc_context.table_specifications

        if not table_specs:
            results["critical_issues"].append(
                "No table specifications found - tables cannot be generated"
            )
        else:
            # Verify each table spec has corresponding comparison data
            for spec in table_specs:
                # Handle both object and dict formats for specs
                row_entities = spec.row_entities if hasattr(spec, 'row_entities') else spec.get('row_entities', [])
                table_id = spec.table_id if hasattr(spec, 'table_id') else spec.get('table_id', 'unknown')

                spec_entities = set(row_entities) if row_entities else set()
                if spec_entities and not spec_entities.intersection(comp_entities):
                    results["warnings"].append(
                        f"Table spec '{table_id}' references entities not in comparisons"
                    )

        # Calculate coverage metrics
        if plan_entities:
            results["coverage"]["entity_data_coverage"] = len(
                data_entities.intersection(plan_entities)
            ) / len(plan_entities)
            results["coverage"]["entity_comparison_coverage"] = len(
                comp_entities.intersection(plan_entities)
            ) / len(plan_entities)
        else:
            results["coverage"]["entity_data_coverage"] = 0.0
            results["coverage"]["entity_comparison_coverage"] = 0.0

        # Check calculation coverage
        # FIXED: Handle both dict and object formats
        if isinstance(calc_context, dict):
            calculations = calc_context.get("calculations", [])
        else:
            calculations = calc_context.calculations

        if calculations:
            calculated_metrics = set()
            for calc in calculations:
                # Extract metrics from calculation description
                desc = calc.description if hasattr(calc, 'description') else calc.get('description', '')
                desc_lower = desc.lower()
                for metric in calc_analysis["metrics_in_data"]:
                    if metric.lower() in desc_lower:
                        calculated_metrics.add(metric)

            if calc_analysis["metrics_in_data"]:
                results["coverage"]["calculation_coverage"] = len(
                    calculated_metrics
                ) / len(calc_analysis["metrics_in_data"])
            else:
                results["coverage"]["calculation_coverage"] = 0.0
        else:
            results["coverage"]["calculation_coverage"] = 0.0
            results["warnings"].append("No calculations found in context")

        return results

    def _capture_golden_sample(self, calc_context: CalculationContext) -> Dict[str, Any]:
        """
        Capture a golden sample for regression testing.

        Returns:
            Sample data for comparison in tests
        """
        sample = {
            "first_data_point": None,
            "first_calculation": None,
            "first_comparison": None,
            "first_table_spec": None
        }

        # FIXED: Handle both dict and object formats
        if isinstance(calc_context, dict):
            extracted_data = calc_context.get("extracted_data", [])
            calculations = calc_context.get("calculations", [])
            key_comparisons = calc_context.get("key_comparisons", [])
            table_specs = calc_context.get("table_specifications", [])
        else:
            extracted_data = calc_context.extracted_data
            calculations = calc_context.calculations
            key_comparisons = calc_context.key_comparisons
            table_specs = calc_context.table_specifications

        if extracted_data:
            dp = extracted_data[0]
            sample["first_data_point"] = {
                "entity": dp.entity if hasattr(dp, 'entity') else dp.get('entity'),
                "metric": dp.metric if hasattr(dp, 'metric') else dp.get('metric'),
                "value": str(dp.value if hasattr(dp, 'value') else dp.get('value')),
                "unit": dp.unit if hasattr(dp, 'unit') else dp.get('unit')
            }

        if calculations:
            calc = calculations[0]
            sample["first_calculation"] = {
                "description": calc.description if hasattr(calc, 'description') else calc.get('description'),
                "formula": calc.formula if hasattr(calc, 'formula') else calc.get('formula'),
                "result": str(calc.result if hasattr(calc, 'result') else calc.get('result'))
            }

        if key_comparisons:
            comp = key_comparisons[0]
            primary_key = comp.primary_key if hasattr(comp, 'primary_key') else comp.get('primary_key')
            metrics = comp.metrics if hasattr(comp, 'metrics') else comp.get('metrics', {})
            sample["first_comparison"] = {
                "primary_key": primary_key,
                "metrics_count": len(metrics)
            }

        if table_specs:
            spec = table_specs[0]
            row_entities = spec.row_entities if hasattr(spec, 'row_entities') else spec.get('row_entities', [])
            column_metrics = spec.column_metrics if hasattr(spec, 'column_metrics') else spec.get('column_metrics', [])
            sample["first_table_spec"] = {
                "table_id": spec.table_id if hasattr(spec, 'table_id') else spec.get('table_id'),
                "purpose": spec.purpose if hasattr(spec, 'purpose') else spec.get('purpose'),
                "row_count": len(row_entities),
                "column_count": len(column_metrics)
            }

        return sample

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to debug file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.debug_dir / f"instrumentation_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Instrumentation metrics saved to {filename}")

    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """
        Compare current metrics with a baseline.

        Args:
            baseline_file: Path to baseline metrics file

        Returns:
            Comparison results
        """
        if not self.metrics:
            return {"error": "No current metrics captured"}

        try:
            with open(baseline_file, "r") as f:
                baseline = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load baseline: {e}"}

        comparison = {
            "improvements": [],
            "regressions": [],
            "unchanged": []
        }

        # Compare coverage metrics
        current_coverage = self.metrics["validation_results"]["coverage"]
        baseline_coverage = baseline.get("validation_results", {}).get("coverage", {})

        for key in current_coverage:
            current_val = current_coverage[key]
            baseline_val = baseline_coverage.get(key, 0.0)

            if current_val > baseline_val + 0.05:
                comparison["improvements"].append(
                    f"{key}: {baseline_val:.2f} → {current_val:.2f}"
                )
            elif current_val < baseline_val - 0.05:
                comparison["regressions"].append(
                    f"{key}: {baseline_val:.2f} → {current_val:.2f}"
                )
            else:
                comparison["unchanged"].append(key)

        return comparison