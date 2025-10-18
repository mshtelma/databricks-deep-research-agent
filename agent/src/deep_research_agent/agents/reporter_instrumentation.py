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
        """Analyze the calculation context structure."""
        analysis = {
            "data_points_count": len(calc_context.extracted_data),
            "calculations_count": len(calc_context.calculations),
            "comparisons_count": len(calc_context.key_comparisons),
            "table_specs_count": len(calc_context.table_specifications),
            "entities_in_data": set(),
            "metrics_in_data": set(),
            "entities_in_comparisons": set(),
            "table_spec_ids": []
        }

        # Extract unique entities and metrics
        for dp in calc_context.extracted_data:
            analysis["entities_in_data"].add(dp.entity)
            analysis["metrics_in_data"].add(dp.metric)

        for comp in calc_context.key_comparisons:
            analysis["entities_in_comparisons"].add(comp.primary_key)

        for spec in calc_context.table_specifications:
            analysis["table_spec_ids"].append(spec.table_id)

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
        if not calc_context.table_specifications:
            results["critical_issues"].append(
                "No table specifications found - tables cannot be generated"
            )
        else:
            # Verify each table spec has corresponding comparison data
            for spec in calc_context.table_specifications:
                spec_entities = set(spec.row_entities)
                if spec_entities and not spec_entities.intersection(comp_entities):
                    results["warnings"].append(
                        f"Table spec '{spec.table_id}' references entities not in comparisons"
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
        if calc_context.calculations:
            calculated_metrics = set()
            for calc in calc_context.calculations:
                # Extract metrics from calculation description
                desc_lower = calc.description.lower()
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

        if calc_context.extracted_data:
            dp = calc_context.extracted_data[0]
            sample["first_data_point"] = {
                "entity": dp.entity,
                "metric": dp.metric,
                "value": str(dp.value),
                "unit": dp.unit
            }

        if calc_context.calculations:
            calc = calc_context.calculations[0]
            sample["first_calculation"] = {
                "description": calc.description,
                "formula": calc.formula,
                "result": str(calc.result)
            }

        if calc_context.key_comparisons:
            comp = calc_context.key_comparisons[0]
            sample["first_comparison"] = {
                "primary_key": comp.primary_key,
                "metrics_count": len(comp.metrics)
            }

        if calc_context.table_specifications:
            spec = calc_context.table_specifications[0]
            sample["first_table_spec"] = {
                "table_id": spec.table_id,
                "purpose": spec.purpose,
                "row_count": len(spec.row_entities),
                "column_count": len(spec.column_metrics)
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