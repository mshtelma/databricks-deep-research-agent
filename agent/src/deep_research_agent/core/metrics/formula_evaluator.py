"""
Safe formula evaluation using numexpr with restrictions.

This module provides simple, safe mathematical formula evaluation without
the complexity of custom AST parsers. Uses numexpr library with sandboxing.
"""

from typing import Dict, Optional, Any, Set
import re
import logging

logger = logging.getLogger(__name__)


class FormulaEvaluation:
    """Result of formula evaluation."""

    def __init__(
        self,
        success: bool,
        value: Optional[float],
        error: Optional[str],
        formula: str,
        inputs_used: Dict[str, float]
    ):
        self.success = success
        self.value = value
        self.error = error
        self.formula = formula
        self.inputs_used = inputs_used


class SimpleFormulaEvaluator:
    """
    Safely evaluate mathematical formulas using numexpr with restrictions.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Parentheses for precedence
    - Variables from input dictionary
    - Common math functions: abs, sqrt, log, exp, sin, cos, tan
    - Safe: No global access, restricted to allowed functions

    Examples:
        evaluator = SimpleFormulaEvaluator()

        # Basic arithmetic
        result = evaluator.evaluate("revenue - costs", {"revenue": 1000, "costs": 600})
        # Returns: FormulaEvaluation(success=True, value=400.0, ...)

        # With math functions
        result = evaluator.evaluate(
            "sqrt(abs(profit)) * (1 + tax_rate)",
            {"profit": 100, "tax_rate": 0.2}
        )
        # Returns: FormulaEvaluation(success=True, value=12.0, ...)

        # Multi-domain examples:
        # Climate: "emissions_2024 - emissions_2020"
        # Medical: "success_rate * 100"
        # Business: "(q4_revenue - q3_revenue) / q3_revenue * 100"
    """

    def __init__(self):
        """Initialize with safe functions."""
        import numpy as np

        # Safe math functions to allow
        self.safe_functions = {
            'abs': np.abs,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
        }

    def evaluate(
        self,
        formula: str,
        inputs: Dict[str, Any],
        default_on_error: Optional[float] = None
    ) -> FormulaEvaluation:
        """
        Safely evaluate a formula with given inputs using numexpr.

        Args:
            formula: Mathematical formula as string
            inputs: Dictionary of variable names to values
            default_on_error: Value to return if evaluation fails

        Returns:
            FormulaEvaluation with result or error
        """
        import numexpr as ne

        try:
            # Convert all inputs to float
            clean_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, (int, float)):
                    clean_inputs[key] = float(value)
                else:
                    # Skip non-numeric inputs
                    continue

            # Track which inputs were used (simple approach: all provided numeric inputs)
            used_inputs = clean_inputs.copy()

            # Evaluate using numexpr with restrictions
            result = ne.evaluate(
                formula,
                global_dict={},  # No global access
                local_dict={**clean_inputs, **self.safe_functions}
            )

            # Convert to float
            value = float(result)

            return FormulaEvaluation(
                success=True,
                value=value,
                error=None,
                formula=formula,
                inputs_used=used_inputs
            )

        except Exception as e:
            logger.error(f"Formula evaluation failed: {formula}, error: {e}")
            return FormulaEvaluation(
                success=False,
                value=default_on_error,
                error=str(e),
                formula=formula,
                inputs_used={}
            )

    def extract_variables(self, formula: str) -> Set[str]:
        """
        Extract all variable names from a formula.

        Useful for validation before evaluation.

        Args:
            formula: Mathematical formula string

        Returns:
            Set of variable names used in formula

        Example:
            variables = evaluator.extract_variables("revenue - costs + tax")
            # Returns: {"revenue", "costs", "tax"}
        """
        # Simple regex to find potential variable names
        # Matches word characters that aren't pure numbers
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        potential_vars = set(re.findall(pattern, formula))

        # Remove known function names
        potential_vars -= set(self.safe_functions.keys())

        return potential_vars

    def validate_formula(self, formula: str) -> tuple[bool, Optional[str]]:
        """
        Validate a formula without evaluating it.

        Args:
            formula: Formula string to validate

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            # Basic validation - check if we can at least extract variables
            variables = self.extract_variables(formula)

            # Try with dummy values to catch syntax errors
            dummy_inputs = {var: 1.0 for var in variables}
            dummy_inputs.update(self.safe_functions)

            import numexpr as ne
            ne.evaluate(formula, global_dict={}, local_dict=dummy_inputs)

            return True, None
        except Exception as e:
            return False, str(e)


def execute_calculation_with_safety(
    formula: str,
    inputs: Dict[str, float],
    metric_name: str
) -> Dict[str, Any]:
    """
    Execute a calculation using the safe evaluator.

    This is a convenience function that wraps the evaluator with
    additional validation and error handling.

    Args:
        formula: Mathematical formula string
        inputs: Dict of variable names to numeric values
        metric_name: Name of the metric being calculated

    Returns:
        Dict with calculation results including:
        - metric_name: str
        - formula: str
        - inputs: Dict[str, float]
        - result: Optional[float]
        - unit: str
        - confidence: float
        - error: Optional[str]
        - verification_status: str

    Example:
        # Tax domain
        calc = execute_calculation_with_safety(
            "net - rent - daycare + benefits",
            {"net": 49000, "rent": 15600, "daycare": 5000, "benefits": 300},
            "disposable_income"
        )

        # Climate domain
        calc = execute_calculation_with_safety(
            "(emissions_2024 - emissions_2020) / emissions_2020 * 100",
            {"emissions_2024": 850, "emissions_2020": 1000},
            "emission_reduction_percentage"
        )

        # Medical domain
        calc = execute_calculation_with_safety(
            "successful_trials / total_trials * 100",
            {"successful_trials": 127, "total_trials": 150},
            "success_rate_percentage"
        )
    """
    evaluator = SimpleFormulaEvaluator()

    # First validate the formula
    is_valid, error = evaluator.validate_formula(formula)
    if not is_valid:
        return {
            "metric_name": metric_name,
            "formula": formula,
            "inputs": inputs,
            "result": None,
            "unit": "",
            "confidence": 0.0,
            "error": f"Invalid formula: {error}",
            "verification_status": "invalid"
        }

    # Check all required variables are provided
    required_vars = evaluator.extract_variables(formula)
    missing_vars = required_vars - set(inputs.keys())
    if missing_vars:
        return {
            "metric_name": metric_name,
            "formula": formula,
            "inputs": inputs,
            "result": None,
            "unit": "",
            "confidence": 0.0,
            "error": f"Missing inputs: {missing_vars}",
            "verification_status": "missing_inputs"
        }

    # Evaluate safely
    evaluation = evaluator.evaluate(formula, inputs)

    if evaluation.success:
        return {
            "metric_name": metric_name,
            "formula": formula,
            "inputs": evaluation.inputs_used,
            "result": evaluation.value,
            "unit": "",  # Will be determined from MetricSpec
            "confidence": 0.95,  # High confidence for successful calculation
            "verification_status": "verified",
            "error": None
        }
    else:
        return {
            "metric_name": metric_name,
            "formula": formula,
            "inputs": inputs,
            "result": None,
            "unit": "",
            "confidence": 0.0,
            "error": evaluation.error,
            "verification_status": "failed"
        }


__all__ = [
    'FormulaEvaluation',
    'SimpleFormulaEvaluator',
    'execute_calculation_with_safety',
]
