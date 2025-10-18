"""Formula validation for safety and correctness checks."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from .. import get_logger


logger = get_logger(__name__)


class ValidationResult(BaseModel):
    """Result of formula validation."""
    
    valid: bool
    confidence: float = Field(default=1.0, description="Confidence in validation")
    reason: Optional[str] = Field(default=None, description="Reason for failure")
    warning: Optional[str] = Field(default=None, description="Warning message")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix")


class TestResult(BaseModel):
    """Result of formula execution test."""
    
    reasonable: bool
    reason: Optional[str] = None
    test_value: Optional[Any] = None


class FormulaValidator:
    """Validate extracted formulas for safety and correctness."""
    
    # Dangerous Python operations to block
    DANGEROUS_PATTERNS = [
        r'\bimport\b',  # imports
        r'\bexec\b',  # exec calls
        r'\beval\b',  # eval calls (though we use safe eval)
        r'\bopen\b',  # file operations
        r'\bfile\b',  # file operations
        r'__\w+__',  # dunder methods
        r'\bos\.',  # os module
        r'\bsys\.',  # sys module
        r'\bsubprocess\.',  # subprocess
    ]
    
    # Safe math operations
    SAFE_OPERATORS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.USub, ast.UAdd,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
        ast.And, ast.Or, ast.Not
    }
    
    def validate_formula(
        self,
        formula: str,
        available_vars: List[str],
        expected_output: Optional[str] = None
    ) -> ValidationResult:
        """Multi-level validation of a formula.
        
        Args:
            formula: Formula string to validate
            available_vars: List of available variable names
            expected_output: Expected output type ('percentage', 'currency', etc.)
        
        Returns:
            ValidationResult with validation status
        """
        # Level 1: Syntax check
        syntax_result = self._validate_syntax(formula)
        if not syntax_result.valid:
            return syntax_result
        
        # Level 2: Variable availability
        var_result = self._validate_variables(formula, available_vars)
        if not var_result.valid:
            return var_result
        
        # Level 3: Safety check
        safety_result = self._validate_safety(formula)
        if not safety_result.valid:
            return safety_result
        
        # Level 4: Semantic validation (if expected output provided)
        if expected_output:
            semantic_result = self._validate_semantics(formula, expected_output)
            if not semantic_result.reasonable:
                return ValidationResult(
                    valid=True,  # Syntactically valid
                    confidence=0.6,
                    warning=f"Output seems unreasonable: {semantic_result.reason}"
                )
        
        return ValidationResult(valid=True, confidence=0.9)
    
    def _sanitize_fstrings(self, code: str) -> str:
        """Remove spaces before colons in f-string format specifiers.
        
        Args:
            code: Python code string
        
        Returns:
            Sanitized code
        """
        # Pattern: matches {variable_name <spaces> :format_spec}
        # Captures the variable name and format spec, removes spaces between them
        pattern = r'\{([^}]+?)\s+:([^}]+?)\}'
        return re.sub(pattern, r'{\1:\2}', code)
    
    def _validate_syntax(self, formula: str) -> ValidationResult:
        """Validate Python syntax of formula.

        Supports both multi-line code with statements (exec mode) and
        simple expressions (eval mode).

        NOTE: F-string sanitization is no longer needed here because the
        planner converts all f-strings to .format() calls before validation.

        Args:
            formula: Formula string

        Returns:
            ValidationResult
        """
        # F-string conversion happens in planner.py, so no need to sanitize here

        try:
            # Try exec mode first (supports statements like try-except, assignments)
            ast.parse(formula, mode='exec')
            return ValidationResult(valid=True)
        
        except SyntaxError:
            # Fallback: try eval mode for simple expressions
            try:
                ast.parse(formula, mode='eval')
                return ValidationResult(valid=True)
            except SyntaxError as e:
                suggested_fix = self._suggest_syntax_fix(formula, str(e))
                return ValidationResult(
                    valid=False,
                    reason=f"Invalid Python syntax: {e}",
                    suggested_fix=suggested_fix
                )
        
        except Exception as e:
            return ValidationResult(
                valid=False,
                reason=f"Failed to parse formula: {e}"
            )
    
    def _validate_variables(
        self,
        formula: str,
        available_vars: List[str]
    ) -> ValidationResult:
        """Validate that all variables in formula are available.
        
        Args:
            formula: Formula string
            available_vars: List of available variable names
        
        Returns:
            ValidationResult
        """
        used_vars = self._extract_variables(formula)
        missing_vars = set(used_vars) - set(available_vars)
        
        # Filter out built-in functions and constants
        builtin_names = {'abs', 'min', 'max', 'sum', 'len', 'round', 'int', 'float', 'str'}
        missing_vars = missing_vars - builtin_names
        
        if missing_vars:
            suggested_fix = self._suggest_variable_mapping(missing_vars, available_vars)
            return ValidationResult(
                valid=False,
                reason=f"Missing variables: {', '.join(sorted(missing_vars))}",
                suggested_fix=suggested_fix
            )
        
        return ValidationResult(valid=True)
    
    def _validate_safety(self, formula: str) -> ValidationResult:
        """Validate formula doesn't contain dangerous operations.
        
        Args:
            formula: Formula string
        
        Returns:
            ValidationResult
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, formula, re.IGNORECASE):
                return ValidationResult(
                    valid=False,
                    reason=f"Formula contains potentially dangerous operation: {pattern}",
                    suggested_fix="Remove system calls, imports, or file operations"
                )
        
        # Check AST for dangerous node types
        try:
            tree = ast.parse(formula, mode='eval')
            for node in ast.walk(tree):
                # Check for dangerous node types
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    return ValidationResult(
                        valid=False,
                        reason="Formula contains import statement",
                        suggested_fix="Remove import statement"
                    )
                
                if isinstance(node, (ast.Call,)):
                    # Check function call is safe
                    func_name = self._get_function_name(node)
                    if func_name in {'exec', 'eval', 'compile', '__import__'}:
                        return ValidationResult(
                            valid=False,
                            reason=f"Formula contains dangerous function: {func_name}",
                            suggested_fix=f"Remove {func_name} call"
                        )
        
        except Exception as e:
            logger.warning(f"[VALIDATOR] AST safety check failed: {e}")
        
        return ValidationResult(valid=True)
    
    def _validate_semantics(
        self,
        formula: str,
        expected_type: str
    ) -> TestResult:
        """Validate formula produces reasonable output.
        
        Args:
            formula: Formula string
            expected_type: Expected output type
        
        Returns:
            TestResult indicating if output is reasonable
        """
        # Generate test data based on expected type
        test_vars = self._generate_test_data(expected_type)
        
        try:
            # Execute in safe environment
            result = self._safe_eval(formula, test_vars)
            
            # Check output reasonableness based on type
            if expected_type == "percentage":
                if not (0 <= result <= 100):
                    return TestResult(
                        reasonable=False,
                        reason=f"Percentage {result} out of range [0, 100]"
                    )
            
            elif expected_type == "currency":
                if result < 0:
                    return TestResult(
                        reasonable=False,
                        reason="Negative currency value"
                    )
            
            elif expected_type == "ratio":
                if not (0 <= result <= 10):  # Most ratios under 10
                    return TestResult(
                        reasonable=False,
                        reason=f"Ratio {result} seems unusually high"
                    )
            
            return TestResult(reasonable=True, test_value=result)
        
        except ZeroDivisionError:
            return TestResult(
                reasonable=False,
                reason="Formula causes division by zero with test data"
            )
        
        except Exception as e:
            return TestResult(
                reasonable=False,
                reason=f"Formula execution failed: {e}"
            )
    
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variable names from formula.
        
        Args:
            formula: Formula string
        
        Returns:
            List of variable names
        """
        try:
            tree = ast.parse(formula, mode='eval')
            variables = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    variables.append(node.id)
            
            return list(set(variables))
        
        except Exception as e:
            logger.warning(f"[VALIDATOR] Failed to extract variables: {e}")
            # Fallback to regex
            return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
    
    def _get_function_name(self, node: ast.Call) -> Optional[str]:
        """Get function name from Call node.
        
        Args:
            node: AST Call node
        
        Returns:
            Function name or None
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def _suggest_syntax_fix(self, formula: str, error: str) -> str:
        """Suggest fix for syntax error.
        
        Args:
            formula: Original formula
            error: Error message
        
        Returns:
            Suggested fix
        """
        # Common fixes
        if "unterminated string" in error.lower():
            return "Check for unmatched quotes in the formula"
        
        if "unmatched ')'" in error.lower() or "unmatched '('" in error.lower():
            return "Check for balanced parentheses"
        
        if "invalid syntax" in error.lower():
            # Try to identify the issue
            if '×' in formula or '÷' in formula:
                return "Use * for multiplication and / for division instead of × and ÷"
        
        return "Check formula syntax and ensure it's valid Python"
    
    def _suggest_variable_mapping(
        self,
        missing_vars: Set[str],
        available_vars: List[str]
    ) -> str:
        """Suggest variable mapping for missing variables.
        
        Args:
            missing_vars: Missing variable names
            available_vars: Available variable names
        
        Returns:
            Suggestion string
        """
        suggestions = []
        
        for missing in missing_vars:
            # Find closest match
            missing_lower = missing.lower().replace('_', '')
            closest = None
            min_distance = float('inf')
            
            for available in available_vars:
                available_lower = available.lower().replace('_', '')
                distance = self._levenshtein_distance(missing_lower, available_lower)
                if distance < min_distance:
                    min_distance = distance
                    closest = available
            
            if closest and min_distance <= 3:  # Close enough match
                suggestions.append(f"'{missing}' -> '{closest}'")
        
        if suggestions:
            return f"Possible mappings: {', '.join(suggestions)}"
        
        return f"Variables {', '.join(sorted(missing_vars))} not found in available data"
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _generate_test_data(self, expected_type: str) -> Dict[str, float]:
        """Generate test data for formula validation.
        
        Args:
            expected_type: Expected output type
        
        Returns:
            Dictionary of test variable values
        """
        # Generate reasonable test values based on type
        if expected_type == "percentage":
            return {
                'gross_income': 100000.0,
                'net_income': 70000.0,
                'tax': 30000.0,
                'deduction': 10000.0,
            }
        
        elif expected_type == "currency":
            return {
                'gross_income': 100000.0,
                'tax_rate': 0.3,
                'deduction': 10000.0,
            }
        
        else:
            # Default test values
            return {
                'value1': 100.0,
                'value2': 50.0,
                'rate': 0.2,
                'count': 10.0,
            }
    
    def _safe_eval(self, formula: str, variables: Dict[str, Any]) -> Any:
        """Safely evaluate formula with given variables.
        
        Args:
            formula: Formula string
            variables: Variable values
        
        Returns:
            Evaluation result
        """
        # Create safe namespace with only math operations
        safe_dict = {
            '__builtins__': {
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'round': round,
                'int': int,
                'float': float,
            }
        }
        safe_dict.update(variables)
        
        # Compile and evaluate
        code = compile(formula, '<formula>', 'eval')
        return eval(code, safe_dict)


__all__ = ["FormulaValidator", "ValidationResult", "TestResult"]

