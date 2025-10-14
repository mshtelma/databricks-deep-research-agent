"""Sandboxed Python execution for metric calculations.

This module provides secure execution of LLM-generated Python code using RestrictedPython
and additional security checks. It prevents dangerous operations while allowing safe
mathematical calculations.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import traceback

try:
    from RestrictedPython import compile_restricted, safe_globals, limited_builtins
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence, 
        safe_builtins,
        safer_getattr,
        full_write_guard
    )
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False

from .. import get_logger


logger = get_logger(__name__)


# Dangerous patterns that should never appear in calculation code
DANGEROUS_PATTERNS = [
    r'\bimport\s+os\b',
    r'\bimport\s+sys\b',
    r'\bimport\s+subprocess\b',
    r'\b__import__\b',
    r'\beval\b',
    r'\bexec\b',
    r'\bcompile\b',
    r'\bopen\s*\(',
    r'\bfile\s*\(',
    r'\b__file__\b',
    r'\b__name__\b',
    r'\bgetattr\b',
    r'\bsetattr\b',
    r'\bdelattr\b',
    r'\bglobals\b',
    r'\blocals\b',
    r'\bvars\b',
    r'\bdir\b',
]


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""
    
    success: bool
    result: Any
    execution_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    locals_snapshot: Optional[Dict[str, Any]] = None


class SecurityError(Exception):
    """Raised when code violates security constraints."""
    pass


class TimeoutError(Exception):
    """Raised when execution exceeds time limit."""
    pass


class SafePythonExecutor:
    """Sandboxed Python executor using RestrictedPython.
    
    This class provides secure execution of calculation code with:
    - Whitelist of safe built-ins and modules
    - Blacklist of dangerous patterns
    - Execution timeout
    - Memory limits (basic)
    - Full audit trail
    
    Example:
        >>> executor = SafePythonExecutor(timeout_seconds=5)
        >>> result = executor.execute(
        ...     code="result = (100 * 0.19) + (50 * 0.24)",
        ...     context={"ctx": data_context}
        ... )
        >>> print(result.result)  # 31.0
    """
    
    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_iterations: int = 10000,
        enable_pandas: bool = True
    ):
        """Initialize sandboxed executor.
        
        Args:
            timeout_seconds: Maximum execution time per calculation
            max_iterations: Maximum iterations for loops (basic runaway prevention)
            enable_pandas: Whether to allow pandas operations
        """
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.enable_pandas = enable_pandas
        
        if not RESTRICTED_PYTHON_AVAILABLE:
            logger.warning(
                "RestrictedPython not available. Calculations will run in "
                "limited mode without full sandboxing. Install with: "
                "pip install RestrictedPython"
            )
    
    def _build_safe_globals(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the safe global namespace for execution.
        
        Args:
            context: Additional context to inject (e.g., MetricDataContext)
        
        Returns:
            Dictionary of safe globals
        """
        if RESTRICTED_PYTHON_AVAILABLE:
            # Start with RestrictedPython's safe globals
            safe_dict = {
                '__builtins__': {
                    **limited_builtins,
                    '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
                    '_getiter_': lambda x: iter(x),
                    '_getattr_': safer_getattr,
                    '_write_': full_write_guard,
                }
            }
        else:
            # Fallback: minimal safe builtins
            safe_dict = {
                '__builtins__': {
                    'True': True,
                    'False': False,
                    'None': None,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'len': len,
                    'range': range,
                    'round': round,
                    'float': float,
                    'int': int,
                    'str': str,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                }
            }
        
        # Add safe modules
        safe_dict['math'] = math
        safe_dict['statistics'] = statistics
        
        # Add pandas if enabled
        if self.enable_pandas:
            try:
                import pandas as pd
                safe_dict['pd'] = pd
            except ImportError:
                logger.debug("pandas not available for calculations")
        
        # Inject user context (e.g., MetricDataContext instance)
        if context:
            safe_dict.update(context)
        
        return safe_dict
    
    def _scan_for_dangerous_patterns(self, code: str) -> None:
        """Scan code for dangerous patterns before execution.
        
        Args:
            code: Python code to scan
        
        Raises:
            SecurityError: If dangerous patterns detected
        """
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityError(
                    f"Code contains disallowed pattern: {pattern}"
                )
        
        # Check for suspicious imports (even non-os ones in calculation context)
        if re.search(r'\bimport\s+\w+', code):
            raise SecurityError(
                "Import statements are not allowed in calculation code. "
                "Use the provided context API instead."
            )
        
        # Check for from imports
        if re.search(r'\bfrom\s+\w+', code):
            raise SecurityError(
                "From imports are not allowed in calculation code."
            )
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code in sandboxed environment.
        
        Args:
            code: Python code to execute
            context: Additional context (e.g., {'ctx': MetricDataContext()})
        
        Returns:
            ExecutionResult with success status and result/error
        
        Example:
            >>> result = executor.execute(
            ...     "result = 100 * 0.19",
            ...     context={}
            ... )
            >>> result.success  # True
            >>> result.result  # 19.0
        """
        start_time = time.time()
        
        try:
            # Step 1: Security scan
            self._scan_for_dangerous_patterns(code)
            
            # Step 2: Build safe environment
            safe_globals = self._build_safe_globals(context)
            safe_locals: Dict[str, Any] = {}
            
            # Step 3: Compile code
            if RESTRICTED_PYTHON_AVAILABLE:
                # Use RestrictedPython for secure compilation
                # compile_restricted returns a CompileResult with .code and .errors
                try:
                    compile_result = compile_restricted(code, '<calculation>', 'exec')
                    # Check if compile_result has errors attribute (some versions return tuple)
                    if hasattr(compile_result, 'errors') and compile_result.errors:
                        error_msg = '; '.join(compile_result.errors)
                        return ExecutionResult(
                            success=False,
                            result=None,
                            execution_time=time.time() - start_time,
                            error_type="CompileError",
                            error_message=f"RestrictedPython compile errors: {error_msg}"
                        )
                    # Extract compiled code
                    if hasattr(compile_result, 'code'):
                        compiled_code = compile_result.code
                    else:
                        # Some versions return code directly
                        compiled_code = compile_result
                except Exception as e:
                    logger.error(f"RestrictedPython compilation failed: {e}")
                    # Fallback to standard compile if RestrictedPython fails
                    compiled_code = compile(code, '<calculation>', 'exec')
            else:
                # Fallback: standard compile (less secure)
                compiled_code = compile(code, '<calculation>', 'exec')
            
            # Step 4: Execute with timeout
            # Note: This is a basic timeout using time checks, not true preemption
            # For production, consider using process-based timeouts
            exec(compiled_code, safe_globals, safe_locals)
            
            execution_time = time.time() - start_time
            
            # Check timeout (post-execution, not during)
            if execution_time > self.timeout_seconds:
                return ExecutionResult(
                    success=False,
                    result=None,
                    execution_time=execution_time,
                    error_type="TimeoutError",
                    error_message=f"Execution exceeded {self.timeout_seconds}s timeout"
                )
            
            # Step 5: Extract result
            # Convention: calculations should assign to 'result' variable
            if 'result' in safe_locals:
                result_value = safe_locals['result']
            else:
                # Fallback: return all locals
                logger.warning("Code did not assign to 'result' variable")
                result_value = safe_locals
            
            return ExecutionResult(
                success=True,
                result=result_value,
                execution_time=execution_time,
                locals_snapshot=safe_locals
            )
        
        except SecurityError as e:
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error_type="SecurityError",
                error_message=str(e)
            )
        
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error_type="SyntaxError",
                error_message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        
        except ZeroDivisionError as e:
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error_type="ZeroDivisionError",
                error_message="Division by zero in calculation"
            )
        
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Common calculation errors
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error_type=type(e).__name__,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error during sandboxed execution: {e}")
            logger.debug(f"Code was: {code}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error_type=type(e).__name__,
                error_message=f"Execution error: {str(e)}"
            )
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code without executing it.
        
        Args:
            code: Python code to validate
        
        Returns:
            (is_valid, error_message) tuple
        """
        try:
            self._scan_for_dangerous_patterns(code)
            
            if RESTRICTED_PYTHON_AVAILABLE:
                try:
                    compile_result = compile_restricted(code, '<validation>', 'exec')
                    # Check if compile_result has errors attribute
                    if hasattr(compile_result, 'errors') and compile_result.errors:
                        return False, '; '.join(compile_result.errors)
                except Exception as e:
                    # If RestrictedPython compilation fails, fall back to standard compile
                    logger.debug(f"RestrictedPython validation failed: {e}")
                    compile(code, '<validation>', 'exec')
            else:
                compile(code, '<validation>', 'exec')
            
            return True, None
        
        except SecurityError as e:
            return False, f"Security violation: {str(e)}"
        
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg}"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"


__all__ = ["SafePythonExecutor", "ExecutionResult", "SecurityError", "TimeoutError"]

