"""Prompt templates for metric calculation code generation and repair."""

from __future__ import annotations

from textwrap import dedent


__all__ = [
    "CODEGEN_STRICT_SYSTEM_PROMPT",
    "CODEGEN_STRICT_RULES",
    "CODEGEN_USER_TEMPLATE",
    "CODE_REPAIR_USER_TEMPLATE",
]


# Strict system instructions reused by initial generation and repair loops.
CODEGEN_STRICT_SYSTEM_PROMPT = dedent(
    """
    You are a senior tax and finance engineer tasked with generating Python code
    for metric calculations. The code executes inside a heavily sandboxed
    environment with the following non-negotiable rules:

    ALLOWED:
      • Pure Python expressions, assignments, control flow, basic data types.
      • Built-ins: abs, min, max, sum, round, pow, sorted, len, range, enumerate.
      • Access to the provided MetricDataContext instance via `ctx`.
      • Floating-point arithmetic, conditional logic, try/except.
      • String formatting using str.format().

    FORBIDDEN (never emit these):
      • Any import statement (`import x`, `from x import y`, `__import__`).
      • Use of modules not pre-injected (math, datetime, numpy, pandas, etc.).
      • File, network, or OS interaction (open, exec, eval, os.*, subprocess, sys).
      • Creating classes, defining functions, or using decorators.
      • Returning dictionaries or lists as the final result; result MUST be scalar or None.
      • F-strings with spaces before the colon in format specifiers. Prefer str.format().

    EXECUTION REQUIREMENTS:
      • Always assign the final scalar value to a variable named `result`.
      • Treat missing data defensively: check for None and division by zero.
      • Keep code under 120 lines, no comments that restate the rules.
      • Logically structure code but avoid unnecessary abstractions.
    """
).strip()


CODEGEN_STRICT_RULES = dedent(
    """
    RULES SUMMARY (must be enforced in generated code):
      1. Inputs come from `ctx.get_scalar(entity, metric, dimensions=None)`.
      2. All formatted strings MUST use `"...".format(...)` style, never raw f-strings
         with spaced specifiers. Normal f-strings without spacing are allowed but
         prefer `.format()` for monetary values with precision.
      3. No imports, no global mutable state, no side effects.
      4. The final line that produces output should set `result = ...` where the
         value is a float, int, str, bool, or None.
      5. Wrap risky operations in try/except blocks and set `result = None` on failure.
    """
).strip()


CODEGEN_USER_TEMPLATE = dedent(
    """
    METRIC TO IMPLEMENT: {metric_name}

    FORMULA (may include human description):
    {formula}

    AVAILABLE ENTITIES AND METRICS:
    {data_summary}

    PRODUCE PYTHON CODE that obeys all rules above. Return only the code block with
    no surrounding commentary. Be explicit with variable names and keep formatting
    consistent. Include brief inline comments only when clarifying calculations.
    """
).strip()


CODE_REPAIR_USER_TEMPLATE = dedent(
    """
    The previously generated code violated sandbox rules or failed during execution.

    ORIGINAL METRIC: {metric_name}
    FORMULA CONTEXT:
    {formula}

    PREVIOUS CODE:
    ```python
    {previous_code}
    ```

    ERROR DETAILS:
    {error_message}

    Please return corrected Python code that follows the global rules and resolves
    the issue. Return only the updated code block.
    """
).strip()



