"""
KaTeX formula formatter for calculation display.

Converts Python arithmetic expressions to KaTeX-formatted mathematical notation
for beautiful rendering in reports and UIs.
"""

import re
from typing import Optional


def convert_to_katex(python_code: str) -> Optional[str]:
    """
    Convert Python arithmetic code to KaTeX mathematical notation.
    
    Performs simple transformations to make calculations more readable:
    - (a / b) → \\frac{a}{b}
    - a * b → a \\times b
    - a ** b → a^{b}
    - Preserves parentheses and operators
    
    Args:
        python_code: Python arithmetic expression
    
    Returns:
        KaTeX-formatted string or None if conversion fails
    
    Examples:
        >>> convert_to_katex("(tax / income) * 100")
        "$\\\\frac{tax}{income} \\\\times 100$"
        
        >>> convert_to_katex("base_rate * (1 + modifier)")
        "$base\\_rate \\\\times (1 + modifier)$"
    """
    if not python_code or not isinstance(python_code, str):
        return None
    
    try:
        # Extract the assignment if present (result = ...)
        code = python_code.strip()
        if 'result =' in code:
            # Get expression after 'result ='
            parts = code.split('result =', 1)
            if len(parts) > 1:
                code = parts[1].strip()
        
        # Remove line breaks and extra whitespace
        code = ' '.join(code.split())
        
        # Skip complex code with control flow
        if any(keyword in code for keyword in ['if ', 'for ', 'while ', 'def ', 'ctx.', 'None']):
            return None
        
        katex = code
        
        # Convert division to fractions (handle nested parentheses)
        # Pattern: (numerator / denominator) -> \frac{numerator}{denominator}
        katex = re.sub(r'\(([^()]+)\s*/\s*([^()]+)\)', r'\\frac{\1}{\2}', katex)
        
        # Convert power operator
        katex = re.sub(r'\*\*', r'^', katex)
        
        # Convert multiplication to times symbol
        katex = katex.replace('*', r' \times ')
        
        # Escape underscores in variable names
        katex = katex.replace('_', r'\_')
        
        # Clean up extra spaces
        katex = re.sub(r'\s+', ' ', katex).strip()
        
        # Wrap in dollar signs for inline math
        return f"${katex}$"
    
    except Exception:
        # Return None if conversion fails
        return None


def format_calculation_for_display(
    description: str,
    formula_python: str,
    result: float,
    unit: str = ""
) -> str:
    """
    Format a complete calculation for display with KaTeX.
    
    Args:
        description: Description of the calculation
        formula_python: Python code expression
        result: Numerical result
        unit: Unit of measurement
    
    Returns:
        Formatted string with description, KaTeX formula, and result
    
    Example:
        >>> format_calculation_for_display(
        ...     "Effective tax rate",
        ...     "(total_tax / gross_income) * 100",
        ...     23.5,
        ...     "%"
        ... )
        "Effective tax rate: $\\\\frac{total\\\\_tax}{gross\\\\_income} \\\\times 100$ = 23.5%"
    """
    katex = convert_to_katex(formula_python)
    
    if katex:
        result_str = f"{result}{unit}" if unit else str(result)
        return f"{description}: {katex} = {result_str}"
    else:
        # Fallback to plain formula
        result_str = f"{result}{unit}" if unit else str(result)
        return f"{description}: {formula_python} = {result_str}"


