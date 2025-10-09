"""
Utilities for processing and fixing markdown content.
"""

import re
import html
from typing import List
from .table_validator import TableValidator


def clean_html_content(text: str) -> str:
    """
    Clean HTML entities and tags from text.

    This function:
    1. Unescapes HTML entities (&amp; -> &, &lt; -> <, &quot; -> ", etc.)
    2. Removes HTML tags (<strong>, <em>, <span>, etc.)
    3. Cleans up excessive whitespace

    Args:
        text: Text potentially containing HTML entities and tags

    Returns:
        Clean text without HTML entities or tags

    Examples:
        >>> clean_html_content("Python &amp; Java")
        'Python & Java'
        >>> clean_html_content("<strong>Python</strong> is great")
        'Python is great'
    """
    if not text:
        return text

    # First, unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)

    # Remove HTML tags (like <strong>, <em>, <span>, etc.)
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up multiple spaces that may be left after tag removal
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def fix_markdown_tables(content: str) -> str:
    """
    Fix malformed markdown tables by correcting common formatting issues.
    
    Common issues fixed:
    - Double pipes (||) replaced with single pipes (|)
    - Missing line breaks between table rows
    - Malformed header separators
    
    Args:
        content: The content potentially containing malformed markdown tables
        
    Returns:
        Content with fixed markdown table formatting
    """
    if not content or '|' not in content:
        return content
    
    # Apply multiple passes to catch all issues
    result = content
    
    # Pass 1: Basic double pipe removal
    result = re.sub(r'\|\|+', '|', result)
    
    # Pass 2: Fix lines with pipes - split and process
    lines = result.split('\n')
    fixed_lines = []
    
    for line in lines:
        if '|' in line:
            # Fix double pipes that might still exist
            fixed_line = re.sub(r'\|\|+', '|', line)
            
            # Ensure proper spacing around pipes (but not too aggressive)
            # Only fix obvious cases where there's no space before/after pipes
            fixed_line = re.sub(r'\|([^\|\s])', r'| \1', fixed_line)  # Add space after pipe
            fixed_line = re.sub(r'([^\|\s])\|', r'\1 |', fixed_line)  # Add space before pipe
            
            # Clean up multiple spaces
            fixed_line = re.sub(r'  +', ' ', fixed_line)
            
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Pass 3: Final cleanup of any remaining double pipes
    result = '\n'.join(fixed_lines)
    result = re.sub(r'\|\|+', '|', result)
    
    return result


def extract_and_fix_tables(content: str) -> str:
    """
    Extract tables from content, fix them, and reassemble the content.
    More comprehensive than fix_markdown_tables.
    
    Args:
        content: The full content
        
    Returns:
        Content with properly formatted tables
    """
    if not content or '|' not in content:
        return content
    
    # Normalize newlines
    result = content.replace('\r\n', '\n')
    
    # If a bold heading and a table are on the same line, split with a blank line
    result = re.sub(r'(\*\*[^*\n]+\*\*)(\s*\|)', r'\1\n\n\2', result)
    
    # Fix tables where separator incorrectly appears as content
    # Pattern: | text |---|---|
    result = re.sub(r'\|\s*([^|]+)\s*\|---\|---\|', r'| \1 |\n|---|---|', result)
    
    # Ensure separator rows are on their own lines
    result = re.sub(r'(\|[^|\n]+\|)\s*(\|[\s\-:]+\|)', r'\1\n\2', result)
    
    # Fix orphaned separators
    result = re.sub(r'\n\s*\|---\|---\|\s*\n', '\n|---|---|\n', result)
    
    # Collapse multiple pipes to single pipe (handles || -> |)
    result = re.sub(r'\|\|+', '|', result)
    
    lines = result.split('\n')
    output_lines: list[str] = []
    table_lines: list[str] = []
    
    def flush_table_block():
        nonlocal output_lines, table_lines
        if not table_lines:
            return
        # Normalize each table line with basic fixer
        block_text = '\n'.join(table_lines)
        block_text = fix_markdown_tables(block_text)
        fixed_block_lines = block_text.split('\n')
        # Ensure header separator exists
        if fixed_block_lines:
            has_separator = len(fixed_block_lines) > 1 and _is_separator_row(fixed_block_lines[1])
            if not has_separator:
                header = fixed_block_lines[0]
                cols = max(1, header.count('|') - 1)
                # Generate a correct separator row without an extra trailing pipe
                separator = '|' + '---|' * cols
                fixed_block_lines.insert(1, separator)
        # Ensure blank line before table
        if output_lines and output_lines[-1].strip() != '':
            output_lines.append('')
        output_lines.extend(fixed_block_lines)
        # Ensure blank line after table
        output_lines.append('')
        table_lines = []
    
    for line in lines:
        if _is_table_row(line):
            table_lines.append(line)
        else:
            if table_lines:
                flush_table_block()
            output_lines.append(line)
    
    # Flush any remaining table at EOF
    if table_lines:
        flush_table_block()
    
    # Join and do a final pass to remove any residual double pipes and tidy spacing
    final_text = '\n'.join(output_lines)
    final_text = re.sub(r'\|\|+', '|', final_text)
    return final_text


def _is_table_row(line: str) -> bool:
    """Check if a line looks like a table row using the central validator.

    Delegates to TableValidator to avoid overly-permissive detection that
    misclassifies orphan '|' lines or pipe-only artifacts as table rows.
    """
    try:
        return TableValidator._is_table_row(line)
    except Exception:
        # Fallback: be conservative
        stripped = (line or '').strip()
        if not stripped or '|' not in stripped:
            return False
        # Exclude lone or repeated pipes like '|' or '||'
        if re.fullmatch(r"\|+\s*", stripped):
            return False
        # Require at least two pipes and some non-pipe content
        return stripped.count('|') >= 2


def _fix_table_block(table_lines: List[str]) -> List[str]:
    """Fix a block of table lines."""
    if not table_lines:
        return []
    
    fixed_lines = []
    
    for i, line in enumerate(table_lines):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Fix the line
        fixed_line = line
        
        # Replace double pipes with single pipes
        fixed_line = re.sub(r'\|\|+', '|', fixed_line)
        
        # Ensure the line starts and ends with pipes if it looks like a table row
        stripped = fixed_line.strip()
        if '|' in stripped and not stripped.startswith('|'):
            # Try to add starting pipe
            fixed_line = '| ' + fixed_line.strip()
        if '|' in stripped and not stripped.endswith('|'):
            # Try to add ending pipe
            fixed_line = fixed_line.rstrip() + ' |'
        
        # Check if we need a separator row after header
        if i == 0 and not _has_separator_row(table_lines):
            # This is likely a header row, we need to add a separator
            fixed_lines.append(fixed_line)
            # Generate separator based on the number of columns
            cols = fixed_line.count('|') - 1 if fixed_line.count('|') > 1 else 1
            separator = '|' + '---|' * cols + '|'
            fixed_lines.append(separator)
        elif i == 1 and _is_separator_row(line):
            # Fix separator row
            fixed_line = re.sub(r'\|\|+', '|', fixed_line)
            # Ensure it has proper dashes
            if not re.search(r'[-:]+', fixed_line):
                # Convert to proper separator
                cols = fixed_line.count('|') - 1 if fixed_line.count('|') > 1 else 1  
                fixed_line = '|' + '---|' * cols
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(fixed_line)
    
    return fixed_lines


def _has_separator_row(table_lines: List[str]) -> bool:
    """Check if the table already has a separator row."""
    if len(table_lines) < 2:
        return False
    
    return _is_separator_row(table_lines[1])


def _is_separator_row(line: str) -> bool:
    """Check if a line is a table separator row."""
    stripped = line.strip()
    # Separator rows contain mostly dashes, pipes, and colons
    return bool(re.match(r'^[\s\|\-:]+$', stripped) and '-' in stripped)
