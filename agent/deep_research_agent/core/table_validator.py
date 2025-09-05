"""
Table validation utilities for markdown table integrity.

This module provides comprehensive validation and analysis tools for markdown tables,
ensuring they remain intact during streaming and processing.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TableIssue(Enum):
    """Types of table formatting issues."""
    DOUBLE_PIPES = "double_pipes"
    BROKEN_SEPARATOR = "broken_separator"
    COLUMN_MISMATCH = "column_mismatch"
    INCOMPLETE_ROW = "incomplete_row"
    MISSING_SEPARATOR = "missing_separator"
    MALFORMED_HEADER = "malformed_header"
    SPLIT_ROW = "split_row"
    INLINE_SEPARATOR = "inline_separator"


@dataclass
class Table:
    """Represents a parsed markdown table."""
    header: str
    separator: str
    rows: List[str]
    start_line: int
    end_line: int
    column_count: int
    
    def to_markdown(self) -> str:
        """Convert back to markdown format."""
        lines = [self.header, self.separator] + self.rows
        return '\n'.join(lines)
    
    @property
    def is_valid(self) -> bool:
        """Check if table structure is valid."""
        # All rows should have same column count
        for row in self.rows:
            if row.count('|') - 1 != self.column_count:
                return False
        return True


@dataclass
class ValidationResult:
    """Result of table validation."""
    valid: bool
    issues: List[Tuple[TableIssue, str]]
    tables: List[Table]
    statistics: Dict[str, Any]
    
    def get_error_summary(self) -> str:
        """Get a summary of validation errors."""
        if self.valid:
            return "No issues found"
        
        issue_counts = {}
        for issue_type, _ in self.issues:
            issue_counts[issue_type.value] = issue_counts.get(issue_type.value, 0) + 1
        
        summary = "Issues found:\n"
        for issue_type, count in issue_counts.items():
            summary += f"  - {issue_type}: {count}\n"
        return summary


class TableValidator:
    """Comprehensive table validation and analysis."""
    
    @staticmethod
    def is_valid_markdown_table(content: str) -> bool:
        """Quick check if content contains valid markdown tables."""
        tables = TableValidator.extract_tables(content)
        if not tables:
            return True  # No tables is valid
        
        for table in tables:
            if not table.is_valid:
                return False
        return True
    
    @staticmethod
    def extract_tables(content: str) -> List[Table]:
        """Extract all markdown tables from content."""
        tables = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            # Look for potential table start
            if TableValidator._is_table_row(lines[i]):
                table = TableValidator._extract_single_table(lines, i)
                if table:
                    tables.append(table)
                    i = table.end_line + 1
                else:
                    i += 1
            else:
                i += 1
        
        return tables
    
    @staticmethod
    def validate_table_structure(table_content: str) -> ValidationResult:
        """Comprehensive validation of table structure."""
        issues = []
        tables = []
        
        # Check for common malformed patterns
        if '||' in table_content:
            issues.append((TableIssue.DOUBLE_PIPES, "Double pipes found in table"))
        
        if re.search(r'\|---|---|---|---|---|', table_content):
            issues.append((TableIssue.BROKEN_SEPARATOR, "Broken separator pattern detected"))
        
        if re.search(r'\|\s*\w+.*\|---|---|', table_content):
            issues.append((TableIssue.INLINE_SEPARATOR, "Separator inline with content"))
        
        # Extract and validate tables
        extracted_tables = TableValidator.extract_tables(table_content)
        
        for table in extracted_tables:
            # Validate header
            if not TableValidator._is_table_row(table.header):
                issues.append((TableIssue.MALFORMED_HEADER, f"Invalid header: {table.header}"))
            
            # Validate separator
            if not TableValidator._is_separator_row(table.separator):
                issues.append((TableIssue.MISSING_SEPARATOR, f"Invalid separator: {table.separator}"))
            
            # Check column consistency
            header_cols = table.header.count('|') - 1
            separator_cols = table.separator.count('|') - 1
            
            if header_cols != separator_cols:
                issues.append((
                    TableIssue.COLUMN_MISMATCH,
                    f"Header has {header_cols} columns, separator has {separator_cols}"
                ))
            
            # Check each data row
            for row_idx, row in enumerate(table.rows):
                row_cols = row.count('|') - 1
                if row_cols != table.column_count:
                    issues.append((
                        TableIssue.COLUMN_MISMATCH,
                        f"Row {row_idx + 1} has {row_cols} columns, expected {table.column_count}"
                    ))
                
                if not row.strip().startswith('|') or not row.strip().endswith('|'):
                    issues.append((
                        TableIssue.INCOMPLETE_ROW,
                        f"Row {row_idx + 1} missing start or end pipe"
                    ))
            
            tables.append(table)
        
        # Calculate statistics
        statistics = {
            "table_count": len(tables),
            "total_rows": sum(len(t.rows) for t in tables),
            "average_columns": sum(t.column_count for t in tables) / len(tables) if tables else 0,
            "issue_count": len(issues)
        }
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            tables=tables,
            statistics=statistics
        )
    
    @staticmethod
    def detect_malformed_patterns(content: str) -> List[Tuple[TableIssue, int, str]]:
        """Detect specific malformed patterns with line numbers."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Double pipes
            if '||' in line:
                issues.append((TableIssue.DOUBLE_PIPES, i, line))
            
            # Broken separators
            if re.match(r'^[\s|]+[-|]+[-|]+[-|]+[-|]+', line):
                if line.count('|') > 6:  # Likely broken
                    issues.append((TableIssue.BROKEN_SEPARATOR, i, line))
            
            # Inline separators
            if re.search(r'\w+.*\|---', line):
                issues.append((TableIssue.INLINE_SEPARATOR, i, line))
            
            # Split rows (pipes without proper spacing)
            if '|' in line:
                # Check if it looks like a broken row
                if not line.strip().startswith('|') and '|' in line:
                    if not any(line.strip().startswith(x) for x in ['#', '*', '-', '+']):
                        issues.append((TableIssue.SPLIT_ROW, i, line))
        
        return issues
    
    @staticmethod
    def fix_common_issues(content: str) -> str:
        """Attempt to fix common table issues."""
        # Fix double pipes
        content = re.sub(r'\|\|+', '|', content)
        
        # Fix inline separators
        content = re.sub(r'(\|[^|\n]+)\|(---+\|)', r'\1\n\2', content)
        
        # Fix broken separator patterns
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix excessive separator columns
            if re.match(r'^\s*\|[\s-:|]+\|$', line) and line.count('|') > 2:
                # This looks like a separator
                cols = line.count('|') - 1
                if cols > 10:  # Likely broken, rebuild it
                    # Try to infer correct column count from nearby rows
                    fixed_lines.append('|' + '---|' * 5)  # Default to 5 columns
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    @staticmethod
    def can_be_chunked_at(content: str, position: int) -> bool:
        """Check if content can be safely chunked at given position without breaking tables."""
        # Get lines before and after position
        before = content[:position]
        after = content[position:]
        
        # Check if we're in the middle of a table
        before_lines = before.split('\n')
        after_lines = after.split('\n')
        
        # Look for table indicators
        recent_table_lines = 0
        for line in reversed(before_lines[-5:]):  # Check last 5 lines
            if TableValidator._is_table_row(line):
                recent_table_lines += 1
        
        # If we have table lines recently and the next line is also a table line, don't chunk here
        if recent_table_lines > 0 and after_lines and TableValidator._is_table_row(after_lines[0]):
            return False
        
        # Check if we're splitting a line
        if position > 0 and position < len(content):
            if content[position - 1] != '\n' and content[position] != '\n':
                # We're in the middle of a line
                # Check if this line contains table formatting
                line_start = content.rfind('\n', 0, position) + 1
                line_end = content.find('\n', position)
                if line_end == -1:
                    line_end = len(content)
                
                line = content[line_start:line_end]
                if '|' in line:
                    return False  # Don't split table rows
        
        return True
    
    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if a line looks like a table row."""
        line = line.strip()
        if not line:
            return False
        
        # Must have pipes
        if '|' not in line:
            return False
        
        # Should start and end with pipe (with possible whitespace)
        if not (line.startswith('|') and line.endswith('|')):
            # Allow some flexibility for malformed tables
            if line.count('|') < 2:
                return False
        
        return True
    
    @staticmethod
    def _is_separator_row(line: str) -> bool:
        """Check if a line is a table separator."""
        line = line.strip()
        if not TableValidator._is_table_row(line):
            return False
        
        # Remove pipes and spaces
        content = line.replace('|', '').replace(' ', '')
        
        # Should be mostly dashes and colons
        if not content:
            return False
        
        dash_colon_count = sum(1 for c in content if c in '-:')
        return dash_colon_count / len(content) > 0.8 and '-' in content
    
    @staticmethod
    def _extract_single_table(lines: List[str], start_idx: int) -> Optional[Table]:
        """Extract a single table starting from given index."""
        if start_idx >= len(lines):
            return None
        
        # First line should be header
        header = lines[start_idx].strip()
        if not TableValidator._is_table_row(header):
            return None
        
        # Second line should be separator
        if start_idx + 1 >= len(lines):
            return None
        
        separator = lines[start_idx + 1].strip()
        if not TableValidator._is_separator_row(separator):
            return None
        
        # Collect data rows
        rows = []
        end_idx = start_idx + 1
        
        for i in range(start_idx + 2, len(lines)):
            if TableValidator._is_table_row(lines[i]):
                rows.append(lines[i].strip())
                end_idx = i
            else:
                break
        
        column_count = header.count('|') - 1
        
        return Table(
            header=header,
            separator=separator,
            rows=rows,
            start_line=start_idx,
            end_line=end_idx,
            column_count=column_count
        )


class StreamingTableReconstructor:
    """Reconstruct tables from streaming chunks."""
    
    def __init__(self):
        self.buffer = ""
        self.completed_content = ""
        self.current_table_buffer = []
        self.in_table = False
    
    def add_chunk(self, chunk: str) -> str:
        """Add a streaming chunk and return any completed content."""
        self.buffer += chunk
        completed = ""
        
        # Process complete lines
        while '\n' in self.buffer:
            line_end = self.buffer.index('\n')
            line = self.buffer[:line_end]
            self.buffer = self.buffer[line_end + 1:]
            
            # Process the line
            processed = self._process_line(line)
            if processed:
                completed += processed + '\n'
        
        return completed
    
    def finalize(self) -> str:
        """Get any remaining content."""
        result = ""
        
        # Process remaining buffer
        if self.buffer:
            result = self._process_line(self.buffer)
            self.buffer = ""
        
        # Flush table buffer if needed
        if self.current_table_buffer:
            result += '\n'.join(self.current_table_buffer)
            self.current_table_buffer = []
        
        return result
    
    def _process_line(self, line: str) -> str:
        """Process a single line with enhanced table detection."""
        trimmed = line.strip()
        
        if TableValidator._is_table_row(line):
            col_count = self._get_column_count(line)
            
            # Check for table continuation
            if not self.in_table and self.last_table_columns > 0:
                if abs(col_count - self.last_table_columns) <= 2:
                    self.in_table = True
                    self.current_table_buffer = [line]
                    self.min_columns = col_count
                    self.max_columns = col_count
            elif not self.in_table:
                self.in_table = True
                self.current_table_buffer = [line]
                self.min_columns = col_count
                self.max_columns = col_count
            else:
                self.current_table_buffer.append(line)
                # Update column range
                if col_count < self.min_columns:
                    self.min_columns = col_count
                if col_count > self.max_columns:
                    self.max_columns = col_count
            
            self.empty_line_count = 0
            return ""  # Buffer table lines
            
        elif self.in_table:
            # Handle content within tables
            if trimmed == '':
                self.empty_line_count += 1
                if self.empty_line_count <= 2:
                    self.current_table_buffer.append(line)
                    return ""
                else:
                    # Too many empty lines, end table
                    return self._flush_table() + ('\n' + line if line.strip() else '')
            elif self._is_table_related_content(trimmed):
                # Keep related content with table
                self.current_table_buffer.append(line)
                self.related_content.append(line)
                self.empty_line_count = 0
                return ""
            else:
                # End table
                result = self._flush_table()
                if line.strip():
                    result += '\n' + line
                return result
        else:
            # Regular non-table line
            if self.empty_line_count > 3:
                self.last_table_columns = 0
            self.empty_line_count += 1
            return line
    
    def _flush_table(self) -> str:
        """Flush current table buffer."""
        if not self.current_table_buffer:
            return ""
        
        table_content = '\n'.join(self.current_table_buffer)
        
        # Quick validation
        if TableValidator.is_valid_markdown_table(table_content):
            result = table_content
        else:
            # Try to fix common issues
            result = TableValidator.fix_common_issues(table_content)
        
        self.last_table_columns = self.max_columns
        self.current_table_buffer = []
        self.in_table = False
        self.min_columns = 0
        self.max_columns = 0
        self.empty_line_count = 0
        
        return result
    
    def _get_column_count(self, line: str) -> int:
        """Get column count from a table row."""
        line = line.strip()
        if not line or '|' not in line:
            return 0
        
        pipes = line.count('|')
        if line.startswith('|') and line.endswith('|'):
            return max(1, pipes - 1)
        return max(1, pipes)
    
    def _is_table_related_content(self, line: str) -> bool:
        """Check if line is table-related content."""
        patterns = [
            r'^\*\*[^*]+\*\*$',
            r'^Note:',
            r'^Source:',
            r'^\d+\.',
            r'^[A-Z][^.!?]*:$',
            r'^Total:',
            r'^Summary:',
            r'^\([^)]+\)$',
        ]
        
        import re
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns)


# Convenience functions
def validate_table(content: str) -> bool:
    """Quick validation of table content."""
    return TableValidator.is_valid_markdown_table(content)


def extract_tables(content: str) -> List[Table]:
    """Extract all tables from content."""
    return TableValidator.extract_tables(content)


def fix_table_issues(content: str) -> str:
    """Fix common table issues in content."""
    return TableValidator.fix_common_issues(content)