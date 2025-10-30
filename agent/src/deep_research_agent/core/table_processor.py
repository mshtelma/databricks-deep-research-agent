"""
Table processing utilities for markdown table manipulation.

This module provides utilities for processing markdown tables, including:
- Fixing malformed tables (missing separator rows)
- Validating table structure
- Normalizing table formatting

Extracted from reporter.py to provide reusable table processing logic.
"""

import re
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class TableProcessor:
    """
    Process and fix markdown tables.

    Provides utilities for ensuring markdown tables have proper structure,
    specifically inserting missing separator rows and normalizing formatting.

    Example:
        processor = TableProcessor()

        # Fix malformed table
        malformed = '''
        | Header 1 | Header 2 |
        | Data 1   | Data 2   |
        '''

        fixed = processor.fix_markdown_tables(malformed)
        # Returns:
        # | Header 1 | Header 2 |
        # | --- | --- |
        # | Data 1   | Data 2   |
    """

    def __init__(self, min_columns: int = 2):
        """
        Initialize table processor.

        Args:
            min_columns: Minimum number of columns to consider a valid table (default: 2)
        """
        self.min_columns = min_columns
        self.tables_fixed = 0

    def fix_markdown_tables(self, markdown_text: str) -> str:
        """
        Ensure ALL markdown tables have proper separator rows.

        This is a safety net that detects markdown tables and inserts missing
        separator rows to ensure proper rendering.

        Tables must have format:
        | Header 1 | Header 2 |
        |----------|----------|
        | Data 1   | Data 2   |

        This method aggressively detects tables and inserts missing separator rows.

        Args:
            markdown_text: Markdown content that may contain tables

        Returns:
            str: Markdown with all tables properly formatted

        Examples:
            >>> processor = TableProcessor()
            >>> text = "| A | B |\\n| 1 | 2 |"
            >>> fixed = processor.fix_markdown_tables(text)
            >>> "| ---" in fixed
            True
        """
        if not markdown_text or '|' not in markdown_text:
            return markdown_text

        lines = markdown_text.split('\n')
        result = []
        i = 0
        tables_fixed_in_run = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect table start (has pipes and enough content)
            if self._is_table_line(stripped):
                # Collect all consecutive table lines
                table_lines, next_index = self._collect_table_lines(lines, i)

                # Check if second line is separator
                if len(table_lines) >= 2:
                    second_line = table_lines[1].strip()
                    has_separator = self._is_separator_row(second_line)

                    if not has_separator:
                        # INSERT separator after header
                        header = table_lines[0]
                        separator = self._generate_separator(header)
                        table_lines.insert(1, separator)
                        tables_fixed_in_run += 1
                        logger.info(f"[TABLE FIX] Added missing separator to table at line {i+1}")

                result.extend(table_lines)
                i = next_index
            else:
                result.append(line)
                i += 1

        fixed_text = '\n'.join(result)

        # Track total fixes
        self.tables_fixed += tables_fixed_in_run

        # Log summary
        if tables_fixed_in_run > 0:
            logger.info(f"[TABLE FIX] Fixed {tables_fixed_in_run} tables with missing separators")

        return fixed_text

    def _is_table_line(self, line: str) -> bool:
        """
        Check if a line is part of a markdown table.

        Args:
            line: Stripped line to check

        Returns:
            bool: True if line appears to be a table row
        """
        return line.startswith('|') and line.count('|') >= (self.min_columns + 1)

    def _is_separator_row(self, line: str) -> bool:
        """
        Check if a line is a valid separator row.

        Valid separators contain pipes with dashes and optional colons for alignment.
        Examples: |---|---|, |:---|:---:|, | --- | --- |

        Args:
            line: Stripped line to check

        Returns:
            bool: True if line is a valid separator
        """
        # Proper separator: pipes with dashes and optional colons
        return bool(re.match(r'^\|[\s\-:|]+\|$', line))

    def _collect_table_lines(self, lines: List[str], start_index: int) -> Tuple[List[str], int]:
        """
        Collect all consecutive table lines starting from index.

        Args:
            lines: All lines in the document
            start_index: Index to start collecting from

        Returns:
            Tuple[List[str], int]: (table_lines, next_non_table_index)
        """
        table_lines = [lines[start_index]]
        j = start_index + 1

        while j < len(lines):
            next_stripped = lines[j].strip()
            if next_stripped.startswith('|') and next_stripped.count('|') >= self.min_columns:
                table_lines.append(lines[j])
                j += 1
            else:
                break

        return table_lines, j

    def _generate_separator(self, header: str) -> str:
        """
        Generate a separator row matching the header's column count.

        Args:
            header: Header row to match

        Returns:
            str: Properly formatted separator row

        Examples:
            >>> processor = TableProcessor()
            >>> processor._generate_separator("| A | B | C |")
            '| --- | --- | --- |'
        """
        col_count = header.count('|') - 1
        # Generate proper separator with spacing
        separator = '| ' + ' | '.join(['---'] * col_count) + ' |'
        return separator

    def get_stats(self) -> dict:
        """
        Get statistics about table processing.

        Returns:
            dict: Statistics including tables_fixed count
        """
        return {
            "tables_fixed": self.tables_fixed
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.tables_fixed = 0


# Convenience function for backward compatibility
def fix_markdown_tables(markdown_text: str) -> str:
    """
    Convenience function to fix markdown tables.

    Args:
        markdown_text: Markdown content

    Returns:
        str: Fixed markdown content
    """
    processor = TableProcessor()
    return processor.fix_markdown_tables(markdown_text)
