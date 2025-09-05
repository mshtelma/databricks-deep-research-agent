"""
Comprehensive table preprocessing to ensure valid markdown tables.

This module handles all known table malformation patterns before content
is streamed to the UI, ensuring consistent and properly formatted tables.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..core import get_logger

logger = get_logger(__name__)


@dataclass
class TableStats:
    """Statistics about table preprocessing."""
    tables_fixed: int = 0
    patterns_fixed: Dict[str, int] = None
    issues_resolved: List[str] = None
    
    def __post_init__(self):
        if self.patterns_fixed is None:
            self.patterns_fixed = {}
        if self.issues_resolved is None:
            self.issues_resolved = []


class TablePreprocessor:
    """
    Comprehensive table preprocessing to ensure valid markdown tables.
    Handles all known malformation patterns before content streaming.
    """
    
    def __init__(self):
        self.stats = TableStats()
        self.table_start_marker = "TABLE_START"
        self.table_end_marker = "TABLE_END"
        
    def preprocess_tables(self, content: str) -> str:
        """
        Main entry point for table preprocessing.
        
        Args:
            content: Content potentially containing malformed tables
            
        Returns:
            Content with properly formatted tables
        """
        if not content or '|' not in content:
            return content
            
        logger.info("Starting comprehensive table preprocessing")
        self.stats = TableStats()  # Reset stats
        
        # Step 0: Separate headings from tables
        content = self.separate_headings_from_tables(content)
        
        # Step 1: Fix two-line cell patterns (Item    Description format)
        content = self.fix_two_line_cells(content)
        
        # Step 2: Fix merged table components
        content = self.fix_merged_table_components(content)
        
        # Step 3: Remove trailing separators first (before mixed content processing)
        content = self.remove_trailing_separators(content)
        
        # Step 4: Fix mixed content-separator patterns
        content = self.fix_mixed_content_separators(content)
        
        # Step 5: Normalize separators
        content = self.normalize_separators(content)
        
        # Step 6: Consolidate orphaned separators with adjacent data
        content = self.consolidate_orphaned_separators(content)
        
        # Step 7: Consolidate adjacent table fragments
        content = self.consolidate_table_fragments(content)
        
        # Step 8: Extract tables and fix column mismatches
        content = self.fix_tables_column_counts(content)
        
        # Step 9: Final cleanup
        content = self.final_cleanup(content)
        
        # Step 10: Add table boundaries for complex tables
        content = self.add_table_boundaries(content)
        
        logger.info(
            "Table preprocessing complete",
            stats=self.get_stats()
        )
        
        return content
    
    def fix_two_line_cells(self, content: str) -> str:
        """
        Fix the pattern where table cells span two lines.
        
        Example:
        Item    Description
        | Reference year
        2024 tax year...
        
        Converts to:
        | Item | Description |
        | --- | --- |
        | Reference year | 2024 tax year... |
        """
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect the header pattern "Item    Description" or similar
            if (not line.startswith('|') and 
                '    ' in line and  # Multiple spaces suggesting columns
                len(line.split()) >= 2 and
                i + 1 < len(lines) and
                lines[i + 1].strip().startswith('|')):
                
                # This looks like a two-column header
                parts = re.split(r'\s{2,}', line)
                if len(parts) == 2:
                    logger.info(f"Detected two-line cell pattern at line {i}")
                    self.stats.patterns_fixed['two_line_cells'] = self.stats.patterns_fixed.get('two_line_cells', 0) + 1
                    
                    # Add proper header
                    fixed_lines.append(f"| {parts[0]} | {parts[1]} |")
                    fixed_lines.append("| --- | --- |")
                    
                    # Process following lines that are part of this table
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        
                        if next_line.startswith('|') and i + 1 < len(lines):
                            # Get the item (current line) and description (next line)
                            item = next_line.lstrip('|').strip()
                            i += 1
                            description = lines[i].strip()
                            
                            # Combine into single row
                            fixed_lines.append(f"| {item} | {description} |")
                        elif not next_line or not next_line.startswith('|'):
                            # End of this table pattern
                            break
                        i += 1
                    continue
            
            fixed_lines.append(lines[i])
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def separate_headings_from_tables(self, content: str) -> str:
        """
        Separate headings that are attached to table headers.
        
        Example:
        **Heading**| Header1 | Header2 |
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check if line starts with bold text followed by table
            if '**' in line and '|' in line:
                # Find where the table starts
                pipe_idx = line.find('|')
                # Check if there's content before the first pipe
                before_pipe = line[:pipe_idx].strip()
                if before_pipe and not before_pipe.endswith('|'):
                    # Split heading from table
                    fixed_lines.append(before_pipe)
                    fixed_lines.append(line[pipe_idx:])
                    self.stats.patterns_fixed['separated_headings'] = self.stats.patterns_fixed.get('separated_headings', 0) + 1
                    logger.info("Separated heading from table")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_merged_table_components(self, content: str) -> str:
        """
        Fix headers-separators-data merged on single line.
        
        Example:
        | Header1 | Header2 || --- | --- | --- | Data1 | Data2 |
        
        This should ONLY trigger for actual merged table structure, not for
        content that happens to have separators mixed in.
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if '|' in line and '---' in line:
                # Check if this looks like a true merged table pattern
                parts = line.split('|')
                cleaned_parts = [p.strip() for p in parts if p.strip()]
                
                # Count consecutive separators
                consecutive_seps = 0
                max_consecutive = 0
                for part in cleaned_parts:
                    if re.match(r'^:?-+:?$', part):
                        consecutive_seps += 1
                        max_consecutive = max(max_consecutive, consecutive_seps)
                    else:
                        consecutive_seps = 0
                
                # Only treat as merged if we have multiple consecutive separators
                # (indicating a separator row) AND the separator count roughly matches
                # the expected column count (not just random separators mixed in)
                if max_consecutive >= 3:  # Increased threshold
                    headers = []
                    separators = []
                    data = []
                    
                    mode = 'headers'
                    for part in cleaned_parts:
                        if re.match(r'^:?-+:?$', part):
                            if mode == 'headers':
                                mode = 'separators'
                            separators.append(part)
                        elif mode == 'separators' and not re.match(r'^:?-+:?$', part):
                            mode = 'data'
                            data.append(part)
                        elif mode == 'headers':
                            headers.append(part)
                        elif mode == 'data':
                            data.append(part)
                    
                    # Only split if we have all three components AND the pattern makes sense
                    # Headers should be roughly equal in count to separators
                    if headers and len(separators) >= 3 and data and abs(len(headers) - len(separators)) <= 1:
                        logger.info("Fixed merged table components")
                        self.stats.patterns_fixed['merged_components'] = self.stats.patterns_fixed.get('merged_components', 0) + 1
                        
                        # Ensure column count matches
                        col_count = max(len(headers), len(data))
                        while len(headers) < col_count:
                            headers.append('')
                        while len(data) < col_count:
                            data.append('')
                        
                        fixed_lines.append('| ' + ' | '.join(headers) + ' |')
                        fixed_lines.append('|' + ' --- |' * col_count)
                        fixed_lines.append('| ' + ' | '.join(data) + ' |')
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_mixed_content_separators(self, content: str) -> str:
        """
        Fix lines with content mixed with separator patterns.
        
        Example:
        | Country | Data | --- | --- | Spain | Value |
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Check for mixed content and separators more comprehensively
            if '|' in line and '---' in line:
                # Check if this is a pure separator line
                if re.match(r'^\|\s*(---\s*\|)+\s*$', line) or re.match(r'^\|\s*(:?-+:?\s*\|)+\s*$', line):
                    # Pure separator, keep as is
                    fixed_lines.append(line)
                else:
                    # Mixed content and separators
                    parts = line.split('|')
                    content_parts = []
                    separator_parts = []
                    
                    for part in parts:
                        part = part.strip()
                        if re.match(r'^:?-+:?$', part):
                            separator_parts.append(part)
                        elif part:
                            content_parts.append(part)
                    
                    if content_parts and separator_parts:
                        # We have both content and separators mixed
                        logger.info(f"Fixed mixed content-separator line with {len(content_parts)} content parts and {len(separator_parts)} separator parts")
                        self.stats.patterns_fixed['mixed_content'] = self.stats.patterns_fixed.get('mixed_content', 0) + 1
                        # Keep only content
                        fixed_lines.append('| ' + ' | '.join(content_parts) + ' |')
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def normalize_separators(self, content: str) -> str:
        """
        Ensure exactly one separator row per table with correct column count.
        Remove duplicate separators and fix malformed separator patterns.
        """
        lines = content.split('\n')
        fixed_lines = []
        last_was_separator = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is a separator line
            is_separator = False
            if line_stripped.startswith('|') and '---' in line_stripped:
                # Check if it's a pure separator line
                parts = line_stripped.split('|')
                cleaned_parts = [p.strip() for p in parts if p.strip()]
                is_separator = all(re.match(r'^:?-+:?$', p) for p in cleaned_parts)
            
            # Handle condensed separator patterns |---|---|---|
            if re.match(r'^\|(-{2,}\|)+\s*$', line_stripped):
                # Convert to proper format
                count = line_stripped.count('---')
                line = '|' + ' --- |' * count
                is_separator = True
                self.stats.patterns_fixed['condensed_separators'] = self.stats.patterns_fixed.get('condensed_separators', 0) + 1
            
            if is_separator:
                if not last_was_separator:
                    # Normalize the separator format
                    parts = line_stripped.split('|')
                    cleaned_parts = [p.strip() for p in parts if p.strip()]
                    normalized = '|' + ' --- |' * len(cleaned_parts)
                    fixed_lines.append(normalized)
                    last_was_separator = True
                else:
                    # Skip duplicate separator
                    logger.info("Removed duplicate separator row")
                    self.stats.patterns_fixed['duplicate_separators'] = self.stats.patterns_fixed.get('duplicate_separators', 0) + 1
            else:
                fixed_lines.append(line)
                # Only reset last_was_separator if we have actual content, not just blank line
                if line_stripped:
                    last_was_separator = False
        
        return '\n'.join(fixed_lines)
    
    def consolidate_orphaned_separators(self, content: str) -> str:
        """
        Consolidate orphaned separator rows with adjacent table data.
        This handles cases where we have a separator followed by data without a header.
        """
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line_raw = lines[i]
            line = line_raw.strip()
            
            # Check if this is a separator line
            if line and (re.match(r'^\|(\s*---\s*\|)+\s*$', line) or 
                         re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', line)):
                
                logger.debug(f"Found separator at line {i}: {repr(line)}")
                
                # Look backward to see if this separator is immediately after a header line
                has_valid_header = False
                
                # Check the immediately preceding non-blank line
                for j in range(i - 1, -1, -1):
                    prev_line = lines[j].strip()
                    if prev_line:
                        logger.debug(f"  Previous non-blank line {j}: {repr(prev_line)}")
                        # Found the previous non-blank line
                        # Check if it's a table row (not a separator, has pipes, and has actual content)
                        # A single pipe or empty pipes are not valid headers
                        if ('|' in prev_line and 
                            prev_line.count('|') >= 2 and  # At least 2 pipes for a valid row
                            not re.match(r'^\|\s*\|*\s*$', prev_line) and  # Not just pipes with no content
                            not re.match(r'^\|(\s*---\s*\|)+\s*$', prev_line) and
                            not re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', prev_line)):
                            # This separator has a valid header line above it
                            has_valid_header = True
                            logger.debug(f"  -> Has valid header")
                        else:
                            logger.debug(f"  -> Not a valid header")
                        break
                
                if not has_valid_header:
                    # This is an orphaned separator, look for data below
                    data_lines = []
                    j = i + 1
                    
                    # Skip blank lines
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    
                    # Collect data lines
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line and '|' in next_line:
                            # Check if it's another separator
                            if (not re.match(r'^\|(\s*---\s*\|)+\s*$', next_line) and
                                not re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', next_line)):
                                data_lines.append(lines[j])  # Keep original line with spacing
                            j += 1
                        else:
                            break
                    
                    if data_lines:
                        # We have orphaned separator with data, consolidate them
                        # Don't add the orphaned separator - let the data stand alone
                        # The table will get its separator added later if needed
                        logger.info(f"Removing orphaned separator, keeping {len(data_lines)} data lines")
                        self.stats.patterns_fixed['orphaned_separators'] = self.stats.patterns_fixed.get('orphaned_separators', 0) + 1
                        # Skip the orphaned separator and add the data lines
                        fixed_lines.extend(data_lines)
                        i = j
                        continue
                    else:
                        # Orphaned separator with no data - just remove it
                        logger.info("Removing orphaned separator with no data")
                        self.stats.patterns_fixed['orphaned_separators'] = self.stats.patterns_fixed.get('orphaned_separators', 0) + 1
                        i += 1
                        continue
            
            fixed_lines.append(line_raw)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def consolidate_table_fragments(self, content: str) -> str:
        """
        Consolidate table fragments that are separated only by blank lines.
        This handles cases where normalize_separators creates disconnected fragments.
        """
        lines = content.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this starts a table fragment
            if line.strip() and '|' in line.strip():
                # Collect all table lines that are close together
                table_lines = []
                j = i
                
                # First, add the current line
                table_lines.append(lines[j])
                last_table_line = j
                j += 1
                
                # Collect this table and any fragments that follow with small gaps (≤2 blank lines)
                while j < len(lines):
                    if lines[j].strip() and '|' in lines[j].strip():
                        # Found another table line, add gap lines if any
                        for k in range(last_table_line + 1, j):
                            table_lines.append(lines[k])
                        table_lines.append(lines[j])
                        last_table_line = j
                        j += 1
                    elif j - last_table_line > 2:
                        # Too big a gap (more than 2 blank lines), end of table
                        break
                    else:
                        # Still within acceptable gap
                        j += 1
                
                # Process the collected table lines to remove duplicate separators
                processed = self._process_table_fragment(table_lines)
                result.extend(processed)
                
                # Skip past all the lines we've processed
                i = last_table_line + 1
            else:
                result.append(line)
                i += 1
        
        return '\n'.join(result)
    
    def _process_table_fragment(self, lines: List[str]) -> List[str]:
        """Process a table fragment to remove duplicate separators."""
        if not lines:
            return lines
        
        processed = []
        has_separator = False  # Track if we've already added a separator to this table
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is a separator
            is_separator = False
            if line_stripped and '|' in line_stripped:
                if (re.match(r'^\|(\s*---\s*\|)+\s*$', line_stripped) or
                    re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', line_stripped)):
                    is_separator = True
            
            if is_separator:
                if not has_separator:
                    # This is the first separator in the table, keep it
                    processed.append(line)
                    has_separator = True
                else:
                    # Skip any additional separators
                    logger.debug("Skipping duplicate separator in fragment consolidation")
            else:
                processed.append(line)
        
        # Check if we have proper structure
        has_content = any(line.strip() and '|' in line and 
                         not re.match(r'^\|(\s*---\s*\|)+\s*$', line.strip()) and
                         not re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', line.strip())
                         for line in processed)
        
        if has_content and len(processed) > 1:
            self.stats.patterns_fixed['consolidated_fragments'] = \
                self.stats.patterns_fixed.get('consolidated_fragments', 0) + 1
        
        return processed
    
    def fix_tables_column_counts(self, content: str) -> str:
        """
        Ensure all rows in a table have the same column count.
        """
        lines = content.split('\n')
        fixed_lines = []
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line.strip():
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table:
                    # End of table, fix column counts and ensure separator
                    fixed_table = self.fix_table_column_count(table_lines)
                    fixed_table = self.ensure_table_separator(fixed_table)
                    fixed_lines.extend(fixed_table)
                    in_table = False
                    table_lines = []
                fixed_lines.append(line)
        
        # Handle table at end of content
        if in_table and table_lines:
            fixed_table = self.fix_table_column_count(table_lines)
            fixed_table = self.ensure_table_separator(fixed_table)
            fixed_lines.extend(fixed_table)
        
        return '\n'.join(fixed_lines)
    
    def ensure_table_separator(self, table_lines: List[str]) -> List[str]:
        """
        Ensure table has a separator row after the header.
        """
        if len(table_lines) < 2:
            return table_lines
        
        # Check if there's already a separator
        has_separator = False
        separator_idx = -1
        
        for i, line in enumerate(table_lines):
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                has_separator = True
                separator_idx = i
                break
        
        if not has_separator and len(table_lines) >= 1:
            # No separator found, add one after first row (header)
            # Count columns in header
            header = table_lines[0]
            parts = header.split('|')
            if header.strip().startswith('|') and header.strip().endswith('|'):
                col_count = len(parts[1:-1]) if len(parts) > 2 else 1
            else:
                col_count = len([p for p in parts if p.strip()])
            
            # Insert separator after header
            result = [table_lines[0]]
            result.append('|' + ' --- |' * col_count)
            result.extend(table_lines[1:])
            
            self.stats.patterns_fixed['added_separator'] = self.stats.patterns_fixed.get('added_separator', 0) + 1
            logger.info("Added missing separator row to table")
            return result
        
        return table_lines
    
    def fix_table_column_count(self, table_lines: List[str]) -> List[str]:
        """
        Fix column count mismatches in a single table.
        """
        if not table_lines:
            return table_lines
        
        # Find the target column count (usually from header or most common)
        column_counts = []
        separator_line_idx = -1
        
        for i, line in enumerate(table_lines):
            if '|' in line:
                # Check if it's a separator line
                if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                    separator_line_idx = i
                    # Count separators
                    parts = line.split('|')
                    count = len([p.strip() for p in parts if p.strip() and re.match(r'^:?-+:?$', p.strip())])
                    if count > 0:
                        column_counts.append(count)
                else:
                    # Regular content line
                    parts = line.split('|')
                    # For standard table format |cell|cell|, count cells between pipes
                    if line.strip().startswith('|') and line.strip().endswith('|'):
                        # Remove first and last empty parts
                        inner_parts = parts[1:-1] if len(parts) > 2 else []
                        count = len(inner_parts)
                    else:
                        # Non-standard format
                        count = len([p for p in parts if p.strip()])
                    if count > 0:
                        column_counts.append(count)
        
        if not column_counts:
            return table_lines
        
        # Use the most common column count
        target_count = max(set(column_counts), key=column_counts.count)
        
        fixed_lines = []
        for line in table_lines:
            if '|' in line:
                # Check if separator line
                if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                    # Normalize separator to target count
                    fixed_line = '|' + ' --- |' * target_count
                    fixed_lines.append(fixed_line)
                else:
                    # Regular content line
                    parts = line.split('|')
                    
                    # Extract actual content (handle pipe-delimited format properly)
                    if line.strip().startswith('|') and line.strip().endswith('|'):
                        # Standard table format |cell|cell|
                        # Take parts between first and last pipe
                        content_parts = [p.strip() for p in parts[1:-1]] if len(parts) > 2 else []
                    else:
                        # Non-standard format - take all non-empty parts
                        content_parts = [p.strip() for p in parts if p.strip()]
                    
                    # Adjust to target column count
                    while len(content_parts) < target_count:
                        content_parts.append('')
                    while len(content_parts) > target_count:
                        # Merge excess columns
                        if len(content_parts) > 1:
                            content_parts[-2] = f"{content_parts[-2]} {content_parts[-1]}".strip()
                        content_parts.pop()
                    
                    # Reconstruct line
                    fixed_line = '| ' + ' | '.join(content_parts) + ' |'
                    fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        # Check if we made any fixes
        if fixed_lines != table_lines:
            self.stats.patterns_fixed['column_mismatches'] = self.stats.patterns_fixed.get('column_mismatches', 0) + 1
            self.stats.tables_fixed += 1
        
        return fixed_lines
    
    def remove_trailing_separators(self, content: str) -> str:
        """
        Remove trailing separator patterns from content rows.
        
        Example:
        | Country | Value | --- |
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Specifically check for trailing separator pattern
            if '|' in line:
                parts = line.split('|')
                
                # Check if line ends with separator pattern(s)
                trailing_sep_count = 0
                for i in range(len(parts) - 1, -1, -1):
                    part = parts[i].strip()
                    if part == '':
                        continue  # Skip empty parts (like the one after final |)
                    elif re.match(r'^:?-+:?$', part):
                        trailing_sep_count += 1
                    else:
                        break  # Found non-separator content
                
                # If we have trailing separators and some content
                if trailing_sep_count > 0 and len(parts) > trailing_sep_count + 2:
                    # Remove the trailing separators
                    cleaned_parts = []
                    for i, part in enumerate(parts):
                        part_stripped = part.strip()
                        # Skip trailing separators
                        if i >= len(parts) - trailing_sep_count - 1 and re.match(r'^:?-+:?$', part_stripped):
                            continue
                        cleaned_parts.append(part)
                    
                    # Reconstruct the line
                    fixed_line = '|'.join(cleaned_parts)
                    
                    # Ensure proper formatting
                    if not fixed_line.endswith('|'):
                        fixed_line += '|'
                    
                    fixed_lines.append(fixed_line)
                    self.stats.patterns_fixed['trailing_separators'] = self.stats.patterns_fixed.get('trailing_separators', 0) + 1
                    logger.info(f"Removed {trailing_sep_count} trailing separator(s) from line")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def final_cleanup(self, content: str) -> str:
        """
        Final cleanup pass for any remaining issues.
        """
        # Remove any remaining double pipes
        content = re.sub(r'\|\|+', '|', content)
        
        # Remove standalone pipe lines
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty pipe rows and standalone pipes
            if stripped in ['|', '||'] or re.match(r'^\|\s*(\|\s*)*$', stripped):
                self.stats.patterns_fixed['empty_pipes'] = self.stats.patterns_fixed.get('empty_pipes', 0) + 1
                continue
            # Skip orphaned separator patterns
            if stripped == '---' or re.match(r'^-{3,}$', stripped):
                self.stats.patterns_fixed['orphaned_separators'] = self.stats.patterns_fixed.get('orphaned_separators', 0) + 1
                continue
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Remove excessive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def add_table_boundaries(self, content: str) -> str:
        """
        Add TABLE_START/TABLE_END markers for complex tables.
        This helps the UI know when a complete table has been received.
        """
        lines = content.split('\n')
        marked_lines = []
        in_table = False
        table_lines = []
        has_separator = False
        has_content = False
        
        for i, line in enumerate(lines):
            if '|' in line.strip():
                # Skip orphaned separators (lone separator lines not part of a table)
                if re.match(r'^\|[\s\-:|]+\|$', line.strip()) and not in_table:
                    # Look ahead to see if next line is table content
                    next_is_table = False
                    if i + 1 < len(lines) and '|' in lines[i + 1]:
                        next_line = lines[i + 1].strip()
                        if not re.match(r'^\|[\s\-:|]+\|$', next_line):
                            next_is_table = True
                    
                    if not next_is_table:
                        # This is an orphaned separator, don't start a table
                        marked_lines.append(line)
                        continue
                
                if not in_table:
                    in_table = True
                    table_lines = [line]
                    # Check if this is a separator
                    if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                        has_separator = True
                    else:
                        has_content = True
                else:
                    table_lines.append(line)
                    # Track if we have both separator and content
                    if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                        has_separator = True
                    else:
                        has_content = True
            else:
                if in_table:
                    # End of table - check if it's a valid table
                    # Valid table must have both content and separator (or be large enough)
                    is_valid_table = (has_separator and has_content) or len(table_lines) >= 3
                    
                    if is_valid_table:
                        # This is a substantial table, add markers
                        marked_lines.append(self.table_start_marker)
                        marked_lines.extend(table_lines)
                        marked_lines.append(self.table_end_marker)
                    else:
                        # Not a valid table structure, don't mark
                        marked_lines.extend(table_lines)
                    
                    in_table = False
                    table_lines = []
                    has_separator = False
                    has_content = False
                    
                marked_lines.append(line)
        
        # Handle table at end of content
        if in_table and table_lines:
            is_valid_table = (has_separator and has_content) or len(table_lines) >= 3
            if is_valid_table:
                marked_lines.append(self.table_start_marker)
                marked_lines.extend(table_lines)
                marked_lines.append(self.table_end_marker)
            else:
                marked_lines.extend(table_lines)
        
        return '\n'.join(marked_lines)
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get preprocessing statistics.
        """
        return {
            'tables_fixed': self.stats.tables_fixed,
            'patterns_fixed': self.stats.patterns_fixed,
            'total_fixes': sum(self.stats.patterns_fixed.values()),
            'issues_resolved': len(self.stats.issues_resolved)
        }