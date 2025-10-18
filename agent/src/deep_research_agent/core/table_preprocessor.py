"""
Comprehensive table preprocessing to ensure valid markdown tables.

This module handles all known table malformation patterns before content
is streamed to the UI, ensuring consistent and properly formatted tables.
"""

import re
from typing import List, Dict, Optional
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
        self.table_header_start_marker = "TABLE_HEADER_START"
        self.table_header_end_marker = "TABLE_HEADER_END"
        # Enhanced markers for better structure tracking
        self.header_row_marker = "§§HEADER_ROW§§"
        self.separator_marker = "§§SEPARATOR§§"
        self.data_start_marker = "§§DATA_START§§"
        self.data_end_marker = "§§DATA_END§§"
        self.footnotes_marker = "§§FOOTNOTES§§"
        self.protected_sections = {}  # Store protected reference sections
        self.footnote_placeholders = {}  # Store footnote markers during processing
        self.table_counter = 0  # For generating unique table IDs
        
    def preprocess_tables(self, content: str) -> str:
        """
        Main entry point for table preprocessing.
        
        Args:
            content: Content potentially containing malformed tables
            
        Returns:
            Content with properly formatted tables
        """
        try:
            if not content or ('|' not in content and '**' not in content):
                logger.debug("No table content detected, skipping preprocessing")
                return content
                
            logger.info("Starting comprehensive table preprocessing", 
                       content_length=len(content), 
                       pipe_count=content.count('|'))
            self.stats = TableStats()  # Reset stats
            
            original_content = content  # Keep for error recovery

            try:
                # PHASE 0: Protect special content BEFORE any table processing
                # This is critical - protect footnotes and references FIRST
                logger.debug("Phase 0: Protecting footnote markers")
                content = self._protect_footnote_markers(content)

                # PHASE 0.5: Convert space-delimited tables to markdown format
                # DISABLED: We use structured generation (TableBlock) which produces valid markdown
                # No need to detect/convert space-delimited tables - causes false positives
                # logger.debug("Phase 0.5: Converting space-delimited tables to markdown")
                # content = self._detect_and_convert_space_delimited_tables(content)

                # Step 0: Protect reference sections from table processing
                logger.debug("Step 0: Protecting reference sections")
                content = self._protect_reference_sections(content)
                
                # Step 1: Separate headings from tables
                logger.debug("Step 1: Separating headings from tables")
                content = self.separate_headings_from_tables(content)
                
                # Step 2: Fix two-line cell patterns (Item    Description format)
                logger.debug("Step 2: Fixing two-line cell patterns")
                content = self.fix_two_line_cells(content)
                
                # Step 3: Fix merged table components
                logger.debug("Step 3: Fixing merged table components")
                content = self.fix_merged_table_components(content)
                
                # Step 4: Remove trailing separators first (before mixed content processing)
                logger.debug("Step 4: Removing trailing separators")
                content = self.remove_trailing_separators(content)
                
                # Step 5: Fix mixed content-separator patterns
                logger.debug("Step 5: Fixing mixed content-separator patterns")
                content = self.fix_mixed_content_separators(content)
                
                # Step 6: Normalize separators
                logger.debug("Step 6: Normalizing separators")
                content = self.normalize_separators(content)
                
                # Step 7: Consolidate orphaned separators with adjacent data
                logger.debug("Step 7: Consolidating orphaned separators")
                content = self.consolidate_orphaned_separators(content)
                
                # Step 8: Consolidate adjacent table fragments
                logger.debug("Step 8: Consolidating table fragments")
                content = self.consolidate_table_fragments(content)
                
                # Step 9: Extract tables and fix column mismatches
                logger.debug("Step 9: Fixing table column counts")
                content = self.fix_tables_column_counts(content)
                
                # Step 10: Fix collapsed comparative tables (specific fix for user's issue)
                logger.debug("Step 10: Fixing collapsed comparative tables")
                content = self.fix_collapsed_comparative_tables(content)
                
                # Step 11: Final cleanup
                logger.debug("Step 11: Final cleanup")
                content = self.final_cleanup(content)
                
                # Step 12: Add table boundaries for ALL tables to ensure proper markers
                logger.debug("Step 12: Adding table boundaries for all tables")
                content = self.add_table_boundaries(content)
                
                # Step 13: Restore protected reference sections
                logger.debug("Step 13: Restoring protected reference sections")
                content = self._restore_reference_sections(content)

                # FINAL STEP: Restore footnote markers (critical - do this LAST)
                logger.debug("Final step: Restoring footnote markers")
                content = self._restore_footnote_markers(content)

            except Exception as step_error:
                logger.error("Error during table preprocessing step", 
                           error=str(step_error), 
                           error_type=type(step_error).__name__,
                           exc_info=True)
                # Return original content if any step fails
                content = original_content
                logger.warning("Returning original content due to preprocessing error")
            
            # Safety check - make sure we haven't accidentally removed major content sections
            length_before = len(original_content)
            length_after = len(content)
            reduction_ratio = (length_before - length_after) / length_before if length_before > 0 else 0
            
            if reduction_ratio > 0.7:  # If we've removed more than 70% of content
                logger.error(f"Table preprocessing removed {reduction_ratio*100:.1f}% of content ({length_before} -> {length_after} chars) - this is likely an error")
                logger.error("Returning original content to prevent data loss")
                content = original_content
            elif reduction_ratio > 0.3:  # If we've removed more than 30% of content  
                logger.warning(f"Table preprocessing removed {reduction_ratio*100:.1f}% of content ({length_before} -> {length_after} chars) - please verify this is expected")
            
            logger.info("Table preprocessing complete", 
                       stats=self.get_stats(),
                       content_length_before=length_before,
                       content_length_after=len(content),
                       reduction_ratio=f"{reduction_ratio*100:.1f}%" if reduction_ratio > 0 else "0%")
            
            return content
            
        except Exception as e:
            logger.error("Critical error in table preprocessing", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        exc_info=True)
            # Return original content if everything fails
            return content if 'content' in locals() else ""

    def _detect_and_convert_space_delimited_tables(self, content: str) -> str:
        """
        Detect tables that are formatted with spaces instead of pipes and convert them.

        This handles the common LLM pattern where tables are generated like:
        Country (Region) Scenario Net take‑home Effective tax %
        **Spain** Single €212,000 23.0 %

        Instead of proper markdown:
        | Country | Scenario | Net take‑home | Effective tax % |
        | --- | --- | --- | --- |
        | Spain | Single | €212,000 | 23.0 % |
        """
        lines = content.split('\n')
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Detect potential table header (multiple column-like words, no pipes)
            if self._looks_like_space_delimited_header(line) and '|' not in line:
                logger.info(f"Detected space-delimited table header at line {i}: {line[:80]}")

                # Try to convert this table
                table_lines, next_i = self._extract_space_delimited_table(lines, i)

                if len(table_lines) >= 2:  # At least header + one data row
                    # Convert to markdown table
                    markdown_table = self._convert_space_table_to_markdown(table_lines)
                    if markdown_table:
                        result_lines.extend(markdown_table)
                        self.stats.patterns_fixed['space_table_converted'] = \
                            self.stats.patterns_fixed.get('space_table_converted', 0) + 1
                        i = next_i
                        continue

            result_lines.append(lines[i])
            i += 1

        return '\n'.join(result_lines)

    def _looks_like_space_delimited_header(self, line: str) -> bool:
        """
        Check if a line looks like a table header without pipes.

        Characteristics:
        - Multiple "words" that look like column headers
        - Contains capitals (Country, Scenario, etc.)
        - Has 3+ potential columns
        - Not a regular sentence
        """
        if not line or len(line) < 20:
            return False

        # Skip if it's clearly not a header
        if line.startswith('#') or line.startswith('-') or line.startswith('*'):
            return False

        # Skip numbered lists (references, citations)
        if re.match(r'^\d+\.\s+', line):
            return False

        # Skip if it contains URLs (likely a reference, not a table)
        if 'http://' in line or 'https://' in line:
            return False

        # Count capitalized words (potential column headers)
        words = line.split()
        capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2]

        # Need at least 3 column headers
        if len(capitalized_words) < 3:
            return False

        # Check if it has financial/data keywords that suggest table headers
        table_keywords = ['country', 'scenario', 'income', 'tax', 'rent', 'cost',
                         'benefit', 'rate', 'net', 'gross', 'total', 'annual',
                         'disposable', 'take-home', 'effective', 'daycare']

        line_lower = line.lower()
        has_table_keyword = any(keyword in line_lower for keyword in table_keywords)

        return has_table_keyword and len(capitalized_words) >= 3

    def _extract_space_delimited_table(self, lines: List[str], start_idx: int) -> tuple:
        """
        Extract all lines that belong to a space-delimited table.

        Returns: (table_lines, next_index)
        """
        table_lines = [lines[start_idx]]  # Start with header
        i = start_idx + 1

        while i < len(lines):
            line = lines[i].strip()

            # Empty line or section break - end of table
            if not line or line.startswith('##') or line.startswith('==='):
                break

            # If line has pipes, it's not part of this space-delimited table
            if '|' in line and line.count('|') >= 3:
                break

            # Check if this looks like a table row
            # (has country/data pattern or starts with bold country name)
            if (self._looks_like_table_data_row(line, table_lines[0]) or
                line.startswith('**')):
                table_lines.append(line)
                i += 1
            else:
                # Not a table row anymore
                break

        return table_lines, i

    def _looks_like_table_data_row(self, line: str, header: str) -> bool:
        """Check if a line looks like data corresponding to the header."""
        # Has currency symbols, numbers, percentages - typical table data
        has_data = any(symbol in line for symbol in ['€', '$', '£', '%', ',000'])

        # Or starts with a country/region name
        common_entities = ['spain', 'france', 'uk', 'united kingdom', 'switzerland',
                          'germany', 'poland', 'bulgaria', 'single', 'married']
        has_entity = any(entity in line.lower() for entity in common_entities)

        return has_data or has_entity

    def _convert_space_table_to_markdown(self, table_lines: List[str]) -> Optional[List[str]]:
        """
        Convert space-delimited table to markdown format.

        Strategy:
        1. Parse header to identify columns
        2. For each data row, align content to columns
        3. Generate proper markdown with pipes
        """
        if len(table_lines) < 2:
            return None

        try:
            # Parse header to get column boundaries
            header = table_lines[0]
            columns = self._parse_space_delimited_columns(header)

            if len(columns) < 2:
                return None

            logger.info(f"Converting space table with {len(columns)} columns and {len(table_lines)} rows")

            # Build markdown table
            markdown_lines = []

            # Header row
            header_cells = [col['text'] for col in columns]
            markdown_lines.append('| ' + ' | '.join(header_cells) + ' |')

            # Separator
            markdown_lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')

            # Data rows
            for line in table_lines[1:]:
                if not line.strip():
                    continue

                # Extract data based on column positions
                cells = self._extract_cells_from_line(line, columns)
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')

            return markdown_lines

        except Exception as e:
            logger.error(f"Error converting space table: {e}", exc_info=True)
            return None

    def _parse_space_delimited_columns(self, header: str) -> List[Dict]:
        """
        Parse space-delimited header to identify column boundaries.

        Returns list of dicts: [{'text': 'Country', 'start': 0, 'end': 10}, ...]
        """
        # Find capitalized words that are likely column headers
        columns = []
        words = []
        current_pos = 0

        # Split but track positions
        for match in re.finditer(r'\S+', header):
            word = match.group()
            start = match.start()
            end = match.end()
            words.append({'text': word, 'start': start, 'end': end})

        # Group consecutive capitalized words as column headers
        i = 0
        while i < len(words):
            word = words[i]

            if word['text'][0].isupper() or word['text'].startswith('('):
                # Start of a column header
                col_words = [word['text']]
                col_start = word['start']
                col_end = word['end']

                # Include following words that are part of this column
                j = i + 1
                while j < len(words):
                    next_word = words[j]
                    # Continue if lowercase or parentheses (e.g., "tax %" or "(Region)")
                    if (next_word['text'][0].islower() or
                        next_word['text'] in ['%', '(yr)'] or
                        next_word['text'].startswith('(')):
                        col_words.append(next_word['text'])
                        col_end = next_word['end']
                        j += 1
                    else:
                        break

                columns.append({
                    'text': ' '.join(col_words),
                    'start': col_start,
                    'end': col_end
                })

                i = j
            else:
                i += 1

        return columns

    def _extract_cells_from_line(self, line: str, columns: List[Dict]) -> List[str]:
        """
        Extract cell content from a data line based on column positions.
        """
        cells = []

        # For each column, extract the corresponding content
        for i, col in enumerate(columns):
            if i == len(columns) - 1:
                # Last column - take everything after previous column's end
                start_pos = col['start'] if i == 0 else columns[i-1]['end']
                cell_content = line[start_pos:].strip()
            else:
                # Take content up to next column's start
                start_pos = col['start'] if i == 0 else columns[i-1]['end']
                end_pos = columns[i+1]['start']
                cell_content = line[start_pos:end_pos].strip()

            # Clean up cell content
            cell_content = re.sub(r'\s{2,}', ' ', cell_content)
            cells.append(cell_content)

        # Ensure we have the right number of cells
        while len(cells) < len(columns):
            cells.append('')

        return cells[:len(columns)]

    def _protect_footnote_markers(self, content: str) -> str:
        """
        Protect footnote markers like *¹, *², *³, *⁴, *⁵, *⁶ from being
        treated as markdown emphasis or being stripped.

        Handles both Unicode superscripts (¹²³) and regular numbers (*1, *2, *3).
        """
        import uuid

        # Reset footnote placeholders for this content
        self.footnote_placeholders = {}

        # Pattern for Unicode superscript footnotes: *¹, *², *³, etc.
        unicode_footnote_pattern = r'\*([¹²³⁴⁵⁶⁷⁸⁹⁰]+)'

        # Pattern for regular number footnotes: *1, *2, *3, etc.
        numeric_footnote_pattern = r'\*(\d+)'

        # Also protect double asterisk footnotes: **1, **2
        double_asterisk_pattern = r'\*\*(\d+)'

        footnote_count = 0

        # Process Unicode superscript footnotes
        for match in re.finditer(unicode_footnote_pattern, content):
            footnote = match.group(0)  # Full match including *
            placeholder_id = f"FN_UNICODE_{uuid.uuid4().hex[:8]}"
            self.footnote_placeholders[placeholder_id] = footnote
            content = content.replace(footnote, placeholder_id, 1)  # Replace only first occurrence
            footnote_count += 1

        # Process numeric footnotes (but only if they look like footnotes, not emphasis)
        # Only process if:
        # 1. After a word/number (no space before *)
        # 2. Before punctuation, space, or pipe
        for match in re.finditer(r'(\w)\*(\d{1,2})([,\.\s\|]|$)', content):
            before_char = match.group(1)
            footnote_num = match.group(2)
            after_char = match.group(3)
            full_match = f"{before_char}*{footnote_num}{after_char}"

            placeholder_id = f"FN_NUM_{uuid.uuid4().hex[:8]}"
            self.footnote_placeholders[placeholder_id] = f"*{footnote_num}"

            # Replace with placeholder while preserving surrounding chars
            replacement = f"{before_char}{placeholder_id}{after_char}"
            content = content.replace(full_match, replacement, 1)
            footnote_count += 1

        if footnote_count > 0:
            logger.info(f"Protected {footnote_count} footnote markers from processing")
            self.stats.patterns_fixed['protected_footnotes'] = footnote_count

        return content

    def _restore_footnote_markers(self, content: str) -> str:
        """
        Restore protected footnote markers after all processing is complete.
        """
        for placeholder_id, original_footnote in self.footnote_placeholders.items():
            content = content.replace(placeholder_id, original_footnote)

        restored_count = len(self.footnote_placeholders)
        if restored_count > 0:
            logger.info(f"Restored {restored_count} footnote markers")

        return content

    def _protect_reference_sections(self, content: str) -> str:
        """
        Protect reference/citation sections from table processing.
        
        References sections often contain dashes, pipes, and URLs that can be
        mistaken for table elements, causing corruption.
        """
        import uuid
        
        # Reset protected sections for this content
        self.protected_sections = {}
        
        # Enhanced patterns to match reference sections
        # These catch both heading-based and plain text reference sections
        reference_patterns = [
            r'(## References.*?)(?=##|\Z)',
            r'(### References.*?)(?=###|\Z)',
            r'(# References.*?)(?=#|\Z)',
            r'(## Bibliography.*?)(?=##|\Z)',
            r'(## Citations.*?)(?=##|\Z)',
            r'(## Key Citations.*?)(?=##|\Z)',           # NEW - Key Citations heading
            r'(---\s*### References.*?)(?=---|\Z)',      # Handle metadata-style references
            r'(References:.*?)(?=##|\Z)',                 # NEW - Plain "References:" without ##
            r'(\n\d+\.\s+.*?https?://.*?)(?=\n(?:\d+\.|##)|\Z)',  # NEW - Numbered references with URLs
        ]
        
        for pattern in reference_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                reference_content = match.group(1)
                # Only protect if it actually contains problematic characters
                if '|' in reference_content or '---' in reference_content:
                    placeholder_id = f"PROTECTED_REFS_{uuid.uuid4().hex[:8]}"
                    self.protected_sections[placeholder_id] = reference_content
                    content = content.replace(reference_content, placeholder_id)
                    logger.info(f"Protected reference section with {len(reference_content)} characters")
        
        return content
    
    def _restore_reference_sections(self, content: str) -> str:
        """
        Restore protected reference sections after table processing.
        """
        for placeholder_id, original_content in self.protected_sections.items():
            content = content.replace(placeholder_id, original_content)
            
        logger.info(f"Restored {len(self.protected_sections)} protected reference sections")
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
                            logger.debug("  -> Has valid header")
                        else:
                            logger.debug("  -> Not a valid header")
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
        
        for line in table_lines:
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                has_separator = True
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
        
        for line in table_lines:
            if '|' in line:
                # Check if it's a separator line
                if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
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
    
    def fix_collapsed_comparative_tables(self, content: str) -> str:
        """
        Fix collapsed comparative tables where multiple rows are merged into a single line.
        
        This handles cases like:
        - | | Location | Scenario 1 | Scenario 2 | Scenario 3 | | | --- | | | San Francisco | data1 | data2 | data3 | | | | New York | data1 | data2 | data3 |
        - Performance characteristics...: | Technology | Typical Capacity Factor | Efficiency | | Solar PV | 21%-34% | 20%-22% |
        
        Which should be:
        | Location | Scenario 1 | Scenario 2 | Scenario 3 |
        | --- | --- | --- | --- |
        | San Francisco | data1 | data2 | data3 |
        | New York | data1 | data2 | data3 |
        """
        try:
            lines = content.split('\n')
            fixed_lines = []
            processed_count = 0
            
            logger.debug(f"Processing {len(lines)} lines for collapsed tables")
            
            for line_idx, line in enumerate(lines):
                try:
                    if '|' in line and len(line) > 120:  # Only process long table-like lines
                        logger.debug(f"Processing potential collapsed table at line {line_idx}, length: {len(line)}")
                        
                        # Extract potential table content from mixed content lines
                        table_content = self._extract_table_from_mixed_content(line)
                        if table_content and table_content != line:
                            logger.debug(f"Extracted table content: {table_content[:100]}...")
                            
                            # Process the extracted table content
                            generic_table = self._reconstruct_generic_collapsed_table(table_content)
                            if generic_table:
                                # Add any descriptive text before the table
                                descriptive_text = line[:len(line) - len(table_content)].strip()
                                if descriptive_text and not descriptive_text.endswith(':'):
                                    fixed_lines.append(descriptive_text)
                                elif descriptive_text.endswith(':'):
                                    fixed_lines.append(descriptive_text[:-1].strip())
                                
                                fixed_lines.extend(generic_table)
                                self.stats.patterns_fixed['collapsed_mixed_content'] = \
                                    self.stats.patterns_fixed.get('collapsed_mixed_content', 0) + 1
                                self.stats.tables_fixed += 1
                                processed_count += 1
                                
                                column_estimate = len([
                                    segment for segment in generic_table[0].split('|')
                                    if segment.strip()
                                ]) if generic_table else 0
                                logger.info(
                                    "Detected collapsed table with mixed content: %d rows and %d columns",
                                    max(len(generic_table) - 2, 0),
                                    column_estimate
                                )
                                continue
                        
                        # First try a generic reconstruction that works for any entity list
                        generic_table = self._reconstruct_generic_collapsed_table(line)
                        if generic_table:
                            fixed_lines.extend(generic_table)
                            self.stats.patterns_fixed['collapsed_generic'] = \
                                self.stats.patterns_fixed.get('collapsed_generic', 0) + 1
                            self.stats.tables_fixed += 1
                            processed_count += 1
                            
                            column_estimate = len([
                                segment for segment in generic_table[0].split('|')
                                if segment.strip()
                            ]) if generic_table else 0
                            logger.info(
                                "Detected collapsed table with %d rows and %d columns",
                                max(len(generic_table) - 2, 0),
                                column_estimate
                            )
                            continue

                    fixed_lines.append(line)
                    
                except Exception as line_error:
                    logger.warning(f"Error processing line {line_idx} in collapsed table detection", 
                                 error=str(line_error), 
                                 line_preview=line[:100] if line else "empty",
                                 exc_info=True)
                    # Continue with the original line if processing fails
                    fixed_lines.append(line)

            logger.debug(f"Collapsed table processing complete: {processed_count} tables processed")
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            logger.error("Critical error in fix_collapsed_comparative_tables", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        exc_info=True)
            return content  # Return original content on critical error


    def _extract_table_from_mixed_content(self, line: str) -> Optional[str]:
        """
        Extract table content from lines that contain both descriptive text and table data.
        
        Handles patterns like:
        - "Performance characteristics...: | Technology | data |"
        - "The following table shows: | Header1 | Header2 | | Row1 | Row2 |"
        
        Returns:
            The extracted table content or None if no clear separation found
        """
        try:
            logger.debug(f"Extracting table from mixed content: {line[:150]}...")
            
            # Look for common separators between descriptive text and table content
            separators = [': |', ':| ', ': ',':|']
            
            for separator in separators:
                if separator in line:
                    logger.debug(f"Found separator: '{separator}'")
                    parts = line.split(separator, 1)  # Split only on first occurrence
                    if len(parts) == 2:
                        text_part, table_part = parts
                        logger.debug(f"Text part: {text_part[:50]}...")
                        logger.debug(f"Table part: {table_part[:100]}...")
                        
                        # Ensure the table part looks like table data (has pipes and reasonable length)
                        if table_part.count('|') >= 4 and len(table_part) > 50:
                            # Add the leading pipe if missing
                            if not table_part.startswith('|'):
                                table_part = '|' + table_part
                            # Add trailing pipe if missing
                            if not table_part.strip().endswith('|'):
                                table_part = table_part.strip() + ' |'
                            
                            logger.debug(f"Extracted table content successfully: {len(table_part)} chars")
                            return table_part
            
            # Try pattern where table starts after a colon and space
            colon_pattern = re.search(r'^([^|]*?):\s*(\|.+)$', line)
            if colon_pattern:
                logger.debug("Found colon pattern match")
                table_part = colon_pattern.group(2)
                if table_part.count('|') >= 4 and len(table_part) > 50:
                    logger.debug(f"Colon pattern extraction successful: {len(table_part)} chars")
                    return table_part
            
            # Try pattern where table content appears after common phrases
            table_intro_patterns = [
                r'^.*(following table|table shows|characteristics|metrics|performance data).*?(\|.+)$',
                r'^.*(as follows|shown below|presented below).*?(\|.+)$'
            ]
            
            for pattern in table_intro_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    logger.debug(f"Found intro pattern match: {match.group(1)}")
                    table_part = match.group(2)
                    if table_part.count('|') >= 4 and len(table_part) > 50:
                        logger.debug(f"Intro pattern extraction successful: {len(table_part)} chars")
                        return table_part
            
            logger.debug("No table content extracted from mixed content line")
            return None
            
        except Exception as e:
            logger.error("Error in _extract_table_from_mixed_content", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        line_preview=line[:100] if line else "empty",
                        exc_info=True)
            return None

    def _reconstruct_generic_collapsed_table(self, line: str) -> Optional[List[str]]:
        """
        Attempt to rebuild a collapsed table without relying on specific entity names.
        
        Improved version that handles:
        - Empty cells (multiple consecutive pipes)
        - Complex cell content with spaces and special characters
        - Variable column counts across rows
        - Better row boundary detection
        """
        try:
            logger.debug(f"Attempting to reconstruct collapsed table from line: {line[:200]}...")
            
            if line.count('|') < 8:  # Reduced threshold for better detection
                logger.debug(f"Line has too few pipes ({line.count('|')}), skipping")
                return None

            # Clean the line and normalize pipe patterns
            clean_line = re.sub(r'\s*\|\s*', '|', line.strip())
            logger.debug(f"Cleaned line: {clean_line[:100]}...")
            
            # Split on pipes, keeping empty strings to preserve structure
            parts = clean_line.split('|')
            logger.debug(f"Split into {len(parts)} parts")
            
            # Remove leading/trailing empty parts
            while parts and not parts[0].strip():
                parts.pop(0)
            while parts and not parts[-1].strip():
                parts.pop()
                
            if len(parts) < 8:  # Need at least some content
                logger.debug(f"Too few content parts after cleanup ({len(parts)}), skipping")
                return None

            rows: List[List[str]] = []
            current_row: List[str] = []
            
            i = 0
            while i < len(parts):
                part = parts[i].strip()
                
                # Detect potential row boundaries by looking for patterns that indicate new rows
                # This includes technology names, location names, or typical table row starters
                is_row_boundary = (
                    current_row and  # We already have some content
                    part and  # Current part is not empty
                    (
                        # Check for common table row starters (entities/technologies)
                        any(keyword in part.lower() for keyword in [
                            'solar', 'wind', 'hydro', 'battery', 'nuclear', 'coal', 'gas',
                            'onshore', 'offshore', 'bioenergy', 'hydrogen', 'geothermal',
                            'pump', 'storage', 'pv', 'photovoltaic', 'heat pump'
                        ]) or
                        # Check for location/entity patterns (capitalized words)
                        (re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*$', part) and len(part) > 2 and 
                         not any(header_word in part.lower() for header_word in [
                             'technology', 'capacity', 'efficiency', 'utilisation', 'constraints',
                             'factor', 'electrical', 'resource', 'scalability', 'primary'
                         ])) or
                        # Check if this looks like a new row start (multiple words, first capitalized)
                        # but avoid common header words
                        (len(part.split()) >= 2 and part[0].isupper() and 
                         not any(header_word in part.lower() for header_word in [
                             'technology', 'capacity', 'efficiency', 'utilisation', 'constraints',
                             'factor', 'electrical', 'resource', 'scalability', 'primary'
                         ]))
                    )
                )
                
                if is_row_boundary:
                    # Complete current row and start new one
                    if current_row:
                        rows.append(current_row)
                        logger.debug(f"Completed row: {current_row[:3]}...")
                    current_row = [part]
                else:
                    # Add to current row
                    current_row.append(part)
                
                i += 1
            
            # Add the last row
            if current_row:
                rows.append(current_row)
                logger.debug(f"Final row: {current_row[:3]}...")
            
            logger.debug(f"Found {len(rows)} total rows")
            
            # Filter out rows that are too short or seem invalid
            valid_rows = []
            for row_idx, row in enumerate(rows):
                # Remove empty cells at the end
                while row and not row[-1].strip():
                    row.pop()
                # Keep rows with at least 2 non-empty cells
                non_empty_cells = [cell for cell in row if cell.strip()]
                if len(non_empty_cells) >= 2:
                    valid_rows.append(row)
                    logger.debug(f"Valid row {row_idx}: {len(row)} cells, {len(non_empty_cells)} non-empty")
                else:
                    logger.debug(f"Skipping invalid row {row_idx}: {len(non_empty_cells)} non-empty cells")
            
            if len(valid_rows) < 2:
                logger.debug(f"Too few valid rows ({len(valid_rows)}), table reconstruction failed")
                return None

            # Determine target column count from the longest row
            target_columns = max(len(row) for row in valid_rows)
            if target_columns < 2:
                logger.debug(f"Target column count too low ({target_columns})")
                return None
                
            logger.debug(f"Target columns: {target_columns}")
                
            # Assume first row is header
            header = valid_rows[0]
            data_rows = valid_rows[1:]

            def normalize_row(row: List[str]) -> List[str]:
                # Pad short rows
                if len(row) < target_columns:
                    return row + [''] * (target_columns - len(row))
                # Merge overflow into last column
                if len(row) > target_columns:
                    overflow = ' '.join(row[target_columns - 1:])
                    return row[:target_columns - 1] + [overflow]
                return row

            # Normalize all rows
            normalized_header = normalize_row(header)
            normalized_data_rows = [normalize_row(row) for row in data_rows]
            
            # Build the markdown table
            table_lines = []
            
            # Header row
            table_lines.append('| ' + ' | '.join(cell.strip() for cell in normalized_header) + ' |')
            
            # Separator row
            table_lines.append('|' + ' --- |' * target_columns)
            
            # Data rows
            for row in normalized_data_rows:
                table_lines.append('| ' + ' | '.join(cell.strip() for cell in row) + ' |')
            
            # Final validation - make sure we have a proper table
            if len(table_lines) < 3:  # Header + separator + at least one data row
                logger.debug(f"Final table too short ({len(table_lines)} lines)")
                return None
                
            logger.debug(f"Successfully reconstructed table with {len(table_lines)} lines")
            return table_lines
            
        except Exception as e:
            logger.error("Error in _reconstruct_generic_collapsed_table", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        line_preview=line[:100] if line else "empty",
                        exc_info=True)
            return None
    
    def final_cleanup(self, content: str) -> str:
        """
        Final cleanup pass for any remaining issues.
        Be careful not to remove legitimate report formatting.
        """
        # Remove any remaining double pipes
        content = re.sub(r'\|\|+', '|', content)
        
        # Remove standalone pipe lines and ONLY very specific orphaned separators
        lines = content.split('\n')
        fixed_lines = []
        
        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty pipe rows and standalone pipes (table-related only)
            if stripped in ['|', '||'] or re.match(r'^\|\s*(\|\s*)*$', stripped):
                self.stats.patterns_fixed['empty_pipes'] = self.stats.patterns_fixed.get('empty_pipes', 0) + 1
                logger.debug(f"Removing empty pipe line: '{stripped}'")
                continue
                
            # ONLY skip very specific orphaned table separators, NOT report formatting
            # Only remove standalone "---" (exactly 3 dashes) that are clearly orphaned table separators
            # Do NOT remove longer separator lines like ======= which are report formatting
            if stripped == '---':  # Only exactly 3 dashes, not more
                # Check context - is this actually an orphaned table separator?
                # Don't remove it if it's part of a larger formatting pattern
                is_orphaned_table_sep = True
                
                # Get surrounding context
                context_lines = []
                for offset in [-2, -1, 1, 2]:
                    idx = line_idx + offset
                    if 0 <= idx < len(lines):
                        context_lines.append(lines[idx].strip())
                
                # If surrounded by non-table content, this might be legitimate formatting
                table_like_context = any('|' in ctx and len(ctx) > 5 for ctx in context_lines)
                
                if not table_like_context:
                    is_orphaned_table_sep = False
                    logger.debug("Preserving '---' line as it appears to be report formatting, not orphaned table separator")
                
                if is_orphaned_table_sep:
                    self.stats.patterns_fixed['orphaned_separators'] = self.stats.patterns_fixed.get('orphaned_separators', 0) + 1
                    logger.debug("Removing orphaned table separator: '---'")
                    continue
                    
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Remove excessive blank lines (but preserve intentional spacing)
        content = re.sub(r'\n{4,}', '\n\n\n', content)  # Allow up to 3 line breaks
        
        # Safety check - if we've removed too much content, something went wrong
        original_lines = len(lines)
        final_lines = len(content.split('\n'))
        removed_lines = original_lines - final_lines
        
        if removed_lines > 0:
            logger.debug(f"Final cleanup removed {removed_lines} lines out of {original_lines}")
            
            # If we've removed more than 50% of the content, that's suspicious
            if removed_lines > original_lines * 0.5:
                logger.warning(f"Final cleanup removed {removed_lines}/{original_lines} lines ({removed_lines/original_lines*100:.1f}%) - this seems excessive")
        
        return content
    
    def add_table_boundaries(self, content: str) -> str:
        """
        Add TABLE_START/TABLE_END markers for ALL tables.
        This helps the UI know when a complete table has been received.
        """
        lines = content.split('\n')
        marked_lines = []
        in_table = False
        table_lines = []
        has_separator = False
        has_content = False

        def flush_pipe_table():
            nonlocal in_table, table_lines, has_separator, has_content
            if not in_table:
                return
            # Mark ALL tables, even simple ones, to ensure consistent UI handling
            is_valid_table = (has_separator and has_content) or len(table_lines) >= 2
            if is_valid_table:
                marked_lines.append(self.table_start_marker)
                marked_lines.extend(table_lines)
                marked_lines.append(self.table_end_marker)
            else:
                marked_lines.extend(table_lines)
            in_table = False
            table_lines = []
            has_separator = False
            has_content = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # If we were in a pipe table and hit a non-table line, flush it first
            if in_table and '|' not in stripped:
                flush_pipe_table()
                # Re-evaluate current line after flushing
                continue

            # Detect bold-header tables lacking pipe delimiters
            bold_cells = re.findall(r'\*\*(.+?)\*\*', line)
            if (
                len(bold_cells) >= 3
                and not stripped.startswith(('- ', '* ', '> '))
                and not stripped.startswith('#')
            ):
                data_lines = []
                j = i + 1
                while j < len(lines):
                    candidate = lines[j]
                    candidate_stripped = candidate.strip()
                    if not candidate_stripped:
                        break
                    if candidate_stripped.startswith(('#', '##', '###', '####', '---')):
                        break
                    if candidate_stripped.startswith(('- ', '* ', '> ')):
                        break
                    if candidate_stripped.startswith(self.table_start_marker) or candidate_stripped.startswith(self.table_end_marker):
                        break
                    if re.findall(r'\*\*(.+?)\*\*', candidate_stripped):
                        break
                    data_lines.append(candidate)
                    j += 1

                if len(data_lines) >= 2:
                    marked_lines.append(self.table_start_marker)
                    marked_lines.append(self.table_header_start_marker)
                    marked_lines.append(line)
                    marked_lines.append(self.table_header_end_marker)
                    marked_lines.extend(data_lines)
                    marked_lines.append(self.table_end_marker)
                    i = j
                    continue

            if '|' in stripped:
                # Skip orphaned separators (lone separator lines not part of a table)
                if re.match(r'^\|[\s\-:|]+\|$', stripped) and not in_table:
                    next_is_table = False
                    if i + 1 < len(lines) and '|' in lines[i + 1]:
                        next_line = lines[i + 1].strip()
                        if not re.match(r'^\|[\s\-:|]+\|$', next_line):
                            next_is_table = True
                    if not next_is_table:
                        marked_lines.append(line)
                        i += 1
                        continue

                if not in_table:
                    in_table = True
                    table_lines = [line]
                    if re.match(r'^\|[\s\-:|]+\|$', stripped):
                        has_separator = True
                    else:
                        has_content = True
                else:
                    table_lines.append(line)
                    if re.match(r'^\|[\s\-:|]+\|$', stripped):
                        has_separator = True
                    else:
                        has_content = True
                i += 1
                continue

            # Regular line (not part of any table pattern)
            marked_lines.append(line)
            i += 1

        # Flush trailing pipe table if present
        flush_pipe_table()

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
