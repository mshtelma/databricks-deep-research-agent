"""
Tests for the comprehensive table preprocessor.
"""

import pytest
from deep_research_agent.core.table_preprocessor import TablePreprocessor


class TestTablePreprocessor:
    """Test suite for the table preprocessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TablePreprocessor()
    
    def test_two_line_cell_pattern(self):
        """Test fixing the Item/Description two-line pattern."""
        input_text = """Here is the scope:

Item    Description
| Reference year
2024 tax year (latest published rules)
| Household composition
Married / civil-partnered couple, both earning a full-time salary
| Gross income assumption
Each spouse ≈ €40 000 gross per year

More content here."""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should have proper table structure
        assert "| Item | Description |" in result
        assert "| --- | --- |" in result
        assert "| Reference year | 2024 tax year (latest published rules) |" in result
        assert "| Household composition | Married / civil-partnered couple, both earning a full-time salary |" in result
        
        # Should have fixed the pattern
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('two_line_cells', 0) > 0
    
    def test_merged_components(self):
        """Test splitting merged headers-separators-data."""
        input_text = """Analysis results:

| Header1 | Header2 || --- | --- | --- | Data1 | Data2 |

End of analysis."""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should be split into proper lines
        assert "| Header1 | Header2 |" in result
        assert "| --- | --- |" in result
        assert "| Data1 | Data2 |" in result
        
        # Should not have merged content
        assert "|| --- | --- |" not in result
        
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('merged_components', 0) > 0
    
    def test_excessive_separators(self):
        """Test removing duplicate separator rows."""
        input_text = """| Country | Data |
| --- | --- |
|---|---|

|

| --- | --- |
|---|---|

| Spain | Value |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should have only one separator
        assert result.count("| --- | --- |") == 1
        
        # Should not have condensed separators
        assert "|---|---|" not in result
        
        # Should preserve data
        assert "| Spain | Value |" in result
        
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('duplicate_separators', 0) > 0
    
    def test_mixed_content_separators(self):
        """Test fixing content mixed with separator patterns."""
        input_text = """| Country | Value | --- | --- | Spain | €1000 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should extract only content
        assert "| Country | Value | Spain | €1000 |" in result
        
        # Should not have separators mixed with content
        assert "| --- |" not in result or result.count("| --- |") <= 4  # Allow proper separator row
        
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('mixed_content', 0) > 0
    
    def test_column_mismatch(self):
        """Test fixing column count mismatches."""
        input_text = """| Header1 | Header2 | Header3 |
|---|---|---|---|---|
| Data1 | Data2 |
| A | B | C | D | E |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # All rows should have same column count
        lines = result.strip().split('\n')
        table_lines = [l for l in lines if '|' in l]
        
        if table_lines:
            # Count columns in each row
            column_counts = []
            for line in table_lines:
                if '---' not in line:  # Skip separator rows for this check
                    parts = line.split('|')
                    # Count non-empty parts between pipes
                    count = len([p for p in parts[1:-1]])
                    column_counts.append(count)
            
            # All should have the same count
            if column_counts:
                assert len(set(column_counts)) == 1, f"Column counts vary: {column_counts}"
    
    def test_trailing_separators(self):
        """Test removing trailing separators from data rows."""
        input_text = """| Country | Population | GDP | --- |
| Spain | 47M | €1.4T | --- |
| France | 67M | €2.9T | --- |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should remove trailing separators
        assert "| Spain | 47M | €1.4T |" in result
        assert "| France | 67M | €2.9T |" in result
        
        # Should not have content ending with | --- |
        lines = result.split('\n')
        for line in lines:
            if any(c.isalpha() for c in line):  # Has content
                assert not line.strip().endswith("| --- |"), f"Line still has trailing separator: {line}"
        
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('trailing_separators', 0) > 0
    
    def test_condensed_separators(self):
        """Test fixing condensed separator patterns."""
        input_text = """| Header1 | Header2 | Header3 |
|---|---|---|
| Data1 | Data2 | Data3 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should convert to proper format
        assert "| --- | --- | --- |" in result
        assert "|---|---|---|" not in result
        
        stats = self.preprocessor.get_stats()
        assert stats['patterns_fixed'].get('condensed_separators', 0) > 0
    
    def test_empty_pipes_removal(self):
        """Test removing empty pipe rows."""
        input_text = """| Header | Value |
|
| --- | --- |
||
| Data | 100 |
|"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should remove standalone pipes
        lines = result.strip().split('\n')
        assert "|" not in [l.strip() for l in lines]
        assert "||" not in [l.strip() for l in lines]
        
        # Should keep valid content
        assert "| Header | Value |" in result
        assert "| Data | 100 |" in result
    
    def test_table_boundaries(self):
        """Test adding TABLE_START/TABLE_END markers."""
        input_text = """Some text before.

| Country | GDP | Population | Growth |
| --- | --- | --- | --- |
| USA | $25T | 335M | 2.1% |
| China | $17T | 1.4B | 5.2% |
| Japan | $4.2T | 125M | 1.0% |
| Germany | $4.0T | 84M | 0.1% |

Some text after."""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should add markers for complex tables
        assert "TABLE_START" in result
        assert "TABLE_END" in result
        
        # Table should be between markers
        start_idx = result.index("TABLE_START")
        end_idx = result.index("TABLE_END")
        assert start_idx < end_idx
        
        table_section = result[start_idx:end_idx]
        assert "| USA |" in table_section
        assert "| China |" in table_section
    
    def test_complex_malformed_table(self):
        """Test fixing a complex table with multiple issues."""
        input_text = """**Analyst Rating Snapshot**| Company | Rating | Recent Moves | Source |
| --- | --- | --- | --- | Microsoft | Positive | Upgrades noted | Yahoo | Apple | Neutral | No changes | Yahoo |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should separate heading from table
        assert "**Analyst Rating Snapshot**" in result
        assert "| Company | Rating | Recent Moves | Source |" in result
        
        # Should have proper structure
        assert "| --- | --- | --- | --- |" in result
        assert "| Microsoft | Positive | Upgrades noted | Yahoo |" in result
        assert "| Apple | Neutral | No changes | Yahoo |" in result
    
    def test_no_table_content(self):
        """Test that content without tables passes through unchanged."""
        input_text = """This is regular text without any tables.
        
It has multiple paragraphs but no pipe characters.

Just normal markdown content."""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should be unchanged
        assert result == input_text
        
        stats = self.preprocessor.get_stats()
        assert stats['total_fixes'] == 0
    
    def test_valid_table_preservation(self):
        """Test that valid tables are preserved."""
        input_text = """| Header1 | Header2 | Header3 |
| --- | --- | --- |
| Data1 | Data2 | Data3 |
| Data4 | Data5 | Data6 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should preserve valid table structure
        assert "| Header1 | Header2 | Header3 |" in result
        assert "| --- | --- | --- |" in result
        assert "| Data1 | Data2 | Data3 |" in result
        assert "| Data4 | Data5 | Data6 |" in result
    
    def test_double_pipe_removal(self):
        """Test that all double pipes are removed."""
        input_text = """| Header1 || Header2 |
|---|---|
| Data1 || Data2 |
| More || Content ||"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should have no double pipes
        assert "||" not in result
        
        # Should preserve content
        assert "Header1" in result
        assert "Header2" in result
        assert "Data1" in result
        assert "Data2" in result
    
    def test_stats_tracking(self):
        """Test that statistics are properly tracked."""
        # Reset stats
        self.preprocessor = TablePreprocessor()
        
        input_text = """| Header | Value | --- |
|---|---|
| Data | 100 |
||
|"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        stats = self.preprocessor.get_stats()
        
        # Should track various fixes
        assert stats['total_fixes'] > 0
        assert 'condensed_separators' in stats['patterns_fixed']
        assert 'trailing_separators' in stats['patterns_fixed']
        assert 'empty_pipes' in stats['patterns_fixed']
    
    def test_orphaned_separator_removal(self):
        """Test detection and removal of orphaned separators."""
        # Test orphaned separator with data below
        input_text = """Some text before

| --- | --- |

| Data1 | Data2 |
| Data3 | Data4 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Orphaned separator should be removed
        separator_count = result.count("| --- | --- |")
        assert separator_count <= 1, f"Expected at most 1 separator, got {separator_count}"
        
        # Data should be preserved
        assert "| Data1 | Data2 |" in result
        assert "| Data3 | Data4 |" in result
        
        # Test orphaned separator without data
        input_text2 = """| Header | Value |
| --- | --- |
| Data | 100 |

| --- | --- |

Some text after"""
        
        result2 = self.preprocessor.preprocess_tables(input_text2)
        
        # Second orphaned separator should be removed
        assert result2.count("| --- | --- |") == 1
        
        # Test separator with valid header (should keep)
        input_text3 = """| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |"""
        
        result3 = self.preprocessor.preprocess_tables(input_text3)
        
        # Valid separator should be kept
        assert "| --- | --- |" in result3
        assert result3.count("| --- | --- |") == 1
    
    def test_table_fragment_consolidation(self):
        """Test consolidation of table fragments separated by blank lines."""
        # Test fragments with 1 blank line gap (should merge)
        input_text = """| Header | Data |
| --- | --- |

| Row1 | Value1 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should have consolidated into one table with one separator
        assert result.count("| --- | --- |") == 1
        assert "| Header | Data |" in result
        assert "| Row1 | Value1 |" in result
        
        # Test fragments with 2 blank lines gap (should merge)
        input_text2 = """| Header | Data |
| --- | --- |


| Row1 | Value1 |
| --- | --- |

| Row2 | Value2 |"""
        
        result2 = self.preprocessor.preprocess_tables(input_text2)
        
        # Should consolidate and have only one separator
        assert result2.count("| --- | --- |") == 1
        
        # Test fragments with 3+ blank lines gap (should NOT merge)
        input_text3 = """| Table1 | Data1 |
| --- | --- |
| Row1 | Value1 |



| Table2 | Data2 |
| --- | --- |
| Row2 | Value2 |"""
        
        result3 = self.preprocessor.preprocess_tables(input_text3)
        
        # Should keep as separate tables
        assert "TABLE_START" in result3
        assert "TABLE_END" in result3
        # Two separate tables means two separators
        assert result3.count("| --- | --- |") == 2
    
    def test_header_validation_logic(self):
        """Test improved header validation that excludes invalid headers."""
        # Test single pipe is not a valid header
        input_text = """|

| --- | --- |

| Data1 | Data2 |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # The separator after single pipe should be treated as orphaned
        # and the data should be preserved
        assert "| Data1 | Data2 |" in result
        
        # Test empty pipes are not valid headers
        input_text2 = """||

| --- | --- |

| Data1 | Data2 |"""
        
        result2 = self.preprocessor.preprocess_tables(input_text2)
        
        # The separator should be treated as orphaned
        assert "| Data1 | Data2 |" in result2
        
        # Test valid headers with at least 2 columns
        input_text3 = """| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |"""
        
        result3 = self.preprocessor.preprocess_tables(input_text3)
        
        # Valid header should be preserved with separator
        assert "| Header1 | Header2 |" in result3
        assert "| --- | --- |" in result3
        assert "| Data1 | Data2 |" in result3
    
    def test_process_table_fragment_helper(self):
        """Test the _process_table_fragment helper method."""
        # Test removing duplicate separators within a fragment
        fragment_lines = [
            '| Header | Data |',
            '| --- | --- |',
            '',
            '| Row1 | Value1 |',
            '| --- | --- |',
            '| Row2 | Value2 |'
        ]
        
        processed = self.preprocessor._process_table_fragment(fragment_lines)
        
        # Should have only one separator
        separator_count = sum(1 for line in processed if '---' in line and '|' in line)
        assert separator_count == 1
        
        # All data should be preserved
        assert any('Header' in line for line in processed)
        assert any('Row1' in line for line in processed)
        assert any('Row2' in line for line in processed)
        
        # Test preserving single separator
        fragment_lines2 = [
            '| Header | Data |',
            '| --- | --- |',
            '| Row1 | Value1 |'
        ]
        
        processed2 = self.preprocessor._process_table_fragment(fragment_lines2)
        
        # Should preserve the single separator
        assert len([line for line in processed2 if '---' in line]) == 1
        assert len(processed2) == 3
    
    def test_edge_cases(self):
        """Test edge cases in table processing."""
        # Test table with only header and separator, no data
        input_text = """| Header1 | Header2 |
| --- | --- |"""
        
        result = self.preprocessor.preprocess_tables(input_text)
        
        # Should preserve the header and separator
        assert "| Header1 | Header2 |" in result
        assert "| --- | --- |" in result
        
        # Test table with only data, no header
        input_text2 = """| Data1 | Data2 |
| Data3 | Data4 |"""
        
        result2 = self.preprocessor.preprocess_tables(input_text2)
        
        # Should add a separator after first line (treated as header)
        assert "| Data1 | Data2 |" in result2
        assert "| --- | --- |" in result2
        assert "| Data3 | Data4 |" in result2
        
        # Test very large gap between fragments (should not merge)
        input_text3 = """| Table1 | Data1 |
| --- | --- |
| Row1 | Value1 |





| Table2 | Data2 |
| --- | --- |
| Row2 | Value2 |"""
        
        result3 = self.preprocessor.preprocess_tables(input_text3)
        
        # Should keep as separate tables
        assert result3.count("| --- | --- |") == 2
        assert "| Table1 | Data1 |" in result3
        assert "| Table2 | Data2 |" in result3