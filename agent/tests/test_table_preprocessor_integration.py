"""
Integration tests for table preprocessor with real-world examples.
"""

import pytest
from deep_research_agent.core.table_preprocessor import TablePreprocessor


class TestTablePreprocessorIntegration:
    """Integration tests with real agent output examples."""
    
    def test_real_world_tax_comparison_table(self):
        """Test the exact problematic table format from the user's example."""
        input_text = """Scope & Methodology

Item    Description
| Reference year
2024 tax year (latest published rules)
| Household composition
Married / civil‑partnered couple, both earning a full‑time salary.
| Gross income assumption
Each spouse ≈ €40 000 gross per year → €80 000 household gross (≈ CHF 84 000 for Switzerland, £ 71 000 for the UK, PLN 380 000 for Poland, BGN 155 000 for Bulgaria).
| Tax filing
Joint filing where the country allows it; otherwise the two individual returns are summed.
| Social‑security
Employee‑share only (employer contributions are not part of the take‑home pay).
| Child‑related benefits
Only the cash benefit that is paid directly to families (child allowance / child benefit). Tax‑free child‑tax‑credits that are already reflected in the net‑pay calculations are not added separately.
| Currency
All figures are shown in annual euros (€) for easy comparison (Swiss francs, pounds, zloty and lev are converted at 2024 average rates: 1 CHF ≈ 0.95 €, 1 £ ≈ 1.17 €, 1 PLN ≈ 0.20 €, 1 BGN ≈ 0.51 €).
| Sources
National tax authority tables, OECD "Tax‑Benefit Models", and government‑published benefit schedules (see citations in the tables)."""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Should have converted to proper table
        assert "| Item | Description |" in result
        assert "| --- | --- |" in result
        
        # Check key rows are properly formatted
        assert "| Reference year | 2024 tax year (latest published rules) |" in result
        assert "| Household composition | Married / civil‑partnered couple, both earning a full‑time salary. |" in result
        assert "| Tax filing | Joint filing where the country allows it; otherwise the two individual returns are summed. |" in result
        
        # Should have TABLE markers
        assert "TABLE_START" in result
        assert "TABLE_END" in result
        
        # No malformed patterns should remain
        assert "| Reference year\n2024" not in result  # No split cells
        
        stats = preprocessor.get_stats()
        assert stats['patterns_fixed'].get('two_line_cells', 0) > 0
    
    def test_microsoft_sentiment_table(self):
        """Test the Microsoft sentiment overview table pattern."""
        input_text = """**Microsoft (MSFT) – Sentiment Overview**| Sentiment Driver | Indicator | Tone | Key Points |
| --- | --- | --- | --- | Quarterly earnings news | CNBC – "stock dipped as investors focused on disappointing Azure revenue" | Mixed/Negative | The share price fell after Azure growth missed some expectations, creating short‑term downside pressure. |"""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Should have separated heading from table
        assert "**Microsoft (MSFT) – Sentiment Overview**" in result
        assert result.index("**Microsoft (MSFT)") < result.index("| Sentiment Driver")
        
        # Table should be properly structured
        assert "| Sentiment Driver | Indicator | Tone | Key Points |" in result
        assert "| --- | --- | --- | --- |" in result
        
        # Data should not be mixed with separators
        assert "| --- | --- | --- | --- | Quarterly earnings" not in result
        
    def test_country_comparison_with_excessive_separators(self):
        """Test country comparison table with excessive separators."""
        input_text = """| Country | Gross Income | Tax | Net |
| --- | --- | --- | --- |
|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|

| Spain | €90,000 | €27,000 | €63,000 |
|---|---|---|---|

| France | €90,000 | €30,000 | €60,000 |"""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Should have only ONE separator row
        assert result.count("| --- | --- | --- | --- |") == 1
        
        # Should not have condensed separators
        assert "|---|---|---|---|" not in result
        
        # Data should be preserved
        assert "| Spain | €90,000 | €27,000 | €63,000 |" in result
        assert "| France | €90,000 | €30,000 | €60,000 |" in result
    
    def test_mixed_content_separator_pattern(self):
        """Test real-world mixed content and separator patterns."""
        input_text = """| Country | Income Tax | Social Security | --- | --- | Total | Net Income |
| Spain | €15,000 | €6,000 | --- | --- | €21,000 | €69,000 |"""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Should extract only content columns
        assert "| Country | Income Tax | Social Security | Total | Net Income |" in result
        assert "| Spain | €15,000 | €6,000 | €21,000 | €69,000 |" in result
        
        # Separators should only be in separator row
        content_lines = [line for line in result.split('\n') if '€' in line]
        for line in content_lines:
            assert '| --- |' not in line
    
    def test_end_to_end_complex_document(self):
        """Test a complex document with multiple table issues."""
        input_text = """# Tax Comparison Report

## Summary

This report compares tax rates across countries.

## Methodology

Item    Description
| Reference year
2024 tax year
| Countries
Spain, France, UK, Germany

## Results

**Tax Rates**| Country | Rate | Effective | Notes |
| --- | --- | --- | --- | Spain | 30% | 27% | Progressive system | France | 33% | 31% | High social charges |

| Country | Income | Tax | --- | Net |
| Germany | €100,000 | €35,000 | --- | €65,000 |

## Conclusion

The analysis shows significant variations."""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Document structure should be preserved
        assert "# Tax Comparison Report" in result
        assert "## Summary" in result
        assert "## Methodology" in result
        assert "## Results" in result
        assert "## Conclusion" in result
        
        # First table should be fixed
        assert "| Item | Description |" in result
        assert "| Reference year | 2024 tax year |" in result
        
        # Second table should be separated from heading
        assert "**Tax Rates**" in result
        assert "| Country | Rate | Effective | Notes |" in result
        
        # Third table should have separators removed from content
        assert "| Country | Income | Tax | Net |" in result
        assert "| Germany | €100,000 | €35,000 | €65,000 |" in result
        
        # Check that tables are properly marked
        assert result.count("TABLE_START") >= 2  # At least 2 complex tables
        assert result.count("TABLE_END") >= 2
    
    def test_streaming_compatibility(self):
        """Test that preprocessed tables work with streaming scenarios."""
        # Simulate a table that might be streamed
        input_text = """Starting analysis...

TABLE_START
| Metric | Value | Status |
| --- | --- | --- |
| Performance | 95% | Good |
| Reliability | 99.9% | Excellent |
TABLE_END

Analysis complete."""
        
        preprocessor = TablePreprocessor()
        result = preprocessor.preprocess_tables(input_text)
        
        # Markers should be preserved for already-marked tables
        assert "TABLE_START" in result
        assert "TABLE_END" in result
        
        # Table should remain intact
        assert "| Metric | Value | Status |" in result
        assert "| Performance | 95% | Good |" in result