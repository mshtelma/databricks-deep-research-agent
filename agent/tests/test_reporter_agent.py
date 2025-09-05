"""
Tests for the Reporter Agent.

Tests report generation with multiple styles and citation formatting.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from deep_research_agent.agents.reporter import ReporterAgent
from deep_research_agent.core.report_styles import (
    ReportStyle, StyleTemplate, ReportFormatter, CitationStyle
)
from deep_research_agent.core import Citation
from tests.utils.factories import MockFactory, ResponseFactory


class TestReporterAgent:
    """Tests for the Reporter Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock()
        return llm
    
    @pytest.fixture
    def reporter_agent(self, mock_llm):
        """Create a ReporterAgent instance for testing."""
        config = {
            "report": {
                "default_style": "professional",
                "include_citations": True,
                "include_grounding_markers": True
            }
        }
        return ReporterAgent(llm=mock_llm, config=config)
    
    @pytest.fixture
    def state_with_research(self):
        """Create a state with completed research."""
        state = MockFactory.create_mock_state(
            research_topic="Quantum computing advances"
        )
        
        state["observations"] = [
            "Quantum computers use qubits instead of bits",
            "Recent breakthrough in error correction",
            "Applications in cryptography and drug discovery"
        ]
        
        state["citations"] = [
            MockFactory.create_mock_citation(
                title="Quantum Computing Review",
                source="https://arxiv.org/quantum"
            ),
            MockFactory.create_mock_citation(
                title="Error Correction Methods",
                source="https://nature.com/quantum"
            )
        ]
        
        state["report_style"] = ReportStyle.PROFESSIONAL
        
        return state
    
    def test_reporter_initialization(self, reporter_agent, mock_llm):
        """Test ReporterAgent initialization."""
        assert reporter_agent.llm == mock_llm
        assert reporter_agent.name == "Reporter"
        assert reporter_agent.default_style == ReportStyle.PROFESSIONAL
        assert reporter_agent.include_citations is True
    
    def test_generate_professional_report(self, reporter_agent, mock_llm, state_with_research):
        """Test generation of professional style report."""
        # Mock LLM report generation
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "# Quantum Computing Advances\\n\\n## Executive Summary\\nQuantum computing represents a paradigm shift...\\n\\n## Key Findings\\n- Quantum computers use qubits\\n- Error correction breakthrough\\n- Applications in multiple fields\\n\\n## Conclusion\\nSignificant progress has been made...",
            "sections": {
                "Executive Summary": "Quantum computing represents...",
                "Key Findings": "Multiple breakthroughs...",
                "Conclusion": "Significant progress..."
            }
        }""")
        
        # Generate report
        result = reporter_agent(state_with_research, {})
        
        # Verify report generated
        assert isinstance(result, Command)
        assert "final_report" in result.update
        assert "Quantum Computing Advances" in result.update["final_report"]
        assert "report_sections" in result.update
    
    def test_different_report_styles(self, reporter_agent, mock_llm):
        """Test generation of reports in different styles."""
        styles = [
            ReportStyle.ACADEMIC,
            ReportStyle.POPULAR_SCIENCE,
            ReportStyle.NEWS,
            ReportStyle.TECHNICAL,
            ReportStyle.EXECUTIVE
        ]
        
        for style in styles:
            state = MockFactory.create_mock_state()
            state["report_style"] = style
            state["observations"] = ["Test observation"]
            state["citations"] = []
            
            # Mock style-specific report
            mock_llm.invoke.return_value = AIMessage(content=f'''{{"report": "{style.value} style report content"}}''')
            
            result = reporter_agent(state, {})
            
            assert isinstance(result, Command)
            assert style.value in result.update["final_report"]
    
    def test_citation_formatting(self, reporter_agent, mock_llm):
        """Test citation formatting in reports."""
        state = MockFactory.create_mock_state()
        state["report_style"] = ReportStyle.ACADEMIC
        state["citation_style"] = CitationStyle.APA
        
        # Add multiple citations
        state["citations"] = [
            Citation(
                source="https://journal.com/paper1",
                title="First Paper",
                url="https://journal.com/paper1",
                snippet="Important finding"
            ),
            Citation(
                source="https://arxiv.org/paper2",
                title="Second Paper",
                url="https://arxiv.org/paper2",
                snippet="Supporting evidence"
            )
        ]
        
        state["observations"] = ["Finding from papers"]
        
        # Mock report with citations
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "Research shows important findings (Smith & Doe, 2024)...\\n\\n## References\\nSmith, J., & Doe, A. (2024). First Paper...\\nJohnson, K. (2023). Second Paper...",
            "bibliography": [
                "Smith, J., & Doe, A. (2024). First Paper. Retrieved from https://journal.com/paper1",
                "Johnson, K. (2023). Second Paper. Retrieved from https://arxiv.org/paper2"
            ]
        }""")
        
        result = reporter_agent(state, {})
        
        # Verify citations included
        assert "References" in result.update["final_report"]
        assert "Smith" in result.update["final_report"]
        assert "Johnson" in result.update["final_report"]
    
    def test_grounding_markers_integration(self, reporter_agent, mock_llm):
        """Test integration of grounding markers in report."""
        state = MockFactory.create_mock_state()
        state["enable_grounding"] = True
        
        # Add grounding results
        state["grounding_results"] = [
            MockFactory.create_mock_grounding_result(confidence_score=0.9),
            MockFactory.create_mock_grounding_result(confidence_score=0.85)
        ]
        
        state["factuality_score"] = 0.875
        state["observations"] = ["Grounded observation"]
        
        # Mock report with grounding markers
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "## Findings\\n[✓ High confidence] Finding 1...\\n[✓ Moderate confidence] Finding 2...\\n\\n## Factuality Score: 87.5%",
            "grounding_summary": "87.5% of claims are well-grounded"
        }""")
        
        result = reporter_agent(state, {})
        
        # Verify grounding markers
        assert "confidence" in result.update["final_report"].lower()
        assert "87.5%" in result.update["final_report"]
    
    def test_social_media_style(self, reporter_agent, mock_llm):
        """Test social media style report generation."""
        state = MockFactory.create_mock_state()
        state["report_style"] = ReportStyle.SOCIAL_MEDIA
        state["observations"] = [
            "Breaking: New AI breakthrough",
            "Implications for future tech"
        ]
        
        # Mock social media style
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "🚀 BREAKING: Major AI Breakthrough!\\n\\n🔬 Key Points:\\n• Revolutionary new approach\\n• Game-changing implications\\n\\n💡 Why it matters: This could transform...",
            "hashtags": ["#AI", "#Technology", "#Innovation"]
        }""")
        
        result = reporter_agent(state, {})
        
        # Verify social media formatting
        assert "🚀" in result.update["final_report"]
        assert "#" in result.update["final_report"]
    
    def test_executive_summary_generation(self, reporter_agent, mock_llm):
        """Test executive summary for executive style."""
        state = MockFactory.create_mock_state()
        state["report_style"] = ReportStyle.EXECUTIVE
        state["observations"] = [
            "Market opportunity identified",
            "Competitive advantage possible",
            "Investment required: $10M"
        ]
        
        # Mock executive report
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "# Executive Brief\\n\\n## Bottom Line\\nSignificant market opportunity...\\n\\n## Key Metrics\\n• ROI: 150%\\n• Timeline: 18 months\\n• Investment: $10M\\n\\n## Recommendation\\nProceed with investment...",
            "key_metrics": {
                "roi": "150%",
                "timeline": "18 months",
                "investment": "$10M"
            }
        }""")
        
        result = reporter_agent(state, {})
        
        # Verify executive elements
        assert "Bottom Line" in result.update["final_report"]
        assert "ROI" in result.update["final_report"]
        assert "Recommendation" in result.update["final_report"]
    
    def test_technical_report_generation(self, reporter_agent, mock_llm):
        """Test technical style report with code examples."""
        state = MockFactory.create_mock_state()
        state["report_style"] = ReportStyle.TECHNICAL
        state["observations"] = [
            "Algorithm complexity: O(n log n)",
            "Implementation uses dynamic programming"
        ]
        
        # Mock technical report
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "# Technical Analysis\\n\\n## Algorithm Design\\n```python\\ndef optimized_solution(arr):\\n    # Dynamic programming approach\\n    dp = [0] * len(arr)\\n    ...\\n```\\n\\n## Complexity Analysis\\n- Time: O(n log n)\\n- Space: O(n)",
            "code_examples": ["def optimized_solution..."]
        }""")
        
        result = reporter_agent(state, {})
        
        # Verify technical elements
        assert "```" in result.update["final_report"]
        assert "O(n" in result.update["final_report"]
    
    def test_empty_research_handling(self, reporter_agent, mock_llm):
        """Test handling of empty research results."""
        state = MockFactory.create_mock_state()
        state["observations"] = []
        state["citations"] = []
        
        # Mock minimal report
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "# Research Report\\n\\nNo substantial findings were discovered for this topic.",
            "status": "insufficient_data"
        }""")
        
        result = reporter_agent(state, {})
        
        # Should still generate a report
        assert isinstance(result, Command)
        assert "final_report" in result.update
    
    def test_report_metadata_inclusion(self, reporter_agent, mock_llm, state_with_research):
        """Test inclusion of metadata in report."""
        state_with_research["total_duration_seconds"] = 120
        state_with_research["plan_iterations"] = 2
        state_with_research["factuality_score"] = 0.92
        
        # Mock report with metadata
        mock_llm.invoke.return_value = AIMessage(content="""{
            "report": "# Report\\n\\n---\\n**Research Duration:** 2 minutes\\n**Plan Iterations:** 2\\n**Factuality Score:** 92%\\n---\\n\\nContent...",
            "metadata": {
                "duration": "2 minutes",
                "iterations": 2,
                "factuality": 0.92
            }
        }""")
        
        result = reporter_agent(state_with_research, {})
        
        # Verify metadata included
        assert "Duration" in result.update["final_report"]
        assert "92%" in result.update["final_report"]


class TestReporterIntegration:
    """Integration tests for Reporter with other components."""
    
    def test_reporter_final_output(self):
        """Test that reporter produces final output."""
        mock_llm = Mock()
        reporter = ReporterAgent(llm=mock_llm, config={})
        
        # Setup complete research state
        state = MockFactory.create_mock_state()
        state["observations"] = ["Finding 1", "Finding 2", "Finding 3"]
        state["citations"] = [
            MockFactory.create_mock_citation(),
            MockFactory.create_mock_citation()
        ]
        state["factuality_score"] = 0.88
        
        # Mock final report
        mock_llm.invoke.return_value = AIMessage(content='{"report": "Final comprehensive report..."}')
        
        result = reporter(state, {})
        
        # Should end workflow
        assert result.goto == "end"
        assert "final_report" in result.update
    
    def test_reporter_with_grounding_results(self):
        """Test reporter integration with fact checker results."""
        mock_llm = Mock()
        reporter = ReporterAgent(llm=mock_llm, config={})
        
        # Add fact checking results
        state = MockFactory.create_mock_state()
        state["factuality_report"] = MockFactory.create_mock_factuality_report(
            total_claims=10,
            grounded_claims=9,
            ungrounded_claims=1
        )
        state["observations"] = ["Verified finding"]
        
        mock_llm.invoke.return_value = AIMessage(content='{"report": "Report with 90% verified claims..."}')
        
        result = reporter(state, {})
        
        # Report should reflect factuality
        assert "90%" in result.update["final_report"] or "verified" in result.update["final_report"].lower()
