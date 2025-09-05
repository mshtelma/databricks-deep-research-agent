"""
Tests for core components of the multi-agent system.

Tests plan models, report styles, and grounding framework.
"""

import pytest
from datetime import datetime
from typing import List

from deep_research_agent.core.plan_models import (
    Plan, Step, StepType, StepStatus, PlanQuality, PlanFeedback
)
from deep_research_agent.core.report_styles import (
    ReportStyle, StyleTemplate, ReportFormatter, CitationStyle
)
from deep_research_agent.core.grounding import (
    Claim, ClaimType, GroundingResult, GroundingStatus,
    FactualityReport, Contradiction, VerificationLevel,
    ClaimExtractor, GroundingChecker, HallucinationPrevention
)
from deep_research_agent.core.multi_agent_state import (
    EnhancedResearchState, StateManager
)

from tests.utils.factories import MockFactory, ResponseFactory


class TestPlanModels:
    """Tests for planning system data models."""
    
    def test_step_creation(self):
        """Test Step model creation and validation."""
        step = MockFactory.create_mock_step(
            step_id="test_001",
            title="Research quantum computing",
            step_type=StepType.RESEARCH
        )
        
        assert step.step_id == "test_001"
        assert step.title == "Research quantum computing"
        assert step.step_type == StepType.RESEARCH
        assert step.status == StepStatus.PENDING
        assert step.need_search is True
        assert step.confidence_score == 0.8
    
    def test_plan_creation(self):
        """Test Plan model creation with steps."""
        plan = MockFactory.create_mock_plan(
            plan_id="plan_001",
            title="Quantum Computing Research Plan",
            num_steps=3
        )
        
        assert plan.plan_id == "plan_001"
        assert plan.title == "Quantum Computing Research Plan"
        assert len(plan.steps) == 3
        assert plan.iteration == 0
        assert plan.has_enough_context is False
        
        # Check step types
        assert plan.steps[0].step_type == StepType.RESEARCH
        assert plan.steps[-1].step_type == StepType.SYNTHESIS
    
    def test_plan_quality_assessment(self):
        """Test PlanQuality model and scoring."""
        plan = MockFactory.create_mock_plan()
        
        assert plan.quality_assessment is not None
        assert plan.quality_assessment.overall_score == 0.825
        assert plan.quality_assessment.completeness_score == 0.8
        assert plan.quality_assessment.feasibility_score == 0.9
        assert plan.quality_assessment.clarity_score == 0.85
        assert plan.quality_assessment.coverage_score == 0.75
    
    def test_plan_feedback(self):
        """Test PlanFeedback model."""
        feedback = PlanFeedback(
            feedback_type="improvement",
            feedback="Add more detail on implementation steps",
            suggestions=[
                "Include timeline estimates",
                "Add resource requirements"
            ],
            requires_revision=True
        )
        
        assert feedback.feedback_type == "improvement"
        assert len(feedback.suggestions) == 2
        assert feedback.requires_revision == True
    
    def test_step_dependencies(self):
        """Test step dependency handling."""
        step1 = MockFactory.create_mock_step(step_id="step_001")
        step2 = MockFactory.create_mock_step(
            step_id="step_002",
            title="Analyze results"
        )
        step2.depends_on = ["step_001"]
        
        assert step2.depends_on == ["step_001"]
        assert step1.depends_on is None


class TestReportStyles:
    """Tests for report styling system."""
    
    def test_report_style_enum(self):
        """Test ReportStyle enum values."""
        styles = [
            ReportStyle.ACADEMIC,
            ReportStyle.POPULAR_SCIENCE,
            ReportStyle.NEWS,
            ReportStyle.SOCIAL_MEDIA,
            ReportStyle.PROFESSIONAL,
            ReportStyle.TECHNICAL,
            ReportStyle.EXECUTIVE
        ]
        
        assert len(styles) == 7
        assert ReportStyle.ACADEMIC.value == "academic"
        assert ReportStyle.EXECUTIVE.value == "executive"
    
    def test_style_template(self):
        """Test StyleTemplate configuration."""
        template = StyleTemplate(
            name="Academic",
            tone="formal",
            language_complexity="high",
            use_technical_terms=True,
            paragraph_length="long",
            sentence_structure="complex",
            citation_style=CitationStyle.APA,
            include_abstract=True,
            include_methodology=True,
            include_references=True
        )
        
        assert template.name == "Academic"
        assert template.tone == "formal"
        assert template.citation_style == CitationStyle.APA
        assert template.include_abstract is True
    
    def test_report_formatter(self):
        """Test ReportFormatter functionality."""
        formatter = ReportFormatter()
        
        # Test section formatting
        section = formatter.format_section(
            title="Introduction",
            content="This is the introduction content.",
            style=ReportStyle.ACADEMIC
        )
        
        assert "Introduction" in section
        assert "This is the introduction content." in section
    
    def test_citation_styles(self):
        """Test CitationStyle enum."""
        styles = [
            CitationStyle.APA,
            CitationStyle.MLA,
            CitationStyle.CHICAGO,
            CitationStyle.IEEE
        ]
        
        assert len(styles) == 4
        assert CitationStyle.APA.value == "APA"
        assert CitationStyle.IEEE.value == "IEEE"
    
    def test_get_style_template(self):
        """Test retrieving style templates."""
        formatter = ReportFormatter()
        
        academic_template = formatter.get_style_template(ReportStyle.ACADEMIC)
        news_template = formatter.get_style_template(ReportStyle.NEWS)
        
        assert "Formal" in academic_template.tone
        assert "objective" in news_template.tone.lower() or "informative" in news_template.tone.lower()
        assert academic_template.citation_style == CitationStyle.APA
        assert news_template.citation_style == CitationStyle.APA  # Default


class TestGroundingFramework:
    """Tests for grounding and factuality checking."""
    
    def test_claim_creation(self):
        """Test Claim model creation."""
        claim = MockFactory.create_mock_claim(
            text="Quantum computers can break RSA encryption",
            claim_type=ClaimType.FACTUAL,
            confidence=0.9
        )
        
        assert claim.text == "Quantum computers can break RSA encryption"
        assert claim.claim_type == ClaimType.FACTUAL
        assert claim.confidence == 0.9
        assert len(claim.entities) > 0
    
    def test_claim_types(self):
        """Test ClaimType enum."""
        types = [
            ClaimType.FACTUAL,
            ClaimType.STATISTICAL,
            ClaimType.OPINION,
            ClaimType.PREDICTION,
            ClaimType.DEFINITION,
            ClaimType.QUOTATION
        ]
        
        assert len(types) == 6
        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.OPINION.value == "opinion"
    
    def test_grounding_result(self):
        """Test GroundingResult model."""
        result = MockFactory.create_mock_grounding_result(
            status=GroundingStatus.GROUNDED,
            confidence_score=0.85
        )
        
        assert result.status == GroundingStatus.GROUNDED
        assert result.confidence_score == 0.85
        assert len(result.supporting_sources) > 0
        assert result.explanation is not None
    
    def test_grounding_status(self):
        """Test GroundingStatus enum."""
        statuses = [
            GroundingStatus.GROUNDED,
            GroundingStatus.PARTIALLY_GROUNDED,
            GroundingStatus.UNGROUNDED,
            GroundingStatus.CONTRADICTED
        ]
        
        assert len(statuses) == 4
        assert GroundingStatus.GROUNDED.value == "grounded"
        assert GroundingStatus.CONTRADICTED.value == "contradicted"
    
    def test_factuality_report(self):
        """Test FactualityReport model."""
        report = MockFactory.create_mock_factuality_report(
            total_claims=10,
            grounded_claims=8,
            ungrounded_claims=2
        )
        
        assert report.total_claims == 10
        assert report.grounded_claims == 8
        assert report.ungrounded_claims == 2
        assert report.overall_factuality_score == 0.8
        assert len(report.grounding_results) == 10
        assert report.verification_level == VerificationLevel.MODERATE
    
    def test_verification_levels(self):
        """Test VerificationLevel enum."""
        levels = [
            VerificationLevel.STRICT,
            VerificationLevel.MODERATE,
            VerificationLevel.LENIENT
        ]
        
        assert len(levels) == 3
        assert VerificationLevel.STRICT.value == "strict"
        assert VerificationLevel.LENIENT.value == "lenient"
    
    def test_contradiction_detection(self):
        """Test Contradiction model."""
        contradiction = MockFactory.create_mock_contradiction(
            claim="AI will replace all jobs by 2030",
            severity=0.8
        )
        
        assert contradiction.claim == "AI will replace all jobs by 2030"
        assert contradiction.severity == 0.8
        assert contradiction.source1 is not None
        assert contradiction.source2 is not None
        assert contradiction.resolution_strategy is not None
    
    def test_claim_extractor(self):
        """Test ClaimExtractor functionality."""
        extractor = ClaimExtractor()
        
        text = "The Earth orbits the Sun. Water boils at 100°C at sea level."
        claims = extractor.extract_claims(text)
        
        # Should extract multiple claims
        assert isinstance(claims, list)
        # Note: Actual extraction requires LLM, so we test structure
    
    def test_grounding_checker(self):
        """Test GroundingChecker functionality."""
        checker = GroundingChecker()
        
        claim = MockFactory.create_mock_claim()
        sources = [MockFactory.create_mock_search_result()]
        
        # Test verify_content method signature
        # Note: Actual checking requires LLM
        assert hasattr(checker, 'verify_content')
        assert callable(checker.verify_content)
    
    def test_hallucination_prevention(self):
        """Test HallucinationPrevention functionality."""
        preventer = HallucinationPrevention()
        
        # Test revision suggestion generation
        ungrounded_claims = [
            MockFactory.create_mock_grounding_result(
                status=GroundingStatus.UNGROUNDED
            )
        ]
        
        suggestions = preventer.generate_revision_suggestions(ungrounded_claims)
        assert isinstance(suggestions, list)


class TestMultiAgentState:
    """Tests for multi-agent state management."""
    
    def test_state_initialization(self):
        """Test EnhancedResearchState initialization."""
        state = MockFactory.create_mock_state(
            research_topic="Quantum computing advances",
            enable_grounding=True
        )
        
        assert state["research_topic"] == "Quantum computing advances"
        assert state["enable_grounding"] is True
        assert state["enable_background_investigation"] is True
        assert state["report_style"] == ReportStyle.PROFESSIONAL
        assert state["messages"] == []
        assert state["current_plan"] is None
    
    def test_state_manager_initialization(self):
        """Test StateManager.initialize_state method."""
        config = {
            "grounding": {"enabled": True},
            "enable_iterative_planning": True,
            "default_report_style": "technical"
        }
        
        state = StateManager.initialize_state(
            research_topic="AI ethics",
            config=config
        )
        
        assert state["research_topic"] == "AI ethics"
        assert state["enable_grounding"] is True
        assert state["enable_iterative_planning"] is True
        assert state["report_style"] == ReportStyle.TECHNICAL
    
    def test_state_update(self):
        """Test StateManager.update_state method."""
        state = MockFactory.create_mock_state()
        
        updates = {
            "current_step": MockFactory.create_mock_step(),
            "observations": ["New observation"],
            "confidence_score": 0.9
        }
        
        updated_state = StateManager.update_state(state, updates)
        
        assert updated_state["current_step"] is not None
        assert len(updated_state["observations"]) == 1
        assert updated_state["confidence_score"] == 0.9
    
    def test_state_validation(self):
        """Test state validation methods."""
        state = MockFactory.create_mock_state()
        
        # Test required fields
        assert "research_topic" in state
        assert "messages" in state
        assert "current_plan" in state
        assert "enable_grounding" in state
        assert "report_style" in state
    
    def test_state_agent_handoffs(self):
        """Test agent handoff tracking."""
        state = MockFactory.create_mock_state()
        
        # Add handoff
        handoff = {
            "from_agent": "coordinator",
            "to_agent": "planner",
            "timestamp": datetime.now(),
            "reason": "Initial planning required"
        }
        
        state["agent_handoffs"].append(handoff)
        
        assert len(state["agent_handoffs"]) == 1
        assert state["agent_handoffs"][0]["from_agent"] == "coordinator"
        assert state["agent_handoffs"][0]["to_agent"] == "planner"


class TestFactoryUtilities:
    """Tests for test factory utilities."""
    
    def test_mock_factory_methods(self):
        """Test all MockFactory methods exist and work."""
        # Test each factory method
        step = MockFactory.create_mock_step()
        assert step is not None
        
        plan = MockFactory.create_mock_plan()
        assert plan is not None
        
        search_result = MockFactory.create_mock_search_result()
        assert search_result is not None
        
        citation = MockFactory.create_mock_citation()
        assert citation is not None
        
        claim = MockFactory.create_mock_claim()
        assert claim is not None
        
        grounding = MockFactory.create_mock_grounding_result()
        assert grounding is not None
        
        report = MockFactory.create_mock_factuality_report()
        assert report is not None
        
        state = MockFactory.create_mock_state()
        assert state is not None
        
        contradiction = MockFactory.create_mock_contradiction()
        assert contradiction is not None
    
    def test_response_factory_methods(self):
        """Test all ResponseFactory methods exist and work."""
        # Note: These create Command objects from langgraph
        # We'll test structure but not full functionality
        
        coord_response = ResponseFactory.create_coordinator_response()
        assert coord_response is not None
        
        planner_response = ResponseFactory.create_planner_response()
        assert planner_response is not None
        
        researcher_response = ResponseFactory.create_researcher_response()
        assert researcher_response is not None
        
        checker_response = ResponseFactory.create_fact_checker_response()
        assert checker_response is not None
        
        reporter_response = ResponseFactory.create_reporter_response()
        assert reporter_response is not None
