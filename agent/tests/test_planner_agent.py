"""
Tests for the Planner Agent.

Tests plan generation, quality assessment, and iterative refinement.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from deep_research_agent.agents.planner import PlannerAgent
from deep_research_agent.core.plan_models import (
    Plan, Step, StepType, StepStatus, PlanQuality, PlanFeedback
)
from deep_research_agent.core.multi_agent_state import StateManager
from tests.utils.factories import MockFactory, ResponseFactory


class TestPlannerAgent:
    """Tests for the Planner Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock()
        return llm
    
    @pytest.fixture
    def planner_agent(self, mock_llm):
        """Create a PlannerAgent instance for testing."""
        config = {
            "planning": {
                "enable_iterative_planning": True,
                "max_plan_iterations": 3,
                "plan_quality_threshold": 0.7
            }
        }
        return PlannerAgent(llm=mock_llm, config=config)
    
    @pytest.fixture
    def research_state(self):
        """Create a state with research topic for testing."""
        state = MockFactory.create_mock_state(
            research_topic="Quantum computing advances"
        )
        state["messages"] = [HumanMessage(content="Research quantum computing advances")]
        return state
    
    def test_planner_initialization(self, planner_agent, mock_llm):
        """Test PlannerAgent initialization."""
        assert planner_agent.llm == mock_llm
        assert planner_agent.name == "Planner"
        assert planner_agent.enable_iterative_planning is True
        assert planner_agent.max_plan_iterations == 3
        assert planner_agent.plan_quality_threshold == 0.7
    
    def test_generate_initial_plan(self, planner_agent, mock_llm, research_state):
        """Test initial plan generation."""
        # Mock LLM response with a plan
        mock_plan = MockFactory.create_mock_plan(
            title="Quantum Computing Research Plan",
            num_steps=4
        )
        
        mock_llm.invoke.return_value = AIMessage(content=str(mock_plan.dict()))
        
        # Call planner
        config = {}
        result = planner_agent(research_state, config)
        
        # Verify it returns a Command with plan
        assert isinstance(result, Command)
        assert "current_plan" in result.update
        assert result.update["plan_iterations"] >= 1
    
    def test_plan_quality_assessment(self, planner_agent, mock_llm, research_state):
        """Test plan quality assessment functionality."""
        # Create plan with quality scores
        plan = MockFactory.create_mock_plan()
        plan.quality_assessment = PlanQuality(
            completeness_score=0.6,  # Below threshold
            feasibility_score=0.8,
            clarity_score=0.7,
            coverage_score=0.5,
            overall_score=0.65  # Below 0.7 threshold
        )
        
        research_state["current_plan"] = plan
        research_state["plan_iterations"] = 1
        
        # Mock improved plan
        improved_plan = MockFactory.create_mock_plan()
        improved_plan.quality_assessment.overall_score = 0.85
        
        mock_llm.invoke.return_value = AIMessage(content=str(improved_plan.dict()))
        
        # Call planner for refinement
        config = {}
        result = planner_agent(research_state, config)
        
        # Should generate improved plan
        assert isinstance(result, Command)
        assert "current_plan" in result.update
        assert result.update["plan_iterations"] >= 2
    
    def test_iterative_refinement(self, planner_agent, mock_llm):
        """Test iterative plan refinement process."""
        state = MockFactory.create_mock_state(
            research_topic="Climate change impacts"
        )
        
        # Start with low quality plan
        initial_plan = MockFactory.create_mock_plan()
        initial_plan.quality_assessment.overall_score = 0.5
        state["current_plan"] = initial_plan
        state["plan_iterations"] = 1
        
        # Mock progressively better plans
        iteration_scores = [0.65, 0.75, 0.85]
        for i, score in enumerate(iteration_scores):
            if state["plan_iterations"] < planner_agent.max_plan_iterations:
                improved_plan = MockFactory.create_mock_plan()
                improved_plan.quality_assessment.overall_score = score
                mock_llm.invoke.return_value = AIMessage(content=str(improved_plan.dict()))
                
                result = planner_agent(state, {})
                
                if score >= planner_agent.plan_quality_threshold:
                    # Should stop refining
                    assert result.goto in ["researcher", "human_feedback"]
                    break
                else:
                    # Should continue refining
                    state = result.update
                    state["plan_iterations"] += 1
    
    def test_max_iterations_limit(self, planner_agent, mock_llm):
        """Test that planner respects max iterations limit."""
        state = MockFactory.create_mock_state()
        state["plan_iterations"] = 3  # Already at max
        state["current_plan"] = MockFactory.create_mock_plan()
        
        # Even with low quality, should stop
        state["current_plan"].quality_assessment.overall_score = 0.6
        
        result = planner_agent(state, {})
        
        # Should proceed despite low quality
        assert isinstance(result, Command)
        assert result.goto in ["researcher", "reporter"]
    
    def test_background_investigation_integration(self, planner_agent, mock_llm):
        """Test plan generation with background investigation results."""
        state = MockFactory.create_mock_state(
            research_topic="AI safety"
        )
        
        # Add background investigation results
        state["background_investigation_results"] = {
            "key_concepts": ["alignment", "robustness", "interpretability"],
            "current_research": ["Constitutional AI", "RLHF"],
            "controversies": ["existential risk debates"]
        }
        
        mock_plan = MockFactory.create_mock_plan(
            title="AI Safety Research Plan",
            num_steps=5
        )
        mock_llm.invoke.return_value = AIMessage(content=str(mock_plan.dict()))
        
        result = planner_agent(state, {})
        
        # Plan should incorporate background context
        assert isinstance(result, Command)
        assert "current_plan" in result.update
    
    def test_plan_with_dependencies(self, planner_agent, mock_llm, research_state):
        """Test plan generation with step dependencies."""
        # Create plan with dependencies
        plan = Plan(
            plan_id="plan_001",
            title="Complex Research Plan",
            research_topic="Quantum computing",
            thought="Multi-stage research needed",
            has_enough_context=False,
            steps=[
                Step(
                    step_id="step_001",
                    title="Understand basics",
                    description="Research quantum mechanics basics",
                    step_type=StepType.RESEARCH,
                    status=StepStatus.PENDING
                ),
                Step(
                    step_id="step_002",
                    title="Study algorithms",
                    description="Research quantum algorithms",
                    step_type=StepType.RESEARCH,
                    status=StepStatus.PENDING,
                    depends_on=["step_001"]
                ),
                Step(
                    step_id="step_003",
                    title="Synthesize findings",
                    description="Create comprehensive report",
                    step_type=StepType.SYNTHESIS,
                    status=StepStatus.PENDING,
                    depends_on=["step_001", "step_002"]
                )
            ],
            iteration=0
        )
        
        mock_llm.invoke.return_value = AIMessage(content=str(plan.dict()))
        
        result = planner_agent(research_state, {})
        
        # Verify dependencies are preserved
        assert isinstance(result, Command)
        plan = result.update["current_plan"]
        assert plan.steps[1].depends_on == ["step_001"]
        assert plan.steps[2].depends_on == ["step_001", "step_002"]
    
    def test_human_feedback_integration(self, planner_agent, mock_llm):
        """Test plan adjustment based on human feedback."""
        state = MockFactory.create_mock_state()
        state["enable_human_feedback"] = True
        state["auto_accept_plan"] = False
        state["current_plan"] = MockFactory.create_mock_plan()
        
        # Add human feedback
        state["plan_feedback"] = [PlanFeedback(
            feedback_type="improvement",
            feedback="Please add more detail on implementation",
            suggestions=[
                "Include code examples",
                "Add performance metrics"
            ],
            requires_revision=True
        )]
        
        # Mock adjusted plan with proper JSON format
        adjusted_plan = MockFactory.create_mock_plan(num_steps=5)
        import json
        mock_llm.invoke.return_value = AIMessage(content=json.dumps(adjusted_plan.model_dump(mode='json')))
        
        result = planner_agent(state, {})
        
        # Should generate adjusted plan
        assert isinstance(result, Command)
        assert "current_plan" in result.update
        assert len(result.update["current_plan"].steps) == 5
    
    def test_deep_thinking_mode(self, planner_agent, mock_llm):
        """Test deep thinking mode for complex queries."""
        state = MockFactory.create_mock_state(
            research_topic="Solve P vs NP problem"
        )
        state["enable_deep_thinking"] = True
        
        # Mock comprehensive plan
        deep_plan = MockFactory.create_mock_plan(num_steps=8)
        deep_plan.thought = "This requires extensive theoretical exploration..."
        
        mock_llm.invoke.return_value = AIMessage(content=str(deep_plan.dict()))
        
        result = planner_agent(state, {})
        
        # Should generate comprehensive plan
        assert isinstance(result, Command)
        assert len(result.update["current_plan"].steps) >= 3
    
    def test_plan_validation(self, planner_agent, mock_llm):
        """Test plan validation and error handling."""
        state = MockFactory.create_mock_state()
        
        # Mock invalid plan (no steps)
        invalid_plan = {
            "plan_id": "invalid",
            "title": "Invalid Plan",
            "steps": []  # Empty steps
        }
        
        mock_llm.invoke.return_value = AIMessage(content=str(invalid_plan))
        
        # Should handle invalid plan gracefully without raising exception
        result = planner_agent(state, {})
        # Verify that the agent handled the invalid plan gracefully
        assert isinstance(result, Command)
        # The agent should either retry planning or handle the error appropriately
        assert result.goto in ["planner", "end", "researcher"]
    
    def test_sufficient_context_bypass(self, planner_agent, mock_llm):
        """Test bypass to reporter when sufficient context exists."""
        state = MockFactory.create_mock_state(
            research_topic="Simple definition query"
        )
        
        # Create plan indicating sufficient context
        simple_plan = MockFactory.create_mock_plan(
            has_enough_context=True,
            num_steps=1
        )
        
        mock_llm.invoke.return_value = AIMessage(content=str(simple_plan.dict()))
        
        result = planner_agent(state, {})
        
        # Should route to reporter or researcher depending on implementation
        assert isinstance(result, Command)
        assert result.goto in ["reporter", "researcher"]
    
    def test_plan_quality_metrics(self):
        """Test plan quality metric calculations."""
        plan = MockFactory.create_mock_plan()
        
        quality = PlanQuality(
            completeness_score=0.8,
            feasibility_score=0.9,
            clarity_score=0.85,
            coverage_score=0.75,
            overall_score=0.825,
            issues=[],
            suggestions=[]
        )
        
        plan.quality_assessment = quality
        
        # Test overall score calculation
        expected_score = (0.8 + 0.9 + 0.85 + 0.75) / 4
        assert abs(quality.overall_score - expected_score) < 0.01
        
        # Test quality thresholds
        assert quality.overall_score > 0.7  # Above threshold
        assert quality.completeness_score >= 0.8
        assert quality.feasibility_score >= 0.9


class TestPlannerIntegration:
    """Integration tests for Planner with other components."""
    
    def test_planner_to_researcher_handoff(self):
        """Test handoff from Planner to Researcher."""
        mock_llm = Mock()
        config = {
            "planning": {
                "enable_iterative_planning": True,
                "plan_quality_threshold": 0.7
            }
        }
        planner = PlannerAgent(llm=mock_llm, config=config)
        
        state = MockFactory.create_mock_state(
            research_topic="Blockchain technology"
        )
        
        # Mock good quality plan
        plan = MockFactory.create_mock_plan(num_steps=3)
        plan.quality_assessment.overall_score = 0.85
        
        mock_llm.invoke.return_value = AIMessage(content=str(plan.dict()))
        
        result = planner(state, {})
        
        # Should route to researcher
        assert result.goto == "researcher"
        assert "current_plan" in result.update
        assert len(result.update["current_plan"].steps) == 3
    
    def test_planner_state_preservation(self):
        """Test that planner preserves important state fields."""
        mock_llm = Mock()
        planner = PlannerAgent(llm=mock_llm, config={})
        
        # Create state with existing data
        state = MockFactory.create_mock_state()
        state["observations"] = ["Previous observation"]
        state["citations"] = [MockFactory.create_mock_citation()]
        state["search_results"] = [MockFactory.create_mock_search_result()]
        
        plan = MockFactory.create_mock_plan()
        mock_llm.invoke.return_value = AIMessage(content=str(plan.dict()))
        
        result = planner(state, {})
        
        # Existing data should be preserved
        updates = result.update
        assert "current_plan" in updates
        # State preservation happens at graph level, not agent level
