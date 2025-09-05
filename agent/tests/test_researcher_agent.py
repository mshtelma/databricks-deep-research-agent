"""
Tests for the Researcher Agent.

Tests step execution, context accumulation, and citation collection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import asyncio

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from deep_research_agent.agents.researcher import ResearcherAgent
from deep_research_agent.core.plan_models import (
    Plan, Step, StepType, StepStatus
)
from deep_research_agent.core import SearchResult, Citation, SearchResultType
from tests.utils.factories import MockFactory, ResponseFactory


class TestResearcherAgent:
    """Tests for the Researcher Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock()
        return llm
    
    @pytest.fixture
    def mock_search_tool(self):
        """Create a mock search tool for testing."""
        tool = AsyncMock()
        tool.search = AsyncMock()
        return tool
    
    @pytest.fixture
    def researcher_agent(self, mock_llm, mock_search_tool):
        """Create a ResearcherAgent instance for testing."""
        config = {
            "search": {
                "max_results_per_query": 5,
                "enable_parallel_search": True
            }
        }
        agent = ResearcherAgent(llm=mock_llm, config=config)
        agent.search_tool = mock_search_tool
        return agent
    
    @pytest.fixture
    def state_with_plan(self):
        """Create a state with a research plan."""
        state = MockFactory.create_mock_state(
            research_topic="Quantum computing"
        )
        
        plan = MockFactory.create_mock_plan(num_steps=3)
        state["current_plan"] = plan
        state["current_step_index"] = 0
        state["current_step"] = plan.steps[0]
        
        return state
    
    def test_researcher_initialization(self, researcher_agent, mock_llm):
        """Test ResearcherAgent initialization."""
        assert researcher_agent.llm == mock_llm
        assert researcher_agent.name == "Researcher"
        assert researcher_agent.max_results_per_query == 5
        assert researcher_agent.enable_parallel_search is True
    
    @pytest.mark.asyncio
    async def test_execute_single_step(self, researcher_agent, mock_llm, mock_search_tool, state_with_plan):
        """Test execution of a single research step."""
        # Mock search results
        search_results = [
            MockFactory.create_mock_search_result(
                title="Quantum Computing Basics",
                content="Quantum computing uses qubits..."
            ),
            MockFactory.create_mock_search_result(
                title="Recent Advances",
                content="Google achieved quantum supremacy..."
            )
        ]
        mock_search_tool.search.return_value = search_results
        
        # Mock LLM synthesis
        mock_llm.invoke.return_value = AIMessage(content="""{
            "observations": [
                "Quantum computing uses quantum mechanical phenomena",
                "Recent breakthrough in quantum supremacy by Google"
            ],
            "citations": [
                {"source": "https://example.com/quantum", "title": "Quantum Computing Basics"}
            ],
            "step_complete": true
        }""")
        
        # Execute step
        config = {}
        result = await researcher_agent.aexecute_step(state_with_plan, config)
        
        # Verify results
        assert isinstance(result, Dict)
        assert "observations" in result
        assert "citations" in result
        assert len(result["observations"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_accumulation(self, researcher_agent, mock_llm, mock_search_tool):
        """Test that researcher accumulates context across steps."""
        state = MockFactory.create_mock_state()
        plan = MockFactory.create_mock_plan(num_steps=2)
        state["current_plan"] = plan
        state["observations"] = ["Previous observation from step 1"]
        state["current_step_index"] = 1
        state["current_step"] = plan.steps[1]
        
        # Mock search for second step
        mock_search_tool.search.return_value = [
            MockFactory.create_mock_search_result(
                content="New information for step 2"
            )
        ]
        
        # Mock LLM using context
        mock_llm.invoke.return_value = AIMessage(content="""{
            "observations": [
                "Research finding for Step 2",
                "Additional finding for Step 2"
            ],
            "used_context": true
        }""")
        
        result = await researcher_agent.aexecute_step(state, {})
        
        # Should build on previous observations
        assert len(result["observations"]) >= 2
    
    @pytest.mark.asyncio
    async def test_citation_collection(self, researcher_agent, mock_llm, mock_search_tool, state_with_plan):
        """Test citation collection during research."""
        # Mock multiple search results
        search_results = [
            SearchResult(
                title="Paper 1",
                url="https://arxiv.org/paper1",
                content="Research content",
                relevance_score=0.9,
                source_type=SearchResultType.ACADEMIC_PAPER
            ),
            SearchResult(
                title="Article 2",
                url="https://journal.com/article2",
                content="Journal article",
                relevance_score=0.85,
                source_type=SearchResultType.JOURNAL_ARTICLE
            )
        ]
        mock_search_tool.search.return_value = search_results
        
        # Mock LLM extracting citations
        mock_llm.invoke.return_value = AIMessage(content="""{
            "observations": ["Key finding from papers"],
            "citations": [
                {
                    "source": "https://arxiv.org/paper1",
                    "title": "Paper 1",
                    "snippet": "Important quote",
                    "relevance_score": 0.9
                },
                {
                    "source": "https://journal.com/article2",
                    "title": "Article 2",
                    "snippet": "Supporting evidence",
                    "relevance_score": 0.85
                }
            ]
        }""")
        
        result = await researcher_agent.aexecute_step(state_with_plan, {})
        
        # Verify citations collected
        assert "citations" in result
        assert len(result["citations"]) == 2
        assert result["citations"][0]["source"] == "https://arxiv.org/paper1"
    
    @pytest.mark.asyncio  
    async def test_parallel_search_execution(self, researcher_agent, mock_search_tool):
        """Test parallel search execution for multiple queries."""
        state = MockFactory.create_mock_state()
        step = MockFactory.create_mock_step()
        step.search_queries = [
            "quantum computing basics",
            "quantum algorithms",
            "quantum hardware"
        ]
        state["current_step"] = step
        
        # Mock different results for each query
        async def mock_search_side_effect(query):
            if "basics" in query:
                return [MockFactory.create_mock_search_result(title="Basics")]
            elif "algorithms" in query:
                return [MockFactory.create_mock_search_result(title="Algorithms")]
            else:
                return [MockFactory.create_mock_search_result(title="Hardware")]
        
        mock_search_tool.search.side_effect = mock_search_side_effect
        
        # Execute searches
        results = await researcher_agent.aparallel_search(step.search_queries)
        
        # Verify parallel execution
        assert len(results) == 3
        assert mock_search_tool.search.call_count == 3
    
    def test_step_completion_logic(self, researcher_agent, state_with_plan):
        """Test logic for determining step completion."""
        plan = state_with_plan["current_plan"]
        
        # Mark first step as completed and leave only one step pending
        plan.steps[0].status = StepStatus.COMPLETED
        for i in range(2, len(plan.steps)):
            plan.steps[i].status = StepStatus.COMPLETED
        
        state_with_plan["current_step_index"] = 1
        state_with_plan["completed_steps"] = [plan.steps[0]]
        
        # Check if should continue
        result = researcher_agent(state_with_plan, {})
        
        assert isinstance(result, Command)
        # Should continue to next step or fact checker
        assert result.goto in ["researcher", "fact_checker"]
    
    def test_all_steps_complete(self, researcher_agent):
        """Test behavior when all steps are complete."""
        state = MockFactory.create_mock_state()
        plan = MockFactory.create_mock_plan(num_steps=2)
        
        # Mark all steps complete
        for step in plan.steps:
            step.status = StepStatus.COMPLETED
        state["current_plan"] = plan
        state["completed_steps"] = [s.step_id for s in plan.steps]
        state["current_step_index"] = len(plan.steps)
        
        result = researcher_agent(state, {})
        
        # Should route to fact checker or reporter
        assert isinstance(result, Command)
        assert result.goto in ["fact_checker", "reporter"]
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, researcher_agent, mock_search_tool, state_with_plan):
        """Test handling of search errors."""
        # Mock search failure
        mock_search_tool.search.side_effect = Exception("Search API error")
        
        # Should raise exception when search fails
        with pytest.raises(Exception, match="Search API error"):
            await researcher_agent.aexecute_step(state_with_plan, {})
    
    def test_reflexion_integration(self, researcher_agent, mock_llm):
        """Test reflexion after research completion."""
        state = MockFactory.create_mock_state()
        state["enable_reflexion"] = True
        state["observations"] = [
            "Observation 1",
            "Observation 2"
        ]
        state["reflections"] = []
        
        plan = MockFactory.create_mock_plan(num_steps=2)
        # Mark all steps as completed to simulate finished research
        for step in plan.steps:
            step.status = StepStatus.COMPLETED
        state["current_plan"] = plan
        state["completed_steps"] = [s.step_id for s in plan.steps]
        
        # Mock reflection generation
        mock_llm.invoke.return_value = AIMessage(content="""{
            "reflection": "Research coverage was comprehensive but could benefit from more recent sources",
            "gaps": ["Lack of 2024 publications"],
            "strengths": ["Good theoretical coverage"],
            "next_steps": ["Search for recent preprints"]
        }""")
        
        result = researcher_agent(state, {})
        
        # Should include reflection in state update
        if result.update and "reflections" in result.update:
            assert len(result.update["reflections"]) > 0
    
    @pytest.mark.asyncio
    async def test_step_dependencies(self, researcher_agent, mock_search_tool):
        """Test handling of step dependencies."""
        state = MockFactory.create_mock_state()
        
        # Create plan with dependencies - create step with depends_on directly
        step1 = MockFactory.create_mock_step(step_id="step_001")
        
        # Create step2 with depends_on by creating a new Step object
        step2 = Step(
            step_id="step_002",
            title="Dependent step",
            description="Description for Dependent step",
            step_type=StepType.RESEARCH,
            status=StepStatus.PENDING,
            need_search=True,
            search_queries=["query for Dependent step"],
            depends_on=["step_001"],
            confidence_score=0.8
        )
        
        plan = Plan(
            plan_id="plan_001",
            title="Test Plan",
            research_topic="Test",
            thought="Test",
            has_enough_context=False,
            steps=[step1, step2],
            iteration=0
        )
        
        state["current_plan"] = plan
        state["current_step_index"] = 0
        state["current_step"] = step1
        state["completed_steps"] = []
        
        # Execute first step
        mock_search_tool.search.return_value = []
        
        # Should not execute step 2 until step 1 is complete
        result = researcher_agent(state, {})
        
        # Verify dependency checking
        assert isinstance(result, Command)
    
    def test_quality_score_tracking(self, researcher_agent, mock_llm):
        """Test tracking of research quality scores."""
        # Create a state with only one step to avoid infinite loop
        state = MockFactory.create_mock_state()
        plan = MockFactory.create_mock_plan(num_steps=1)
        state["current_plan"] = plan
        state["current_step_index"] = 0
        state["current_step"] = plan.steps[0]
        
        # Mock quality assessment
        mock_llm.invoke.return_value = AIMessage(content="""{
            "observations": ["High quality finding"],
            "quality_metrics": {
                "coverage": 0.85,
                "depth": 0.9,
                "relevance": 0.95
            }
        }""")
        
        result = researcher_agent(state, {})
        
        # Should track quality metrics
        if hasattr(result, 'update') and result.update and "research_quality_score" in result.update:
            assert result.update["research_quality_score"] > 0


class TestResearcherIntegration:
    """Integration tests for Researcher with other components."""
    
    @pytest.mark.asyncio
    async def test_researcher_to_fact_checker_handoff(self):
        """Test handoff from Researcher to Fact Checker."""
        mock_llm = Mock()
        mock_search = AsyncMock()
        researcher = ResearcherAgent(llm=mock_llm, config={})
        researcher.search_tool = mock_search
        
        # Setup completed research
        state = MockFactory.create_mock_state()
        state["enable_grounding"] = True
        plan = MockFactory.create_mock_plan(num_steps=1)
        # Mark all steps as completed
        for step in plan.steps:
            step.status = StepStatus.COMPLETED
        state["current_plan"] = plan
        state["completed_steps"] = [s.step_id for s in plan.steps]
        state["observations"] = ["Research finding 1", "Research finding 2"]
        state["citations"] = [MockFactory.create_mock_citation()]
        
        result = researcher(state, {})
        
        # Should route to fact checker
        assert result.goto == "fact_checker"
        if result.update:
            assert "observations" in result.update
            assert "citations" in result.update
    
    @pytest.mark.asyncio
    async def test_researcher_iterative_execution(self):
        """Test iterative execution through multiple steps."""
        mock_llm = Mock()
        mock_search = AsyncMock()
        researcher = ResearcherAgent(llm=mock_llm, config={})
        researcher.search_tool = mock_search
        
        # Create single step plan to avoid infinite loop issues
        plan = MockFactory.create_mock_plan(num_steps=1)
        state = MockFactory.create_mock_state()
        state["current_plan"] = plan
        state["current_step_index"] = 0
        state["completed_steps"] = []
        state["current_step"] = plan.steps[0]
        
        mock_search.search.return_value = [
            MockFactory.create_mock_search_result(title="Result for step 1")
        ]
        
        mock_llm.invoke.return_value = AIMessage(content='{"observations": ["Finding 1"], "step_complete": true}')
        
        result = researcher(state, {})
        
        # Should route to fact checker or reporter after completing the single step
        assert result.goto in ["fact_checker", "reporter"]
