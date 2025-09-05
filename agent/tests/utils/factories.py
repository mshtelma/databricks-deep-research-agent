"""
Test factories for creating mock objects for multi-agent system testing.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from deep_research_agent.core import SearchResult, Citation, SearchResultType
from deep_research_agent.core.plan_models import (
    Plan, Step, StepType, StepStatus, PlanQuality, PlanFeedback
)
from deep_research_agent.core.grounding import (
    Claim, ClaimType, GroundingResult, GroundingStatus,
    FactualityReport, Contradiction, VerificationLevel
)
from deep_research_agent.core.report_styles import ReportStyle
from deep_research_agent.core.multi_agent_state import EnhancedResearchState


class MockFactory:
    """Factory for creating mock objects for testing."""
    
    @staticmethod
    def create_mock_step(
        step_id: Optional[str] = None,
        title: str = "Test Step",
        step_type: StepType = StepType.RESEARCH,
        status: StepStatus = StepStatus.PENDING,
        need_search: bool = True
    ) -> Step:
        """Create a mock Step object."""
        return Step(
            step_id=step_id or f"step_{uuid4().hex[:8]}",
            title=title,
            description=f"Description for {title}",
            step_type=step_type,
            status=status,
            need_search=need_search,
            search_queries=[f"query for {title}"],
            depends_on=None,
            confidence_score=0.8
        )
    
    @staticmethod
    def create_mock_plan(
        plan_id: Optional[str] = None,
        title: str = "Test Research Plan",
        num_steps: int = 3,
        has_enough_context: bool = False
    ) -> Plan:
        """Create a mock Plan object."""
        steps = [
            MockFactory.create_mock_step(
                step_id=f"step_{i:03d}",
                title=f"Step {i}",
                step_type=StepType.RESEARCH if i < num_steps-1 else StepType.SYNTHESIS
            )
            for i in range(1, num_steps + 1)
        ]
        
        plan = Plan(
            plan_id=plan_id or f"plan_{uuid4().hex[:8]}",
            title=title,
            research_topic="Test research topic",
            thought="Test planning reasoning",
            has_enough_context=has_enough_context,
            steps=steps,
            iteration=0
        )
        
        # Add quality assessment
        plan.quality_assessment = PlanQuality(
            completeness_score=0.8,
            feasibility_score=0.9,
            clarity_score=0.85,
            coverage_score=0.75,
            overall_score=0.825,
            issues=[],
            suggestions=[]
        )
        
        return plan
    
    @staticmethod
    def create_mock_search_result(
        title: str = "Test Result",
        url: str = "https://example.com",
        content: str = "Test content for search result",
        relevance_score: float = 0.85
    ) -> SearchResult:
        """Create a mock SearchResult object."""
        return SearchResult(
            title=title,
            url=url,
            content=content,
            source="Test Source",
            score=relevance_score,
            result_type=SearchResultType.WEB
        )
    
    @staticmethod
    def create_mock_citation(
        title: str = "Test Citation",
        source: str = "https://example.com",
        relevance_score: float = 0.9
    ) -> Citation:
        """Create a mock Citation object."""
        return Citation(
            source=source,
            title=title,
            url=source,  # using source as URL
            snippet="Test citation snippet",
            relevance_score=relevance_score
        )
    
    @staticmethod
    def create_mock_claim(
        text: str = "Test claim statement",
        claim_type: ClaimType = ClaimType.FACTUAL if hasattr(ClaimType, 'FACTUAL') else list(ClaimType)[0],
        confidence: float = 0.9
    ) -> Claim:
        """Create a mock Claim object."""
        return Claim(
            text=text,
            claim_type=claim_type,
            start_position=0,
            end_position=len(text),
            entities=["TestEntity"],
            confidence=confidence
        )
    
    @staticmethod
    def create_mock_grounding_result(
        claim: Optional[Claim] = None,
        status: GroundingStatus = GroundingStatus.GROUNDED,
        confidence_score: float = 0.85
    ) -> GroundingResult:
        """Create a mock GroundingResult object."""
        if not claim:
            claim = MockFactory.create_mock_claim()
        
        supporting_sources = []
        if status == GroundingStatus.GROUNDED:
            supporting_sources = [
                (MockFactory.create_mock_search_result(), 0.9)
            ]
        
        return GroundingResult(
            claim=claim,
            status=status,
            supporting_sources=supporting_sources,
            contradicting_sources=[],
            confidence_score=confidence_score,
            explanation=f"Test grounding result with status {status}"
        )
    
    @staticmethod
    def create_mock_factuality_report(
        total_claims: int = 10,
        grounded_claims: int = 8,
        ungrounded_claims: int = 2,
        verification_level: VerificationLevel = VerificationLevel.MODERATE
    ) -> FactualityReport:
        """Create a mock FactualityReport object."""
        grounding_results = []
        for i in range(grounded_claims):
            grounding_results.append(
                MockFactory.create_mock_grounding_result(
                    status=GroundingStatus.GROUNDED
                )
            )
        for i in range(ungrounded_claims):
            grounding_results.append(
                MockFactory.create_mock_grounding_result(
                    status=GroundingStatus.UNGROUNDED
                )
            )
        
        return FactualityReport(
            total_claims=total_claims,
            grounded_claims=grounded_claims,
            partially_grounded_claims=0,
            ungrounded_claims=ungrounded_claims,
            contradicted_claims=0,
            overall_factuality_score=grounded_claims / total_claims,
            confidence_score=0.8,
            grounding_results=grounding_results,
            contradictions=[],
            hallucination_risks=[],
            recommendations=["Review ungrounded claims"],
            verification_level=verification_level
        )
    
    @staticmethod
    def create_mock_state(
        research_topic: str = "Test research topic",
        enable_grounding: bool = True,
        enable_background_investigation: bool = True,
        report_style: ReportStyle = ReportStyle.PROFESSIONAL
    ) -> Dict[str, Any]:
        """Create a mock EnhancedResearchState."""
        return {
            "messages": [],
            "research_topic": research_topic,
            "research_context": None,
            "current_plan": None,
            "plan_iterations": 0,
            "plan_feedback": None,
            "plan_quality": None,
            "enable_iterative_planning": True,
            "max_plan_iterations": 3,
            "enable_background_investigation": enable_background_investigation,
            "background_investigation_results": None,
            "observations": [],
            "completed_steps": [],
            "current_step": None,
            "current_step_index": 0,
            "search_results": [],
            "search_queries": [],
            "enable_grounding": enable_grounding,
            "grounding_results": None,
            "factuality_report": None,
            "contradictions": None,
            "factuality_score": None,
            "verification_level": VerificationLevel.MODERATE,
            "citations": [],
            "citation_style": "APA",
            "report_style": report_style,
            "final_report": None,
            "report_sections": None,
            "enable_reflexion": True,
            "reflections": [],
            "reflection_memory_size": 5,
            "current_agent": "coordinator",
            "agent_handoffs": [],
            "research_quality_score": None,
            "coverage_score": None,
            "confidence_score": None,
            "enable_deep_thinking": False,
            "enable_human_feedback": False,
            "auto_accept_plan": True,
            "start_time": datetime.now(),
            "end_time": None,
            "total_duration_seconds": None,
            "errors": [],
            "warnings": [],
            "user_preferences": {}
        }
    
    @staticmethod
    def create_mock_contradiction(
        claim: str = "Contradictory claim",
        severity: float = 0.7
    ) -> Contradiction:
        """Create a mock Contradiction object."""
        return Contradiction(
            claim=claim,
            source1=MockFactory.create_mock_search_result("Source 1"),
            source2=MockFactory.create_mock_search_result("Source 2"),
            severity=severity,
            explanation="Test contradiction between sources",
            resolution_strategy="Prefer more recent source"
        )


class ResponseFactory:
    """Factory for creating mock agent responses."""
    
    @staticmethod
    def create_coordinator_response(
        route_to: str = "planner",
        research_topic: str = "Test topic"
    ) -> Dict[str, Any]:
        """Create a mock coordinator response."""
        from langgraph.types import Command
        return Command(
            goto=route_to,
            update={"research_topic": research_topic}
        )
    
    @staticmethod
    def create_planner_response(
        plan: Optional[Plan] = None,
        route_to: str = "researcher"
    ) -> Dict[str, Any]:
        """Create a mock planner response."""
        from langgraph.types import Command
        if not plan:
            plan = MockFactory.create_mock_plan()
        return Command(
            goto=route_to,
            update={"current_plan": plan, "plan_iterations": 1}
        )
    
    @staticmethod
    def create_researcher_response(
        observations: List[str] = None,
        citations: List[Citation] = None,
        route_to: str = "fact_checker"
    ) -> Dict[str, Any]:
        """Create a mock researcher response."""
        from langgraph.types import Command
        if not observations:
            observations = ["Test observation 1", "Test observation 2"]
        if not citations:
            citations = [MockFactory.create_mock_citation()]
        
        return Command(
            goto=route_to,
            update={
                "observations": observations,
                "citations": citations
            }
        )
    
    @staticmethod
    def create_fact_checker_response(
        factuality_report: Optional[FactualityReport] = None,
        route_to: str = "reporter"
    ) -> Dict[str, Any]:
        """Create a mock fact checker response."""
        from langgraph.types import Command
        if not factuality_report:
            factuality_report = MockFactory.create_mock_factuality_report()
        
        return Command(
            goto=route_to,
            update={
                "factuality_report": factuality_report,
                "factuality_score": factuality_report.overall_factuality_score
            }
        )
    
    @staticmethod
    def create_reporter_response(
        final_report: str = "Test final report content",
        report_sections: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a mock reporter response."""
        from langgraph.types import Command
        if not report_sections:
            report_sections = {
                "Introduction": "Test introduction",
                "Findings": "Test findings",
                "Conclusion": "Test conclusion"
            }
        
        return Command(
            goto="end",
            update={
                "final_report": final_report,
                "report_sections": report_sections
            }
        )