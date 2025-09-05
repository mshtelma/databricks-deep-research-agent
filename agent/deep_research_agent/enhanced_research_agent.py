"""
Enhanced Research Agent with Multi-Agent Architecture.

Complete integration of all multi-agent components with planning,
grounding, and report generation capabilities.
"""

import asyncio
import yaml
from typing import Dict, Any, Optional, AsyncIterator
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from deep_research_agent.core import get_logger
from deep_research_agent.core.multi_agent_state import (
    EnhancedResearchState,
    StateManager
)
from deep_research_agent.core.report_styles import ReportStyle
from deep_research_agent.core.grounding import VerificationLevel
from deep_research_agent.workflow_nodes_enhanced import EnhancedWorkflowNodes
from deep_research_agent.agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ReporterAgent,
    FactCheckerAgent
)


logger = get_logger(__name__)


class EnhancedResearchAgent:
    """
    Enhanced Research Agent with multi-agent architecture.
    
    Features:
    - Multi-agent coordination (Coordinator, Planner, Researcher, Reporter, Fact Checker)
    - Background investigation before planning
    - Iterative plan refinement with quality assessment
    - Step-by-step execution with context accumulation
    - Comprehensive grounding and factuality checking
    - Multiple report styles
    - Citation management
    - Reflexion-style self-improvement
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm=None,
        tool_registry=None,
        **kwargs
    ):
        """
        Initialize the enhanced research agent.
        
        Args:
            config_path: Path to configuration file
            llm: Language model to use
            tool_registry: Registry of available tools
            **kwargs: Additional configuration overrides
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value
        
        # Initialize components
        self.llm = llm
        self.tool_registry = tool_registry
        
        # Initialize config manager and agent config early
        self.agent_config = self.config
        # Create a simple config manager for compatibility
        from deep_research_agent.core import ConfigManager
        self.config_manager = ConfigManager(config_override=self.config)
        
        # Initialize semaphore early (before workflow nodes)
        self.search_semaphore = asyncio.Semaphore(
            self.config.get("rate_limiting", {}).get("max_parallel_requests", 10)
        )
        
        # Create workflow nodes (after config_manager and semaphore are set)
        self.workflow_nodes = EnhancedWorkflowNodes(self)
        
        # Build workflow graph
        self.graph = self._build_graph()
        
        logger.info("Enhanced Research Agent initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Use default configuration
            config = {
                "multi_agent": {"enabled": True},
                "planning": {
                    "enable_iterative_planning": True,
                    "max_plan_iterations": 3,
                    "plan_quality_threshold": 0.7,
                    "auto_accept_plan": True
                },
                "background_investigation": {"enabled": True},
                "grounding": {
                    "enabled": True,
                    "verification_level": "moderate"
                },
                "report": {"default_style": "professional"},
                "reflexion": {"enabled": True}
            }
        
        return config
    
    def _build_graph(self) -> Any:
        """Build the LangGraph workflow."""
        
        # Create state graph
        # The graph expects a concrete Python type (like dict), not the typing alias Dict
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("coordinator", self.workflow_nodes.coordinator_node)
        workflow.add_node("background_investigation", self.workflow_nodes.background_investigation_node)
        workflow.add_node("planner", self.workflow_nodes.planner_node)
        workflow.add_node("researcher", self.workflow_nodes.researcher_node)
        workflow.add_node("fact_checker", self.workflow_nodes.fact_checker_node)
        workflow.add_node("reporter", self.workflow_nodes.reporter_node)
        workflow.add_node("human_feedback", self.workflow_nodes.human_feedback_node)
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        # Add conditional edges for coordinator
        def coordinator_router(state):
            """Route from coordinator based on state."""
            if state.get("research_topic"):
                if state.get("enable_background_investigation", True):
                    return "background_investigation"
                return "planner"
            return END
        
        workflow.add_conditional_edges(
            "coordinator",
            coordinator_router,
            {
                "background_investigation": "background_investigation",
                "planner": "planner",
                END: END
            }
        )
        
        # Background investigation always goes to planner
        workflow.add_edge("background_investigation", "planner")
        
        # Add conditional edges for planner
        def planner_router(state):
            """Route from planner based on plan and settings."""
            plan = state.get("current_plan")
            if not plan:
                return END
            
            if state.get("enable_human_feedback") and not state.get("auto_accept_plan"):
                return "human_feedback"
            
            if hasattr(plan, "has_enough_context") and plan.has_enough_context:
                return "reporter"
            
            return "researcher"
        
        workflow.add_conditional_edges(
            "planner",
            planner_router,
            {
                "researcher": "researcher",
                "human_feedback": "human_feedback",
                "reporter": "reporter",
                END: END
            }
        )
        
        # Human feedback routing
        def human_feedback_router(state):
            """Route from human feedback."""
            # Could go back to planner or proceed
            return "researcher"
        
        workflow.add_conditional_edges(
            "human_feedback",
            human_feedback_router,
            {
                "planner": "planner",
                "researcher": "researcher"
            }
        )
        
        # Researcher routing
        def researcher_router(state):
            """Route from researcher based on completion."""
            plan = state.get("current_plan")
            # If there are still pending steps, continue researching
            try:
                if plan and hasattr(plan, "get_next_step") and plan.get_next_step() is not None:
                    return "researcher"
            except Exception:
                # If plan is not well-formed, fall through to default behavior
                pass
            # No pending steps: proceed based on grounding setting
            if state.get("enable_grounding", True):
                return "fact_checker"
            return "reporter"
        
        workflow.add_conditional_edges(
            "researcher",
            researcher_router,
            {
                "fact_checker": "fact_checker",
                "reporter": "reporter",
                "researcher": "researcher",  # Continue research
                "planner": "planner"  # Re-plan if needed
            }
        )
        
        # Fact checker routing
        def fact_checker_router(state):
            """Route from fact checker based on results."""
            factuality_score = state.get("factuality_score", 1.0)
            verification_level = state.get("verification_level", VerificationLevel.MODERATE)
            plan = state.get("current_plan")
            # If plan is complete and there are no sources/citations to improve factuality, finish
            try:
                no_sources = (not state.get("search_results")) and (not state.get("citations"))
                plan_complete = bool(plan and hasattr(plan, "is_complete") and plan.is_complete())
                if plan_complete and no_sources:
                    return "reporter"
            except Exception:
                pass
            if verification_level == VerificationLevel.STRICT and factuality_score < 0.9:
                return "planner"  # Re-plan for better factuality
            if factuality_score < 0.7:
                return "researcher"  # More research needed
            return "reporter"
        
        workflow.add_conditional_edges(
            "fact_checker",
            fact_checker_router,
            {
                "reporter": "reporter",
                "researcher": "researcher",
                "planner": "planner"
            }
        )
        
        # Reporter ends the workflow
        workflow.add_edge("reporter", END)
        
        # Compile with checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def research(
        self,
        query: str,
        report_style: Optional[ReportStyle] = None,
        verification_level: Optional[VerificationLevel] = None,
        enable_streaming: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute research on a query.
        
        Args:
            query: Research question or topic
            report_style: Style for the final report
            verification_level: Strictness of fact checking
            enable_streaming: Whether to stream results
            **kwargs: Additional configuration overrides
            
        Returns:
            Research results including final report and metadata
        """
        logger.info(f"Starting research for: {query}")
        
        # Initialize state
        initial_state = StateManager.initialize_state(
            research_topic=query,
            config=self.config
        )
        
        # Apply overrides
        if report_style:
            initial_state["report_style"] = report_style
        if verification_level:
            initial_state["verification_level"] = verification_level
        
        # Add user message
        initial_state["messages"] = [HumanMessage(content=query)]
        
        # Configure from kwargs
        for key, value in kwargs.items():
            if value is not None:
                initial_state[key] = value
        
        # Execute workflow
        if enable_streaming:
            # Return an async generator directly for the caller to iterate over
            return self._research_streaming(initial_state)
        # Batch mode returns a regular dictionary after awaiting
        return await self._research_batch(initial_state)
    
    async def _research_batch(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research in batch mode."""
        try:
            # Run the graph
            config = {"configurable": {"thread_id": "research_thread"}}
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Extract results
            return self._extract_results(final_state)
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "final_report": f"Research failed: {str(e)}"
            }
    
    async def _research_streaming(
        self,
        initial_state: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute research in streaming mode."""
        try:
            config = {"configurable": {"thread_id": "research_thread"}}
            
            # Stream events from the graph
            async for event in self.graph.astream_events(initial_state, config):
                yield self._process_stream_event(event)
                
        except Exception as e:
            logger.error(f"Streaming research failed: {str(e)}")
            yield {
                "type": "error",
                "content": str(e)
            }
    
    def _extract_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final results from state."""
        return {
            "success": True,
            "final_report": state.get("final_report", ""),
            "research_topic": state.get("research_topic", ""),
            "report_style": state.get("report_style", ReportStyle.PROFESSIONAL),
            "factuality_score": state.get("factuality_score", None),
            "confidence_score": state.get("confidence_score", None),
            "citations": state.get("citations", []),
            "observations": state.get("observations", []),
            "plan": state.get("current_plan", None),
            "grounding_results": state.get("grounding_results", []),
            "contradictions": state.get("contradictions", []),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "reflections": state.get("reflections", []),
            "metadata": {
                "total_duration": state.get("total_duration_seconds"),
                "plan_iterations": state.get("plan_iterations", 0),
                "research_loops": len(state.get("completed_steps", [])),
                "total_sources": len(state.get("citations", []))
            }
        }
    
    def _process_stream_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a streaming event."""
        # Extract relevant information from event
        if "data" in event:
            data = event["data"]
            
            # Determine event type
            if "observation" in data:
                return {
                    "type": "observation",
                    "content": data["observation"]
                }
            elif "plan" in data:
                return {
                    "type": "plan",
                    "content": data["plan"]
                }
            elif "final_report" in data:
                return {
                    "type": "final_report",
                    "content": data["final_report"]
                }
            else:
                return {
                    "type": "progress",
                    "content": data
                }
        
        return {
            "type": "event",
            "content": event
        }


# Example usage
async def main():
    """Example usage of the Enhanced Research Agent."""
    
    # Initialize agent with configuration
    agent = EnhancedResearchAgent(
        config_path="agent_config_enhanced.yaml",
        enable_grounding=True,
        enable_background_investigation=True,
        default_report_style=ReportStyle.ACADEMIC,
        verification_level=VerificationLevel.STRICT
    )
    
    # Execute research
    query = "What are the latest breakthroughs in quantum computing and their implications for cryptography?"
    
    # Batch mode
    results = await agent.research(
        query=query,
        report_style=ReportStyle.TECHNICAL,
        verification_level=VerificationLevel.STRICT,
        enable_iterative_planning=True,
        enable_reflexion=True
    )
    
    # Display results
    print(f"Research Topic: {results['research_topic']}")
    print(f"Report Style: {results['report_style']}")
    print(f"Factuality Score: {results.get('factuality_score', 'N/A')}")
    print(f"Confidence Score: {results.get('confidence_score', 'N/A')}")
    print(f"Total Sources: {results['metadata']['total_sources']}")
    print(f"Plan Iterations: {results['metadata']['plan_iterations']}")
    print("\n" + "="*50 + "\n")
    print(results['final_report'])
    
    # Streaming mode example
    print("\n" + "="*50 + "\n")
    print("Streaming mode example:")
    
    async for event in await agent.research(
        query="Explain the concept of neural networks",
        report_style=ReportStyle.POPULAR_SCIENCE,
        enable_streaming=True
    ):
        if event["type"] == "observation":
            print(f"📊 Observation: {event['content'][:100]}...")
        elif event["type"] == "plan":
            print(f"📋 Plan update: {event['content']}")
        elif event["type"] == "final_report":
            print(f"📄 Final Report Ready!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())