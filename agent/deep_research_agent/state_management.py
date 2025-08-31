"""
State management classes for the research agent.

This module contains the state classes used by the LangGraph workflow
to manage research context and progress.
"""

import time
import operator
from typing import Dict, Any, List, Optional
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages

from deep_research_agent.core import (
    ResearchContext,
    WorkflowMetrics,
    WorkflowNodeType,
    URLResolver,
    SearchTaskState,
    MessageList
)


class AgentStateDict(TypedDict):
    """TypedDict for LangGraph state management with parallel update support."""
    messages: Annotated[list, add_messages]
    research_context: ResearchContext
    current_node: Optional[str]
    start_time: float
    url_resolver: URLResolver
    search_tasks: Dict[str, SearchTaskState]
    batch_info: Optional[Dict[str, Any]]
    # Key field for parallel search results accumulation
    parallel_search_results: Annotated[list, operator.add]


class RefactoredResearchAgentState:
    """Enhanced state management for research workflow."""
    
    def __init__(self):
        """Initialize state."""
        self.messages: MessageList = []
        self.research_context = ResearchContext(original_question="")
        self.workflow_metrics = WorkflowMetrics()
        self.current_node: Optional[WorkflowNodeType] = None
        self.start_time = time.time()
        self.url_resolver = URLResolver()
        self.search_tasks: Dict[str, SearchTaskState] = {}
        self.batch_info: Optional[Dict[str, Any]] = None
    
    def update_metrics(self, **kwargs):
        """Update workflow metrics."""
        for key, value in kwargs.items():
            if hasattr(self.workflow_metrics, key):
                setattr(self.workflow_metrics, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for LangGraph."""
        return {
            "messages": self.messages,
            "research_context": self.research_context,
            "current_node": self.current_node.value if self.current_node else None,
            "start_time": self.start_time,
            "url_resolver": self.url_resolver,
            "search_tasks": self.search_tasks,
            "batch_info": self.batch_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefactoredResearchAgentState":
        """Create state from dictionary."""
        state = cls()
        state.messages = data.get("messages", [])
        state.research_context = data.get("research_context", ResearchContext(original_question=""))
        
        current_node = data.get("current_node")
        if current_node:
            state.current_node = WorkflowNodeType(current_node)
        
        state.start_time = data.get("start_time", time.time())
        state.url_resolver = data.get("url_resolver", URLResolver())
        state.search_tasks = data.get("search_tasks", {})
        state.batch_info = data.get("batch_info", None)
        return state