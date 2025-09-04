"""
Shared type definitions for the research agent.

This module provides common type definitions, dataclasses, and enums
used throughout the research agent codebase.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Annotated, Sequence
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SearchResultType(str, Enum):
    """Types of search results."""
    WEB = "web"
    VECTOR = "vector"
    HYBRID = "hybrid"


class AgentRole(str, Enum):
    """Agent roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class WorkflowNodeType(str, Enum):
    """Types of workflow nodes."""
    QUERY_GENERATION = "generate_queries"
    WEB_RESEARCH = "web_research"
    VECTOR_RESEARCH = "vector_research"
    REFLECTION = "reflect"
    SYNTHESIS = "synthesize_answer"


class ToolType(str, Enum):
    """Types of tools available."""
    TAVILY_SEARCH = "tavily_search"
    BRAVE_SEARCH = "brave_search"
    VECTOR_SEARCH = "vector_search"
    PYTHON_EXEC = "python_exec"
    UC_FUNCTION = "uc_function"


class IntermediateEventType(str, Enum):
    """Types of intermediate events for enhanced UI tracking."""
    # Action events
    ACTION_START = "action_start"
    ACTION_PROGRESS = "action_progress"
    ACTION_COMPLETE = "action_complete"
    
    # Tool-specific events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_PROGRESS = "tool_call_progress"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_CALL_ERROR = "tool_call_error"
    
    # Reasoning/LLM events
    THOUGHT_SNAPSHOT = "thought_snapshot"
    SYNTHESIS_PROGRESS = "synthesis_progress"
    
    # Content/citation events
    CITATION_ADDED = "citation_added"
    
    # Stage transitions (existing, for compatibility)
    STAGE_TRANSITION = "stage_transition"


class ReasoningVisibility(str, Enum):
    """Control levels for reasoning visibility."""
    HIDDEN = "hidden"
    SUMMARIZED = "summarized"
    RAW = "raw"


@dataclass
class SearchResult:
    """Represents a search result from any source."""
    content: str
    source: str
    url: Optional[str] = None
    title: Optional[str] = None
    score: float = 0.0
    published_date: Optional[str] = None
    result_type: SearchResultType = SearchResultType.WEB
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Represents a citation for a piece of information."""
    source: str
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class IntermediateEvent:
    """Represents an intermediate event emitted during agent execution."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    stage_id: Optional[str] = None
    correlation_id: Optional[str] = None  # Groups related events (thought->action->result)
    sequence: int = 0  # Monotonic sequence number for ordering
    event_type: IntermediateEventType = IntermediateEventType.ACTION_START
    data: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchQuery:
    """Represents a research query with metadata."""
    query: str
    query_id: str = field(default_factory=lambda: str(uuid4()))
    priority: int = 1
    expected_sources: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchContext:
    """Context for a research session."""
    original_question: str
    generated_queries: List[ResearchQuery] = field(default_factory=list)
    web_results: List[SearchResult] = field(default_factory=list)
    vector_results: List[SearchResult] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    reflection: str = ""
    research_loops: int = 0
    max_loops: int = 2
    pending_searches: int = 0
    completed_searches: int = 0
    search_progress: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation context tracking
    conversation_history: List[Any] = field(default_factory=list)  # Previous messages for context
    current_turn_index: int = 0  # Which question in the conversation this is
    previous_research_topics: List[str] = field(default_factory=list)  # Topics from previous turns
    
    # Streaming support
    synthesis_chunks: List[str] = field(default_factory=list)  # Store streamed chunks
    enable_streaming: bool = False  # Enable streaming for this research session
    streaming_chunk_size: int = 50  # Characters per streaming chunk
    
    def add_citation(self, url: str, title: str, snippet: str = None):
        """Add a citation to the context."""
        citation = Citation(source=title, url=url, title=title, snippet=snippet)
        self.citations.append(citation)


@dataclass
class AgentConfiguration:
    """Configuration for the research agent."""
    llm_endpoint: str = "databricks-claude-3-7-sonnet"
    max_research_loops: int = 2
    initial_query_count: int = 3
    max_concurrent_searches: int = 2
    batch_delay_seconds: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 4000
    tavily_api_key: Optional[str] = None
    vector_search_index: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_streaming: bool = True
    enable_citations: bool = True
    
    # Intermediate event configuration
    emit_intermediate_events: bool = True
    reasoning_visibility: ReasoningVisibility = ReasoningVisibility.SUMMARIZED
    thought_snapshot_interval_tokens: int = 40
    thought_snapshot_interval_ms: int = 800
    max_thought_chars_per_step: int = 1000
    redact_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
        r'\b[A-Za-z0-9]{20,}\b',  # potential API keys (20+ alphanumeric chars)
        r'\bsk-[A-Za-z0-9]{32,}\b',  # OpenAI-style API keys
    ])
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_research_loops < 1:
            raise ValueError("max_research_loops must be at least 1")
        if self.initial_query_count < 1:
            raise ValueError("initial_query_count must be at least 1")
        if self.max_concurrent_searches < 1:
            raise ValueError("max_concurrent_searches must be at least 1")
        if self.batch_delay_seconds < 0:
            raise ValueError("batch_delay_seconds must be non-negative")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class ToolConfiguration:
    """Configuration for a specific tool."""
    tool_type: ToolType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None


@dataclass 
class SearchTaskState:
    """State for individual search tasks in parallel execution."""
    search_id: str
    query: str
    query_index: int
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DeduplicationMetrics:
    """Metrics for deduplication performance."""
    original_count: int = 0
    deduplicated_count: int = 0
    reduction_percentage: float = 0.0
    clusters_found: int = 0
    processing_time_seconds: float = 0.0


@dataclass
class CoverageAnalysis:
    """Analysis of result coverage for a query."""
    score: float = 0.0  # 0.0 to 1.0
    missing_aspects: List[str] = field(default_factory=list)
    covered_aspects: List[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_refinement: bool = False


@dataclass
class QueryComplexity:
    """Analysis of query complexity and characteristics."""
    complexity_level: str = "simple"  # simple, complex, compound
    intent_type: str = "factual"  # factual, comparative, exploratory
    entities: List[str] = field(default_factory=list)
    expected_sources: int = 3
    requires_recent_data: bool = False


@dataclass
class WorkflowMetrics:
    """Enhanced metrics for workflow execution."""
    total_queries_generated: int = 0
    total_web_results: int = 0
    total_vector_results: int = 0
    total_research_loops: int = 0
    execution_time_seconds: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    parallel_search_count: int = 0
    average_search_time: float = 0.0
    fastest_search_time: float = 0.0
    slowest_search_time: float = 0.0
    
    # Phase 2 metrics
    deduplication_metrics: Optional[DeduplicationMetrics] = None
    adaptive_queries_generated: int = 0
    coverage_improvement: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)


# Type aliases for better readability
MessageList = Annotated[Sequence[BaseMessage], add_messages]
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
OutputDict = Dict[str, Any]

# Common type unions
ContentType = Union[str, List[Dict[str, Any]]]
ResponseContent = Union[str, Dict[str, Any], List[Dict[str, Any]]]