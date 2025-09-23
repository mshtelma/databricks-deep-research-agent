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
    ACADEMIC_PAPER = "academic_paper"  # Added for scholarly sources
    JOURNAL_ARTICLE = "journal_article"
    WEB_PAGE = "web_page"


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
    
    # Enhanced LLM events
    LLM_PROMPT_SENT = "llm_prompt_sent"
    LLM_STREAMING = "llm_streaming"
    LLM_RESPONSE_COMPLETE = "llm_response_complete"
    LLM_THINKING = "llm_thinking"
    
    # Enhanced agent events
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_THINKING = "agent_thinking"
    AGENT_DECISION = "agent_decision"
    
    # Query and search events
    QUERY_GENERATED = "query_generated"
    QUERY_EXECUTING = "query_executing"
    SEARCH_RESULTS_FOUND = "search_results_found"
    
    # Content/citation events
    CITATION_ADDED = "citation_added"
    SOURCE_ANALYZED = "source_analyzed"
    
    # Multi-agent specific events
    AGENT_HANDOFF = "agent_handoff"
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    PLAN_ITERATION = "plan_iteration"
    BACKGROUND_INVESTIGATION = "background_investigation"
    GROUNDING_START = "grounding_start"
    GROUNDING_COMPLETE = "grounding_complete"
    GROUNDING_CONTRADICTION = "grounding_contradiction"
    REPORT_GENERATION = "report_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    
    # Stage transitions (existing, for compatibility)
    STAGE_TRANSITION = "stage_transition"
    
    # NEW: Enhanced event types for transparent UI
    # Reasoning and reflection events
    REASONING_REFLECTION = "reasoning_reflection"
    HYPOTHESIS_FORMED = "hypothesis_formed"
    SOURCE_EVALUATION = "source_evaluation"
    PARTIAL_SYNTHESIS = "partial_synthesis"
    CONFIDENCE_UPDATE = "confidence_update"
    KNOWLEDGE_GAP_IDENTIFIED = "knowledge_gap_identified"
    
    # Planning and strategy events
    PLAN_CONSIDERATION = "plan_consideration"
    SEARCH_STRATEGY = "search_strategy"
    STEP_GENERATED = "step_generated"
    PLAN_QUALITY_ASSESSMENT = "plan_quality_assessment"
    PLAN_REVISION = "plan_revision"
    
    # Investigation and discovery events
    INVESTIGATION_START = "investigation_start"
    SOURCE_DISCOVERED = "source_discovered"
    CONTEXT_ESTABLISHED = "context_established"
    
    # Verification and fact-checking events
    CLAIM_IDENTIFIED = "claim_identified"
    VERIFICATION_ATTEMPT = "verification_attempt"
    CONTRADICTION_FOUND = "contradiction_found"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    
    # Synthesis and reporting events
    SYNTHESIS_STRATEGY = "synthesis_strategy"
    SECTION_GENERATION = "section_generation"
    CITATION_LINKING = "citation_linking"


class EventCategory(str, Enum):
    """Categories of events for UI organization and visualization."""
    SEARCH = "search"              # External searches and queries
    REFLECTION = "reflection"      # Internal reasoning and decision-making
    ANALYSIS = "analysis"          # Data processing and evaluation
    SYNTHESIS = "synthesis"        # Conclusion forming and report generation
    PLANNING = "planning"          # Strategy and approach decisions
    VERIFICATION = "verification"  # Fact-checking and validation
    COORDINATION = "coordination"  # Agent handoffs and workflow transitions
    ERROR = "error"               # Errors and recovery attempts


class ReasoningVisibility(str, Enum):
    """Control levels for reasoning visibility."""
    HIDDEN = "hidden"
    SUMMARIZED = "summarized"
    RAW = "raw"


@dataclass
class SearchResult:
    """Represents a search result from any source."""
    content: str
    source: Optional[str] = None  # Made optional with default None for flexibility
    url: Optional[str] = None
    title: Optional[str] = None
    score: float = 0.0
    relevance_score: float = 0.0  # Alias for tests
    published_date: Optional[str] = None
    result_type: SearchResultType = SearchResultType.WEB
    source_type: SearchResultType = SearchResultType.WEB  # Alias for tests
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # MEMORY OPTIMIZATION: Increased limit for comprehensive research
        MAX_CONTENT_LENGTH = 50000  # ~50KB max per result for detailed content
        if len(self.content) > MAX_CONTENT_LENGTH:
            self.content = self.content[:MAX_CONTENT_LENGTH] + "...[truncated for memory efficiency]"
        
        # Keep aliases in sync
        if self.relevance_score == 0.0 and self.score:
            self.relevance_score = self.score
        if self.score == 0.0 and self.relevance_score:
            self.score = self.relevance_score

        # Mirror result_type and source_type
        if self.source_type and self.result_type != self.source_type:
            self.result_type = self.source_type


@dataclass
class Citation:
    """Represents a citation for a piece of information."""
    source: str
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Citation to dictionary for serialization."""
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score
        }


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
    
    # NEW: Enhanced fields for rich UI events
    category: Optional[EventCategory] = None  # Event category for UI organization
    title: Optional[str] = None  # Human-readable title
    description: Optional[str] = None  # Detailed description of what's happening
    confidence: Optional[float] = None  # Confidence score (0.0-1.0)
    reasoning: Optional[str] = None  # Explanation of why this action was taken
    alternatives_considered: List[str] = field(default_factory=list)  # Other options considered
    related_event_ids: List[str] = field(default_factory=list)  # IDs of related events
    priority: int = 0  # Priority for UI ordering (higher = more important)
    
    def set_category_from_event_type(self) -> None:
        """Set category based on event type if not already set."""
        if self.category is not None:
            return
            
        # Map event types to categories
        category_mapping = {
            # Search events
            IntermediateEventType.QUERY_GENERATED: EventCategory.SEARCH,
            IntermediateEventType.QUERY_EXECUTING: EventCategory.SEARCH,
            IntermediateEventType.SEARCH_RESULTS_FOUND: EventCategory.SEARCH,
            IntermediateEventType.TOOL_CALL_START: EventCategory.SEARCH,
            IntermediateEventType.TOOL_CALL_COMPLETE: EventCategory.SEARCH,
            IntermediateEventType.SOURCE_DISCOVERED: EventCategory.SEARCH,
            IntermediateEventType.SEARCH_STRATEGY: EventCategory.SEARCH,
            
            # Reflection events
            IntermediateEventType.REASONING_REFLECTION: EventCategory.REFLECTION,
            IntermediateEventType.AGENT_THINKING: EventCategory.REFLECTION,
            IntermediateEventType.HYPOTHESIS_FORMED: EventCategory.REFLECTION,
            IntermediateEventType.CONFIDENCE_UPDATE: EventCategory.REFLECTION,
            IntermediateEventType.KNOWLEDGE_GAP_IDENTIFIED: EventCategory.REFLECTION,
            
            # Analysis events  
            IntermediateEventType.SOURCE_EVALUATION: EventCategory.ANALYSIS,
            IntermediateEventType.SOURCE_ANALYZED: EventCategory.ANALYSIS,
            IntermediateEventType.CONTEXT_ESTABLISHED: EventCategory.ANALYSIS,
            IntermediateEventType.QUALITY_ASSESSMENT: EventCategory.ANALYSIS,
            
            # Synthesis events
            IntermediateEventType.PARTIAL_SYNTHESIS: EventCategory.SYNTHESIS,
            IntermediateEventType.SYNTHESIS_PROGRESS: EventCategory.SYNTHESIS,
            IntermediateEventType.SYNTHESIS_STRATEGY: EventCategory.SYNTHESIS,
            IntermediateEventType.SECTION_GENERATION: EventCategory.SYNTHESIS,
            IntermediateEventType.CITATION_LINKING: EventCategory.SYNTHESIS,
            IntermediateEventType.REPORT_GENERATION: EventCategory.SYNTHESIS,
            
            # Planning events
            IntermediateEventType.PLAN_CONSIDERATION: EventCategory.PLANNING,
            IntermediateEventType.PLAN_CREATED: EventCategory.PLANNING,
            IntermediateEventType.PLAN_UPDATED: EventCategory.PLANNING,
            IntermediateEventType.STEP_GENERATED: EventCategory.PLANNING,
            IntermediateEventType.PLAN_QUALITY_ASSESSMENT: EventCategory.PLANNING,
            IntermediateEventType.PLAN_REVISION: EventCategory.PLANNING,
            IntermediateEventType.INVESTIGATION_START: EventCategory.PLANNING,
            
            # Verification events
            IntermediateEventType.CLAIM_IDENTIFIED: EventCategory.VERIFICATION,
            IntermediateEventType.VERIFICATION_ATTEMPT: EventCategory.VERIFICATION,
            IntermediateEventType.CONTRADICTION_FOUND: EventCategory.VERIFICATION,
            IntermediateEventType.GROUNDING_START: EventCategory.VERIFICATION,
            IntermediateEventType.GROUNDING_COMPLETE: EventCategory.VERIFICATION,
            IntermediateEventType.GROUNDING_CONTRADICTION: EventCategory.VERIFICATION,
            IntermediateEventType.CONFIDENCE_ADJUSTMENT: EventCategory.VERIFICATION,
            
            # Coordination events
            IntermediateEventType.AGENT_HANDOFF: EventCategory.COORDINATION,
            IntermediateEventType.AGENT_START: EventCategory.COORDINATION,
            IntermediateEventType.AGENT_COMPLETE: EventCategory.COORDINATION,
            IntermediateEventType.STAGE_TRANSITION: EventCategory.COORDINATION,
            
            # Error events
            IntermediateEventType.TOOL_CALL_ERROR: EventCategory.ERROR,
        }
        
        self.category = category_mapping.get(self.event_type, EventCategory.ANALYSIS)


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
    llm_endpoint: str = "databricks-gpt-oss-120b"
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