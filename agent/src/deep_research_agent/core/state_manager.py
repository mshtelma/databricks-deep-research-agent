"""
Immutable state management system with proper encapsulation.

This module provides type-safe, immutable state management to replace
dictionary-based state handling and prevent bugs from unintended mutations.
"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Union, TypeVar
from datetime import datetime
from enum import Enum
import uuid
import time
import json
from copy import deepcopy

from .search_provider import SearchResult
from .error_handler import safe_call

T = TypeVar('T', bound='ImmutableState')


class WorkflowNodeType(Enum):
    """Enumeration of workflow node types."""
    INIT = "init"
    QUERY_GENERATION = "query_generation"
    BATCH_CONTROLLER = "batch_controller"
    WEB_SEARCH = "web_search"
    VECTOR_SEARCH = "vector_search"
    RESULT_AGGREGATION = "result_aggregation"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


class ResearchPhase(Enum):
    """Enumeration of research phases."""
    INITIALIZATION = "initialization"
    QUERY_PLANNING = "query_planning"
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    FINALIZATION = "finalization"


@dataclass(frozen=True)
class Citation:
    """Immutable citation data structure."""
    
    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.0
    retrieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure immutable metadata."""
        if self.metadata is not None:
            object.__setattr__(self, 'metadata', deepcopy(dict(self.metadata)))
        else:
            object.__setattr__(self, 'metadata', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary format."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "metadata": dict(self.metadata)
        }
    
    @classmethod
    def from_search_result(cls, search_result: SearchResult) -> 'Citation':
        """Create citation from search result."""
        return cls(
            title=search_result.title,
            url=search_result.url,
            snippet=search_result.content[:500] + "..." if len(search_result.content) > 500 else search_result.content,
            source=search_result.source or "web",
            score=search_result.score,
            retrieved_at=datetime.now(),
            metadata=dict(search_result.metadata)
        )


@dataclass(frozen=True)
class ResearchMetrics:
    """Immutable metrics for research operations."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Search metrics
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    search_duration: float = 0.0
    
    # Result metrics
    total_results: int = 0
    unique_results: int = 0
    citation_count: int = 0
    
    # Quality metrics
    coverage_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    
    # Performance metrics
    llm_calls: int = 0
    llm_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Error tracking
    error_count: int = 0
    retry_count: int = 0
    
    def get_duration(self) -> float:
        """Get total research duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_success_rate(self) -> float:
        """Calculate search success rate."""
        total = self.total_searches
        return (self.successful_searches / total) if total > 0 else 1.0
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score based on multiple factors."""
        duration = self.get_duration()
        if duration == 0:
            return 1.0
        
        # Factors: results per second, success rate, cache hit rate
        results_per_sec = self.total_results / duration
        success_rate = self.get_success_rate()
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total) if cache_total > 0 else 0.0
        
        return min(1.0, (results_per_sec * 0.4 + success_rate * 0.4 + cache_hit_rate * 0.2))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "total_searches": self.total_searches,
            "successful_searches": self.successful_searches,
            "failed_searches": self.failed_searches,
            "search_duration": self.search_duration,
            "total_results": self.total_results,
            "unique_results": self.unique_results,
            "citation_count": self.citation_count,
            "coverage_score": self.coverage_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "llm_calls": self.llm_calls,
            "llm_duration": self.llm_duration,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": self.get_success_rate(),
            "efficiency_score": self.get_efficiency_score()
        }


@dataclass(frozen=True)
class ImmutableState:
    """Base class for immutable state objects."""
    
    def with_update(self: T, **kwargs) -> T:
        """Create new state instance with updates (immutable pattern)."""
        return replace(self, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        result = {}
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if hasattr(value, 'to_dict'):
                result[field_info.name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[field_info.name] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item 
                    for item in value
                ]
            elif isinstance(value, dict):
                result[field_info.name] = dict(value)
            elif isinstance(value, Enum):
                result[field_info.name] = value.value
            elif isinstance(value, datetime):
                result[field_info.name] = value.isoformat()
            else:
                result[field_info.name] = value
        return result


@dataclass(frozen=True)
class ResearchState(ImmutableState):
    """
    Immutable research state with proper encapsulation.
    
    This replaces the dictionary-based state management with a type-safe,
    immutable approach that prevents bugs from unintended mutations.
    """
    
    # Identity and timing
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Research configuration
    query: str = ""
    max_iterations: int = 3
    current_iteration: int = 0
    
    # Workflow state
    current_node: WorkflowNodeType = WorkflowNodeType.INIT
    current_phase: ResearchPhase = ResearchPhase.INITIALIZATION
    completed_nodes: List[WorkflowNodeType] = field(default_factory=list)
    
    # Research data
    search_queries: List[str] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    
    # Generated content
    generated_content: str = ""
    intermediate_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Research context
    research_focus: str = ""
    coverage_areas: List[str] = field(default_factory=list)
    quality_threshold: float = 0.7
    
    # Metrics and tracking
    metrics: ResearchMetrics = field(default_factory=ResearchMetrics)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure immutable collections."""
        # Deep copy mutable collections to prevent external mutation
        if self.completed_nodes is not None:
            object.__setattr__(self, 'completed_nodes', list(self.completed_nodes))
        
        if self.search_queries is not None:
            object.__setattr__(self, 'search_queries', list(self.search_queries))
        
        if self.search_results is not None:
            object.__setattr__(self, 'search_results', list(self.search_results))
        
        if self.citations is not None:
            object.__setattr__(self, 'citations', list(self.citations))
        
        if self.coverage_areas is not None:
            object.__setattr__(self, 'coverage_areas', list(self.coverage_areas))
        
        if self.intermediate_outputs is not None:
            object.__setattr__(self, 'intermediate_outputs', dict(self.intermediate_outputs))
        
        if self.errors is not None:
            object.__setattr__(self, 'errors', list(self.errors))
        
        if self.warnings is not None:
            object.__setattr__(self, 'warnings', list(self.warnings))
    
    # =========================================================================
    # State Transition Methods
    # =========================================================================
    
    def advance_to_node(self, node: WorkflowNodeType) -> 'ResearchState':
        """Advance to next workflow node."""
        new_completed = self.completed_nodes + [self.current_node]
        return self.with_update(
            current_node=node,
            completed_nodes=new_completed
        )
    
    def advance_to_phase(self, phase: ResearchPhase) -> 'ResearchState':
        """Advance to next research phase."""
        return self.with_update(current_phase=phase)
    
    def increment_iteration(self) -> 'ResearchState':
        """Increment iteration counter."""
        return self.with_update(current_iteration=self.current_iteration + 1)
    
    def mark_complete(self) -> 'ResearchState':
        """Mark research as complete."""
        final_metrics = self.metrics.with_update(
            end_time=datetime.now()
        )
        
        return self.with_update(
            current_node=WorkflowNodeType.COMPLETE,
            current_phase=ResearchPhase.FINALIZATION,
            metrics=final_metrics
        )
    
    def add_error(self, error: str) -> 'ResearchState':
        """Add error message to state."""
        new_errors = self.errors + [error]
        new_metrics = self.metrics.with_update(
            error_count=self.metrics.error_count + 1
        )
        return self.with_update(errors=new_errors, metrics=new_metrics)
    
    def add_warning(self, warning: str) -> 'ResearchState':
        """Add warning message to state."""
        new_warnings = self.warnings + [warning]
        return self.with_update(warnings=new_warnings)
    
    # =========================================================================
    # Research Data Methods
    # =========================================================================
    
    def add_search_query(self, query: str) -> 'ResearchState':
        """Add search query to state."""
        if query not in self.search_queries:
            new_queries = self.search_queries + [query]
            return self.with_update(search_queries=new_queries)
        return self
    
    def add_search_queries(self, queries: List[str]) -> 'ResearchState':
        """Add multiple search queries to state."""
        new_queries = self.search_queries + [q for q in queries if q not in self.search_queries]
        return self.with_update(search_queries=new_queries)
    
    def add_search_result(self, result: SearchResult) -> 'ResearchState':
        """Add search result and return new state."""
        new_results = self.search_results + [result]
        
        # Update metrics
        new_metrics = self.metrics.with_update(
            total_results=len(new_results),
            successful_searches=self.metrics.successful_searches + 1
        )
        
        return self.with_update(
            search_results=new_results,
            metrics=new_metrics
        )
    
    def add_search_results(self, results: List[SearchResult]) -> 'ResearchState':
        """Add multiple search results to state."""
        if not results:
            return self
        
        new_results = self.search_results + results
        
        # Create citations from new results
        new_citations = self.citations + [
            Citation.from_search_result(result) for result in results
        ]
        
        # Update metrics
        new_metrics = self.metrics.with_update(
            total_results=len(new_results),
            citation_count=len(new_citations),
            successful_searches=self.metrics.successful_searches + 1
        )
        
        return self.with_update(
            search_results=new_results,
            citations=new_citations,
            metrics=new_metrics
        )
    
    def update_generated_content(self, content: str) -> 'ResearchState':
        """Update generated content."""
        return self.with_update(generated_content=content)
    
    def add_intermediate_output(self, key: str, value: Any) -> 'ResearchState':
        """Add intermediate output."""
        new_outputs = {**self.intermediate_outputs, key: value}
        return self.with_update(intermediate_outputs=new_outputs)
    
    # =========================================================================
    # Quality and Progress Assessment
    # =========================================================================
    
    def should_continue_research(self) -> bool:
        """Determine if research should continue."""
        # Check iteration limits
        if self.current_iteration >= self.max_iterations:
            return False
        
        # Check error threshold
        if self.metrics.error_count > 5:
            return False
        
        # Check if we have enough results
        if len(self.search_results) >= 20:
            return False
        
        # Check quality threshold
        if self.get_quality_score() >= self.quality_threshold:
            return False
        
        # Continue if none of the stop conditions are met
        return True
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.search_results:
            return 0.0
        
        # Base score on result count and diversity
        result_score = min(1.0, len(self.search_results) / 10.0)
        
        # Add diversity score (unique sources)
        unique_sources = len(set(result.source for result in self.search_results))
        diversity_score = min(1.0, unique_sources / 5.0)
        
        # Add coverage score (different query coverage)
        coverage_score = min(1.0, len(self.search_queries) / 3.0)
        
        # Weighted average
        return (result_score * 0.4 + diversity_score * 0.3 + coverage_score * 0.3)
    
    def get_progress_percentage(self) -> float:
        """Get research progress as percentage."""
        total_phases = len(ResearchPhase)
        current_phase_idx = list(ResearchPhase).index(self.current_phase)
        
        # Base progress on phase
        phase_progress = (current_phase_idx + 1) / total_phases
        
        # Adjust for iteration progress within phases
        if self.current_phase == ResearchPhase.INFORMATION_GATHERING:
            iteration_progress = self.current_iteration / self.max_iterations
            phase_weight = 0.5  # This phase is typically 50% of the work
            phase_progress = (current_phase_idx / total_phases) + (iteration_progress * phase_weight)
        
        return min(1.0, phase_progress) * 100.0
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            "id": self.id,
            "query": self.query,
            "current_node": self.current_node.value,
            "current_phase": self.current_phase.value,
            "iteration": f"{self.current_iteration}/{self.max_iterations}",
            "progress_percentage": self.get_progress_percentage(),
            "quality_score": self.get_quality_score(),
            "results_count": len(self.search_results),
            "citations_count": len(self.citations),
            "queries_generated": len(self.search_queries),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "duration": self.metrics.get_duration(),
            "should_continue": self.should_continue_research(),
            "metrics": self.metrics.to_dict()
        }
    
    # =========================================================================
    # Serialization and Compatibility
    # =========================================================================
    
    @safe_call("to_langraph_dict", fallback={})
    def to_langraph_dict(self) -> Dict[str, Any]:
        """
        Convert to LangGraph-compatible dictionary format.
        
        This maintains compatibility with existing LangGraph workflows
        while providing the benefits of immutable state management.
        """
        # Extract core data for LangGraph compatibility
        langraph_state = {
            # Basic info
            "query": self.query,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            
            # Workflow tracking
            "current_node": self.current_node.value,
            "completed_nodes": [node.value for node in self.completed_nodes],
            
            # Research data
            "search_queries": list(self.search_queries),
            "search_results": [result.to_dict() for result in self.search_results],
            "citations": [citation.to_dict() for citation in self.citations],
            "generated_content": self.generated_content,
            
            # Metrics (flattened for LangGraph)
            "metrics": self.metrics.to_dict(),
            
            # Status
            "should_continue": self.should_continue_research(),
            "quality_score": self.get_quality_score(),
            
            # Error tracking
            "errors": list(self.errors),
            "warnings": list(self.warnings)
        }
        
        # Add intermediate outputs
        langraph_state.update(self.intermediate_outputs)
        
        return langraph_state
    
    @classmethod
    def from_langraph_dict(cls, data: Dict[str, Any]) -> 'ResearchState':
        """Create ResearchState from LangGraph dictionary."""
        try:
            # Parse enum values
            current_node = WorkflowNodeType(data.get("current_node", "init"))
            completed_nodes = [
                WorkflowNodeType(node) for node in data.get("completed_nodes", [])
            ]
            
            # Parse search results
            search_results = []
            for result_data in data.get("search_results", []):
                if isinstance(result_data, dict):
                    search_results.append(SearchResult.from_dict(result_data))
            
            # Parse citations
            citations = []
            for citation_data in data.get("citations", []):
                if isinstance(citation_data, dict):
                    citations.append(Citation(**citation_data))
            
            # Create metrics
            metrics_data = data.get("metrics", {})
            if isinstance(metrics_data, dict):
                metrics = ResearchMetrics(**metrics_data)
            else:
                metrics = ResearchMetrics()
            
            return cls(
                query=data.get("query", ""),
                current_iteration=data.get("current_iteration", 0),
                max_iterations=data.get("max_iterations", 3),
                current_node=current_node,
                completed_nodes=completed_nodes,
                search_queries=data.get("search_queries", []),
                search_results=search_results,
                citations=citations,
                generated_content=data.get("generated_content", ""),
                metrics=metrics,
                errors=data.get("errors", []),
                warnings=data.get("warnings", []),
                # Extract intermediate outputs (everything not in core fields)
                intermediate_outputs={
                    k: v for k, v in data.items() 
                    if k not in cls.__dataclass_fields__
                }
            )
            
        except Exception as e:
            # Return default state on parsing errors
            return cls(query=data.get("query", ""))


class StateManager:
    """
    Manager for research state transitions with validation and history.
    
    This provides a controlled interface for state transitions while
    maintaining immutability and providing transition validation.
    """
    
    def __init__(self, initial_state: Optional[ResearchState] = None):
        self.current_state = initial_state or ResearchState()
        self.state_history: List[ResearchState] = [self.current_state]
        self.transition_log: List[Dict[str, Any]] = []
    
    def transition_to(self, new_state: ResearchState, reason: str = "") -> bool:
        """
        Safely transition to new state with validation.
        
        Args:
            new_state: Target state
            reason: Reason for transition (for logging)
            
        Returns:
            True if transition was successful
        """
        try:
            # Validate transition
            if not self._validate_transition(self.current_state, new_state):
                return False
            
            # Record transition
            transition = {
                "from_state": self.current_state.id,
                "to_state": new_state.id,
                "from_node": self.current_state.current_node.value,
                "to_node": new_state.current_node.value,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update state
            self.current_state = new_state
            self.state_history.append(new_state)
            self.transition_log.append(transition)
            
            return True
            
        except Exception:
            return False
    
    def _validate_transition(self, from_state: ResearchState, to_state: ResearchState) -> bool:
        """Validate state transition is legal."""
        # Basic validation rules
        if to_state.current_iteration < from_state.current_iteration:
            return False  # Can't go backwards in iterations
        
        if len(to_state.errors) > 10:
            return False  # Too many errors
        
        return True
    
    def get_current_state(self) -> ResearchState:
        """Get current immutable state."""
        return self.current_state
    
    def get_state_history(self) -> List[ResearchState]:
        """Get history of all states."""
        return list(self.state_history)
    
    def get_transition_log(self) -> List[Dict[str, Any]]:
        """Get log of all state transitions."""
        return list(self.transition_log)
    
    def rollback_to_previous_state(self) -> bool:
        """Rollback to previous state if available."""
        if len(self.state_history) > 1:
            # Remove current state
            self.state_history.pop()
            self.current_state = self.state_history[-1]
            
            # Log rollback
            self.transition_log.append({
                "action": "rollback",
                "to_state": self.current_state.id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        return False