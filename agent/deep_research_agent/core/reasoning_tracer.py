"""
Reasoning tracer for capturing and emitting LLM thought snapshots.

This module provides utilities to capture LLM reasoning during agent execution
and emit thought snapshots at appropriate intervals.
"""

import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

from deep_research_agent.core import get_logger
from deep_research_agent.core.types import IntermediateEvent, IntermediateEventType, ReasoningVisibility
from deep_research_agent.core.redaction_utils import get_redactor

logger = get_logger(__name__)


@dataclass
class ThoughtBuffer:
    """Buffer for accumulating LLM thoughts."""
    content: str = ""
    token_count: int = 0
    last_snapshot_time: float = field(default_factory=time.time)
    last_snapshot_tokens: int = 0


class ReasoningTracer:
    """Tracer for capturing and emitting LLM reasoning snapshots."""
    
    def __init__(
        self,
        visibility: ReasoningVisibility = ReasoningVisibility.SUMMARIZED,
        token_interval: int = 40,
        time_interval_ms: int = 800,
        max_chars_per_step: int = 1000,
        redaction_patterns: Optional[List[str]] = None,
        event_emitter: Optional[Callable[[IntermediateEvent], None]] = None
    ):
        """
        Initialize reasoning tracer.
        
        Args:
            visibility: Level of reasoning visibility
            token_interval: Emit snapshot every N tokens
            time_interval_ms: Emit snapshot every N milliseconds
            max_chars_per_step: Maximum characters per thought snapshot
            redaction_patterns: Custom patterns for redacting sensitive info
            event_emitter: Function to emit intermediate events
        """
        self.visibility = visibility
        self.token_interval = token_interval
        self.time_interval_ms = time_interval_ms
        self.max_chars_per_step = max_chars_per_step
        self.event_emitter = event_emitter or (lambda x: None)
        
        self.redactor = get_redactor(redaction_patterns)
        self.thought_buffer = ThoughtBuffer()
        self.current_correlation_id: Optional[str] = None
        self.current_stage_id: Optional[str] = None
        self.sequence_counter = 0
        
        # State tracking
        self.active = False
        self.paused = False
    
    def start_step(self, correlation_id: str, stage_id: Optional[str] = None) -> None:
        """Start a new reasoning step."""
        self.current_correlation_id = correlation_id
        self.current_stage_id = stage_id
        self.thought_buffer = ThoughtBuffer()
        self.active = True
        self.paused = False
        
        logger.debug(f"Started reasoning trace for correlation_id: {correlation_id}")
    
    def end_step(self) -> None:
        """End the current reasoning step."""
        if self.active and self.thought_buffer.content.strip():
            # Emit final thought snapshot for this step
            self._emit_thought_snapshot(is_final=True)
        
        self.active = False
        self.current_correlation_id = None
        self.current_stage_id = None
        
        logger.debug("Ended reasoning trace step")
    
    def pause(self) -> None:
        """Pause thought collection temporarily."""
        self.paused = True
    
    def resume(self) -> None:
        """Resume thought collection."""
        self.paused = False
    
    def add_thought(self, content: str, token_count: Optional[int] = None) -> None:
        """Add content to the current thought buffer."""
        if not self.active or self.paused or self.visibility == ReasoningVisibility.HIDDEN:
            return
        
        if token_count is None:
            # Rough token estimation: ~4 characters per token
            token_count = len(content) // 4
        
        self.thought_buffer.content += content
        self.thought_buffer.token_count += token_count
        
        # Manage buffer size to prevent unlimited growth
        max_buffer_size = self.max_chars_per_step * 2  # Allow some buffer
        if len(self.thought_buffer.content) > max_buffer_size:
            # Keep the most recent content
            self.thought_buffer.content = self.thought_buffer.content[-max_buffer_size:]
        
        # Check if we should emit a snapshot
        self._check_snapshot_conditions()
    
    def add_reasoning_step(self, step_description: str, details: Optional[str] = None) -> None:
        """Add a structured reasoning step."""
        if not self.active or self.paused or self.visibility == ReasoningVisibility.HIDDEN:
            return
        
        formatted_step = f"\n[REASONING] {step_description}"
        if details:
            formatted_step += f": {details}"
        formatted_step += "\n"
        
        self.add_thought(formatted_step)
    
    def add_decision_point(self, decision: str, rationale: Optional[str] = None) -> None:
        """Add a decision point with rationale."""
        if not self.active or self.paused or self.visibility == ReasoningVisibility.HIDDEN:
            return
        
        formatted_decision = f"\n[DECISION] {decision}"
        if rationale:
            formatted_decision += f" | Rationale: {rationale}"
        formatted_decision += "\n"
        
        self.add_thought(formatted_decision)
    
    def _check_snapshot_conditions(self) -> None:
        """Check if conditions are met to emit a thought snapshot."""
        current_time = time.time()
        time_since_last = (current_time - self.thought_buffer.last_snapshot_time) * 1000
        tokens_since_last = self.thought_buffer.token_count - self.thought_buffer.last_snapshot_tokens
        
        should_emit = (
            tokens_since_last >= self.token_interval or
            time_since_last >= self.time_interval_ms
        )
        
        if should_emit and self.thought_buffer.content.strip():
            self._emit_thought_snapshot()
    
    def _emit_thought_snapshot(self, is_final: bool = False) -> None:
        """Emit a thought snapshot event."""
        if not self.current_correlation_id or not self.thought_buffer.content.strip():
            return
        
        # Process content based on visibility level
        content = self._process_content_for_visibility()
        
        if not content.strip():
            return
        
        # Create event data
        event_data = {
            "content": content,
            "is_final": is_final,
            "token_count": self.thought_buffer.token_count,
            "step_duration_ms": int((time.time() - self.thought_buffer.last_snapshot_time) * 1000)
        }
        
        # Redact sensitive information
        event_data = self.redactor.redact_event_data(event_data)
        
        # Create and emit event
        event = IntermediateEvent(
            timestamp=time.time(),
            stage_id=self.current_stage_id,
            correlation_id=self.current_correlation_id,
            sequence=self._get_next_sequence(),
            event_type=IntermediateEventType.THOUGHT_SNAPSHOT,
            data=event_data,
            meta={"visibility": self.visibility.value}
        )
        
        self.event_emitter(event)
        
        # Update tracking
        self.thought_buffer.last_snapshot_time = time.time()
        self.thought_buffer.last_snapshot_tokens = self.thought_buffer.token_count
        
        logger.debug(f"Emitted thought snapshot with {len(content)} characters")
    
    def _process_content_for_visibility(self) -> str:
        """Process thought content based on visibility level."""
        content = self.thought_buffer.content
        
        if self.visibility == ReasoningVisibility.HIDDEN:
            return ""
        elif self.visibility == ReasoningVisibility.RAW:
            # Return raw content but truncated
            return self.redactor.truncate_text(content, self.max_chars_per_step)
        elif self.visibility == ReasoningVisibility.SUMMARIZED:
            # Summarize the content
            return self._summarize_thought_content(content)
        
        return content
    
    def _summarize_thought_content(self, content: str) -> str:
        """Create a summarized version of thought content."""
        if len(content) <= self.max_chars_per_step:
            return content
        
        # Extract key reasoning steps and decisions
        lines = content.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Prioritize reasoning steps, decisions, and important markers
            if any(marker in line for marker in ['[REASONING]', '[DECISION]', '[IMPORTANT]', 'CONCLUSION', 'NEXT:']):
                key_lines.append(line)
            elif len(key_lines) < 3:  # Include first few regular lines
                key_lines.append(line)
        
        summary = '\n'.join(key_lines)
        
        # If still too long, truncate
        if len(summary) > self.max_chars_per_step:
            summary = self.redactor.truncate_text(summary, self.max_chars_per_step)
        
        return summary
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number for event ordering."""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def set_event_emitter(self, emitter: Callable[[IntermediateEvent], None]) -> None:
        """Set the event emitter function."""
        self.event_emitter = emitter
    
    def get_current_thought_summary(self) -> Optional[str]:
        """Get a summary of current thoughts without emitting an event."""
        if not self.active or not self.thought_buffer.content.strip():
            return None
        
        return self._process_content_for_visibility()


# Global reasoning tracer instance
_global_tracer: Optional[ReasoningTracer] = None


def get_reasoning_tracer() -> ReasoningTracer:
    """Get the global reasoning tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ReasoningTracer()
    return _global_tracer


def initialize_reasoning_tracer(
    visibility: ReasoningVisibility = ReasoningVisibility.SUMMARIZED,
    token_interval: int = 40,
    time_interval_ms: int = 800,
    max_chars_per_step: int = 1000,
    redaction_patterns: Optional[List[str]] = None,
    event_emitter: Optional[Callable[[IntermediateEvent], None]] = None
) -> ReasoningTracer:
    """Initialize the global reasoning tracer with configuration."""
    global _global_tracer
    _global_tracer = ReasoningTracer(
        visibility=visibility,
        token_interval=token_interval,
        time_interval_ms=time_interval_ms,
        max_chars_per_step=max_chars_per_step,
        redaction_patterns=redaction_patterns,
        event_emitter=event_emitter
    )
    return _global_tracer
