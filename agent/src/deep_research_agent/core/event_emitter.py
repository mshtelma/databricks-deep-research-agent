"""
Event emitter for intermediate agent events.

This module provides utilities for emitting intermediate events during agent execution,
including action events, tool calls, and reasoning snapshots.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import asdict
from collections import deque
from threading import Lock, RLock, Timer
from uuid import uuid4

from . import get_logger
from .types import IntermediateEvent, IntermediateEventType, ReasoningVisibility, EventCategory
from .redaction_utils import get_redactor
from .event_templates import EventTemplates

logger = get_logger(__name__)


class EventEmitter:
    """Emitter for intermediate agent events with batching and rate limiting."""

    IMPORTANT_EVENT_TYPES = {
        IntermediateEventType.PLAN_CREATED,
        IntermediateEventType.PLAN_UPDATED,
        IntermediateEventType.PLAN_STRUCTURE,
        IntermediateEventType.PLAN_STRUCTURE_VISUALIZE,
        IntermediateEventType.STEP_ACTIVATED,
        IntermediateEventType.STEP_COMPLETED,
        IntermediateEventType.SEARCH_STRATEGY,
        IntermediateEventType.REASONING_REFLECTION,
        IntermediateEventType.VERIFICATION_ATTEMPT,
    }
    IMPORTANT_EVENT_VALUE_SET = {event.value for event in IMPORTANT_EVENT_TYPES}

    IMPORTANT_EVENTS = {
        IntermediateEventType.PLAN_CREATED,
        IntermediateEventType.PLAN_UPDATED,
        IntermediateEventType.PLAN_STRUCTURE,
        IntermediateEventType.PLAN_STRUCTURE_VISUALIZE,
        IntermediateEventType.STEP_ACTIVATED,
        IntermediateEventType.STEP_COMPLETED,
        IntermediateEventType.SEARCH_STRATEGY,
        IntermediateEventType.REASONING_REFLECTION,
        IntermediateEventType.VERIFICATION_ATTEMPT,
    }

    def __init__(
        self,
        stream_emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_events_per_second: int = 10,
        batch_events: bool = True,
        batch_size: int = 5,
        batch_timeout_ms: int = 100,
        redaction_patterns: Optional[List[str]] = None
    ):
        """
        Initialize event emitter.
        
        Args:
            stream_emitter: Function to emit events to stream (e.g., UI)
            max_events_per_second: Rate limit for event emission
            batch_events: Whether to batch events for efficiency
            batch_size: Number of events per batch
            batch_timeout_ms: Maximum time to wait for batch
            redaction_patterns: Custom redaction patterns
        """
        self.stream_emitter = stream_emitter or (lambda x: None)
        self.max_events_per_second = max_events_per_second
        self.batch_events = batch_events
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        self.redactor = get_redactor(redaction_patterns)
        self.sequence_counter = 0
        self.sequence_lock = RLock()
        
        # Rate limiting
        self.emission_times = deque(maxlen=max_events_per_second)
        self.rate_limit_lock = Lock()
        
        # Batching
        self.event_batch = []
        self.batch_lock = RLock()
        self.batch_timer: Optional[Timer] = None
        
        # Statistics
        self.stats = {
            "events_emitted": 0,
            "events_batched": 0,
            "events_rate_limited": 0,
            "events_dropped": 0
        }
    
    def emit(
        self,
        event_type: IntermediateEventType,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        alternatives_considered: Optional[List[str]] = None,
        related_event_ids: Optional[List[str]] = None,
        category: Optional[EventCategory] = None,
        priority: Optional[int] = None
    ) -> bool:
        """
        Emit an intermediate event with rich UI metadata.
        
        Args:
            event_type: Type of event to emit
            data: Event data payload
            correlation_id: Correlation ID for grouping related events
            stage_id: Current stage/phase identifier
            meta: Additional metadata
            title: Human-readable title (auto-generated if None)
            description: Detailed description (auto-generated if None)
            confidence: Confidence score (0.0-1.0)
            reasoning: Explanation of why this action was taken
            alternatives_considered: Other options that were considered
            related_event_ids: IDs of related events
            category: Event category (auto-determined if None)
            priority: Priority for UI ordering (auto-determined if None)
        
        Returns:
            True if event was emitted, False if dropped due to rate limiting
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                self.stats["events_rate_limited"] += 1
                logger.debug("Event dropped due to rate limiting")
                return False
            
            # Redact sensitive information from data
            redacted_data = self.redactor.redact_event_data(data.copy())

            event_type_str = (
                event_type.value if isinstance(event_type, IntermediateEventType) else str(event_type)
            )
            if isinstance(event_type, IntermediateEventType):
                important_event = event_type in self.IMPORTANT_EVENT_TYPES
            else:
                important_event = event_type_str in self.IMPORTANT_EVENT_VALUE_SET
            data_keys = list(redacted_data.keys())[:10]
            meta_keys = list((meta or {}).keys())[:10]
            log_message = (
                f"Preparing to emit intermediate event: type={event_type_str}, "
                f"stage_id={stage_id}, correlation_id={correlation_id}, "
                f"data_keys={data_keys}, meta_keys={meta_keys}, batching={self.batch_events}"
            )
            if important_event:
                logger.info(log_message)
            else:
                logger.debug(log_message)
            
            # Create event with enhanced fields
            event = IntermediateEvent(
                timestamp=time.time(),
                stage_id=stage_id,
                correlation_id=correlation_id,
                sequence=self._get_next_sequence(),
                event_type=event_type,
                data=redacted_data,
                meta=meta or {},
                title=title or EventTemplates.get_title(event_type, redacted_data),
                description=description or EventTemplates.get_description(event_type, redacted_data),
                confidence=confidence,
                reasoning=reasoning,
                alternatives_considered=alternatives_considered or [],
                related_event_ids=related_event_ids or [],
                priority=priority or EventTemplates.get_priority_from_event_type(event_type)
            )
            
            # Set category automatically if not provided
            if category:
                event.category = category
            else:
                event.set_category_from_event_type()
            
            # Check if this is a step event requiring immediate flush
            is_step_event = (
                event_type in [IntermediateEventType.STEP_ACTIVATED, IntermediateEventType.STEP_COMPLETED] or
                event_type_str in ["step_activated", "step_completed"]
            )
            
            # Enhanced debug logging for all events
            logger.info(f"🔍 EVENT_TRACE: Emitting {event_type_str} event")
            logger.info(f"    - Event data: {data}")
            logger.info(f"    - Is step event: {is_step_event}")
            logger.info(f"    - Batching enabled: {self.batch_events}")
            logger.info(f"    - Stage ID: {stage_id}")
            logger.info(f"    - Correlation ID: {correlation_id}")
            
            # Emit directly or batch, but force immediate emission for step events
            if self.batch_events and not is_step_event:
                logger.info(f"📦 EVENT_TRACE: Adding {event_type_str} to batch queue")
                self._add_to_batch(event)
                self.stats["events_emitted"] += 1
                return True  # Batching itself doesn't fail, emission happens later
            else:
                # Force immediate emission for step events or when batching is disabled
                if is_step_event and self.batch_events:
                    logger.info(f"🚀 EVENT_TRACE: Forcing immediate emission for step event: {event_type}")
                
                logger.info(f"📡 EVENT_TRACE: Emitting {event_type_str} event immediately")
                success = self._emit_event(event)
                if success:
                    self.stats["events_emitted"] += 1
                    return True
                else:
                    return False
            
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
            self.stats["events_dropped"] += 1
            return False
    
    def emit_action_start(
        self,
        action: str,
        query: Optional[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Emit an action start event."""
        data = {
            "action": action,
            "query": query,
            **kwargs
        }
        return self.emit(
            IntermediateEventType.ACTION_START,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_action_progress(
        self,
        action: str,
        status: str,
        progress: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Emit an action progress event."""
        data = {
            "action": action,
            "status": status,
            "progress": progress or {},
            **kwargs
        }
        return self.emit(
            IntermediateEventType.ACTION_PROGRESS,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_action_complete(
        self,
        action: str,
        result_summary: Optional[str] = None,
        results_count: Optional[int] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Emit an action complete event."""
        data = {
            "action": action,
            "result_summary": result_summary,
            "results_count": results_count,
            **kwargs
        }
        return self.emit(
            IntermediateEventType.ACTION_COMPLETE,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_tool_call_start(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a tool call start event."""
        data = {
            "tool_name": tool_name,
            "parameters": parameters or {}
        }
        return self.emit(
            IntermediateEventType.TOOL_CALL_START,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_tool_call_complete(
        self,
        tool_name: str,
        success: bool = True,
        result_summary: Optional[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Emit a tool call complete event."""
        data = {
            "tool_name": tool_name,
            "success": success,
            "result_summary": result_summary,
            **kwargs
        }
        return self.emit(
            IntermediateEventType.TOOL_CALL_COMPLETE,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_tool_call_error(
        self,
        tool_name: str,
        error_message: str,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        is_sanitized: bool = True
    ) -> bool:
        """Emit a tool call error event."""
        data = {
            "tool_name": tool_name,
            "error_message": error_message,
            "is_sanitized": is_sanitized
        }
        return self.emit(
            IntermediateEventType.TOOL_CALL_ERROR,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_citation_added(
        self,
        title: str,
        url: str,
        snippet: Optional[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a citation added event."""
        data = {
            "title": title,
            "url": url,
            "snippet": snippet
        }
        return self.emit(
            IntermediateEventType.CITATION_ADDED,
            data,
            correlation_id,
            stage_id
        )
    
    def emit_synthesis_progress(
        self,
        progress_type: str,
        content_preview: Optional[str] = None,
        completion_percentage: Optional[float] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a synthesis progress event."""
        data = {
            "progress_type": progress_type,
            "content_preview": content_preview,
            "completion_percentage": completion_percentage
        }
        return self.emit(
            IntermediateEventType.SYNTHESIS_PROGRESS,
            data,
            correlation_id,
            stage_id
        )
    
    # NEW: Enhanced event emission methods for transparent UI
    
    def emit_reasoning_reflection(
        self,
        reasoning: str,
        options: List[str],
        confidence: float,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a reasoning reflection event."""
        return self.emit(
            IntermediateEventType.REASONING_REFLECTION,
            data={"options": options},
            confidence=confidence,
            reasoning=reasoning,
            alternatives_considered=options,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_hypothesis_formed(
        self,
        hypothesis: str,
        confidence: float,
        supporting_evidence: List[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a hypothesis formation event."""
        return self.emit(
            IntermediateEventType.HYPOTHESIS_FORMED,
            data={
                "hypothesis": hypothesis,
                "supporting_evidence": supporting_evidence or []
            },
            confidence=confidence,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_source_evaluation(
        self,
        title: str,
        url: str,
        relevance: float,
        reasoning: str,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a source evaluation event."""
        return self.emit(
            IntermediateEventType.SOURCE_EVALUATION,
            data={
                "title": title,
                "url": url,
                "relevance": relevance
            },
            reasoning=reasoning,
            confidence=relevance,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_partial_synthesis(
        self,
        conclusion: str,
        source_count: int,
        confidence: float,
        supporting_sources: List[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a partial synthesis event."""
        return self.emit(
            IntermediateEventType.PARTIAL_SYNTHESIS,
            data={
                "conclusion": conclusion,
                "source_count": source_count,
                "supporting_sources": supporting_sources or []
            },
            confidence=confidence,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_knowledge_gap_identified(
        self,
        topic: str,
        purpose: str,
        impact: str = "medium",
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a knowledge gap identification event."""
        return self.emit(
            IntermediateEventType.KNOWLEDGE_GAP_IDENTIFIED,
            data={
                "topic": topic,
                "purpose": purpose,
                "impact": impact
            },
            reasoning=f"Need more information about {topic} to {purpose}",
            priority=7 if impact == "high" else 5,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_search_strategy(
        self,
        query_count: int,
        focus_areas: List[str],
        approach: str,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a search strategy event."""
        return self.emit(
            IntermediateEventType.SEARCH_STRATEGY,
            data={
                "query_count": query_count,
                "focus_areas": focus_areas,
                "approach": approach
            },
            reasoning=f"Focusing on {', '.join(focus_areas)} with {approach} approach",
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_confidence_update(
        self,
        old_confidence: float,
        new_confidence: float,
        reason: str,
        evidence: str,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a confidence update event."""
        direction = "increased" if new_confidence > old_confidence else "decreased"
        return self.emit(
            IntermediateEventType.CONFIDENCE_UPDATE,
            data={
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "direction": direction,
                "evidence": evidence
            },
            confidence=new_confidence,
            reasoning=reason,
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_plan_consideration(
        self,
        approach: str,
        reasoning: str,
        alternatives: List[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a plan consideration event."""
        return self.emit(
            IntermediateEventType.PLAN_CONSIDERATION,
            data={"approach": approach},
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def emit_verification_attempt(
        self,
        claim: str,
        method: str,
        source: str,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> bool:
        """Emit a verification attempt event."""
        return self.emit(
            IntermediateEventType.VERIFICATION_ATTEMPT,
            data={
                "claim": claim,
                "method": method,
                "source": source
            },
            correlation_id=correlation_id,
            stage_id=stage_id
        )
    
    def flush_batch(self) -> None:
        """Force emit any pending batched events."""
        with self.batch_lock:
            if self.event_batch:
                self._emit_batch()
    
    def get_stats(self) -> Dict[str, int]:
        """Get emission statistics."""
        return self.stats.copy()
    
    def _check_rate_limit(self) -> bool:
        """Check if we can emit based on rate limiting."""
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove events older than 1 second
            while self.emission_times and current_time - self.emission_times[0] > 1.0:
                self.emission_times.popleft()
            
            # Check if we're under the rate limit
            if len(self.emission_times) >= self.max_events_per_second:
                return False
            
            self.emission_times.append(current_time)
            return True
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number for event ordering."""
        with self.sequence_lock:
            self.sequence_counter += 1
            return self.sequence_counter
    
    def _add_to_batch(self, event: IntermediateEvent) -> None:
        """Add event to batch for later emission."""
        with self.batch_lock:
            self.event_batch.append(event)
            
            # Check if batch is full
            if len(self.event_batch) >= self.batch_size:
                self._emit_batch()
            else:
                # Start/restart timer for batch timeout
                if self.batch_timer:
                    self.batch_timer.cancel()
                self.batch_timer = Timer(
                    self.batch_timeout_ms / 1000.0,
                    self._emit_batch
                )
                self.batch_timer.start()
    
    def _emit_batch(self) -> None:
        """Emit current batch of events."""
        with self.batch_lock:
            if not self.event_batch:
                return
            
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
            
            # Convert events to dictionaries for emission
            batch_data = {
                "type": "event_batch",
                "events": [asdict(event) for event in self.event_batch],
                "batch_size": len(self.event_batch),
                "timestamp": time.time()
            }
            
            try:
                self.stream_emitter(batch_data)
                self.stats["events_batched"] += len(self.event_batch)
                logger.debug(f"Emitted batch of {len(self.event_batch)} events")
            except Exception as e:
                logger.error(f"Failed to emit event batch: {e}")
                self.stats["events_dropped"] += len(self.event_batch)
            finally:
                self.event_batch.clear()
    
    def _emit_event(self, event: IntermediateEvent) -> bool:
        """Emit a single event immediately."""
        try:
            event_data = asdict(event)
            event_data["type"] = "intermediate_event"
            self.stream_emitter(event_data)
            logger.debug(f"Emitted {event.event_type} event")
            return True
        except Exception as e:
            logger.error(f"Failed to emit single event: {e}")
            self.stats["events_dropped"] += 1
            return False


# Global event emitter instance
_global_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter


def initialize_event_emitter(
    stream_emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_events_per_second: int = 10,
    batch_events: bool = True,
    batch_size: int = 5,
    batch_timeout_ms: int = 100,
    redaction_patterns: Optional[List[str]] = None
) -> EventEmitter:
    """Initialize the global event emitter with configuration."""
    global _global_emitter
    _global_emitter = EventEmitter(
        stream_emitter=stream_emitter,
        max_events_per_second=max_events_per_second,
        batch_events=batch_events,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        redaction_patterns=redaction_patterns
    )
    return _global_emitter
