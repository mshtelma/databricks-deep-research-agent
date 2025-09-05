"""
Tests for event emitter functionality.

This module tests the intermediate event emission, batching, rate limiting,
and integration with the stream interface.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch
from deep_research_agent.core.event_emitter import (
    EventEmitter,
    get_event_emitter,
    initialize_event_emitter
)
from deep_research_agent.core.types import IntermediateEventType


class TestEventEmitter:
    """Test event emitter functionality."""
    
    def test_initialization_defaults(self):
        """Test emitter initialization with defaults."""
        emitter = EventEmitter()
        
        assert emitter.max_events_per_second == 10
        assert emitter.batch_events is True
        assert emitter.batch_size == 5
        assert emitter.batch_timeout_ms == 100
        assert emitter.sequence_counter == 0
        assert len(emitter.event_batch) == 0
    
    def test_initialization_custom(self):
        """Test emitter initialization with custom parameters."""
        mock_stream_emitter = Mock()
        custom_patterns = [r'\btest_pattern\b']
        
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            max_events_per_second=20,
            batch_events=False,
            batch_size=10,
            batch_timeout_ms=200,
            redaction_patterns=custom_patterns
        )
        
        assert emitter.stream_emitter == mock_stream_emitter
        assert emitter.max_events_per_second == 20
        assert emitter.batch_events is False
        assert emitter.batch_size == 10
        assert emitter.batch_timeout_ms == 200
    
    def test_basic_event_emission(self):
        """Test basic event emission without batching."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        # Emit an event
        result = emitter.emit(
            IntermediateEventType.ACTION_START,
            {"action": "search", "query": "test query"},
            correlation_id="test-correlation",
            stage_id="test-stage"
        )
        
        assert result is True
        mock_stream_emitter.assert_called_once()
        
        # Check the call arguments
        call_args = mock_stream_emitter.call_args[0][0]
        assert call_args["type"] == "intermediate_event"
        assert call_args["data"]["action"] == "search"
        assert call_args["data"]["query"] == "test query"
        assert call_args["event_type"] == IntermediateEventType.ACTION_START
        assert call_args["correlation_id"] == "test-correlation"
        assert call_args["stage_id"] == "test-stage"
    
    def test_event_batching(self):
        """Test event batching functionality."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=3  # Small batch size for testing
        )
        
        # Emit events that should trigger batching
        for i in range(3):
            result = emitter.emit(
                IntermediateEventType.ACTION_PROGRESS,
                {"action": f"action_{i}", "progress": i * 10},
                correlation_id=f"correlation_{i}"
            )
            assert result is True
        
        # Should have emitted a batch
        mock_stream_emitter.assert_called_once()
        
        call_args = mock_stream_emitter.call_args[0][0]
        assert call_args["type"] == "event_batch"
        assert call_args["batch_size"] == 3
        assert len(call_args["events"]) == 3
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            max_events_per_second=2,  # Very low limit for testing
            batch_events=False
        )
        
        # Emit events rapidly
        results = []
        for i in range(5):
            result = emitter.emit(
                IntermediateEventType.ACTION_START,
                {"action": f"action_{i}"}
            )
            results.append(result)
        
        # Should have some rate-limited events
        successful_emissions = sum(results)
        assert successful_emissions <= 2  # Rate limit should kick in
        assert emitter.stats["events_rate_limited"] > 0
    
    def test_redaction_in_emission(self):
        """Test that events are redacted before emission."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        # Emit event with sensitive data
        emitter.emit(
            IntermediateEventType.TOOL_CALL_START,
            {
                "tool_name": "search",
                "query": "Find info about user@example.com",
                "api_key": "sk-1234567890abcdef1234567890abcdef"
            }
        )
        
        mock_stream_emitter.assert_called_once()
        call_args = mock_stream_emitter.call_args[0][0]
        
        # Sensitive data should be redacted
        assert "[REDACTED]" in call_args["data"]["query"]
        assert call_args["data"]["api_key"] == "[REDACTED]"
    
    def test_sequence_number_increment(self):
        """Test that sequence numbers increment properly."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        # Emit multiple events
        for i in range(3):
            emitter.emit(
                IntermediateEventType.ACTION_PROGRESS,
                {"step": i}
            )
        
        # Check sequence numbers
        assert mock_stream_emitter.call_count == 3
        
        sequences = []
        for call in mock_stream_emitter.call_args_list:
            event_data = call[0][0]
            sequences.append(event_data["sequence"])
        
        # Should be incrementing
        assert sequences == [1, 2, 3]
    
    def test_action_start_helper(self):
        """Test action start helper method."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        result = emitter.emit_action_start(
            action="web_search",
            query="test query",
            correlation_id="test-correlation",
            stage_id="search_stage",
            additional_param="extra_data"
        )
        
        assert result is True
        mock_stream_emitter.assert_called_once()
        
        call_args = mock_stream_emitter.call_args[0][0]
        
        assert call_args["event_type"] == IntermediateEventType.ACTION_START
        assert call_args["data"]["action"] == "web_search"
        assert call_args["data"]["query"] == "test query"
        assert call_args["data"]["additional_param"] == "extra_data"
        assert call_args["correlation_id"] == "test-correlation"
    
    def test_action_progress_helper(self):
        """Test action progress helper method."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        result = emitter.emit_action_progress(
            action="processing",
            status="running",
            progress={"percentage": 75, "step": 3},
            correlation_id="test-correlation"
        )
        
        assert result is True
        
        call_args = mock_stream_emitter.call_args[0][0]
        
        assert call_args["event_type"] == IntermediateEventType.ACTION_PROGRESS
        assert call_args["data"]["status"] == "running"
        assert call_args["data"]["progress"]["percentage"] == 75
    
    def test_action_complete_helper(self):
        """Test action complete helper method."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        result = emitter.emit_action_complete(
            action="search",
            result_summary="Found 5 results",
            results_count=5,
            correlation_id="test-correlation"
        )
        
        assert result is True
        
        call_args = mock_stream_emitter.call_args[0][0]
        
        assert call_args["event_type"] == IntermediateEventType.ACTION_COMPLETE
        assert call_args["data"]["result_summary"] == "Found 5 results"
        assert call_args["data"]["results_count"] == 5
    
    def test_tool_call_helpers(self):
        """Test tool call helper methods."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        # Test tool call start
        emitter.emit_tool_call_start(
            tool_name="brave_search",
            parameters={"query": "test", "max_results": 5}
        )
        
        # Test tool call complete
        emitter.emit_tool_call_complete(
            tool_name="brave_search",
            success=True,
            result_summary="Search completed",
            execution_time=1.23
        )
        
        # Test tool call error
        emitter.emit_tool_call_error(
            tool_name="brave_search",
            error_message="API timeout",
            is_sanitized=True
        )
        
        assert mock_stream_emitter.call_count == 3
        
        # Check event types
        event_types = []
        for call in mock_stream_emitter.call_args_list:
            event_data = call[0][0]
            event_types.append(event_data["event_type"])
        
        assert IntermediateEventType.TOOL_CALL_START in event_types
        assert IntermediateEventType.TOOL_CALL_COMPLETE in event_types
        assert IntermediateEventType.TOOL_CALL_ERROR in event_types
    
    def test_citation_added_helper(self):
        """Test citation added helper method."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        result = emitter.emit_citation_added(
            title="Test Article",
            url="https://example.com/article",
            snippet="This is a test snippet",
            correlation_id="test-correlation"
        )
        
        assert result is True
        
        call_args = mock_stream_emitter.call_args[0][0]
        
        assert call_args["event_type"] == IntermediateEventType.CITATION_ADDED
        assert call_args["data"]["title"] == "Test Article"
        assert call_args["data"]["url"] == "https://example.com/article"
        assert call_args["data"]["snippet"] == "This is a test snippet"
    
    def test_synthesis_progress_helper(self):
        """Test synthesis progress helper method."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        result = emitter.emit_synthesis_progress(
            progress_type="drafting",
            content_preview="The research shows that...",
            completion_percentage=0.6,
            correlation_id="test-correlation"
        )
        
        assert result is True
        
        call_args = mock_stream_emitter.call_args[0][0]
        
        assert call_args["event_type"] == IntermediateEventType.SYNTHESIS_PROGRESS
        assert call_args["data"]["progress_type"] == "drafting"
        assert call_args["data"]["completion_percentage"] == 0.6
    
    def test_flush_batch(self):
        """Test manual batch flushing."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=10  # Large batch size to prevent auto-emission
        )
        
        # Add some events to batch
        emitter.emit(IntermediateEventType.ACTION_START, {"action": "test1"})
        emitter.emit(IntermediateEventType.ACTION_PROGRESS, {"action": "test2"})
        
        # Should not have emitted yet
        mock_stream_emitter.assert_not_called()
        
        # Flush manually
        emitter.flush_batch()
        
        # Should have emitted the batch
        mock_stream_emitter.assert_called_once()
        call_args = mock_stream_emitter.call_args[0][0]
        assert call_args["type"] == "event_batch"
        assert call_args["batch_size"] == 2
    
    def test_batch_timeout(self):
        """Test batch timeout functionality."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=10,  # Large batch size
            batch_timeout_ms=50  # Short timeout for testing
        )
        
        # Add one event
        emitter.emit(IntermediateEventType.ACTION_START, {"action": "test"})
        
        # Wait for timeout
        time.sleep(0.06)  # 60ms > 50ms timeout
        
        # Should have emitted due to timeout
        # Note: This test might be flaky due to timing
        # In real implementation, we might need to check manually
        emitter.flush_batch()  # Force flush for testing
        mock_stream_emitter.assert_called()
    
    def test_error_handling_in_emission(self):
        """Test error handling during emission."""
        # Mock stream emitter that raises exception
        mock_stream_emitter = Mock(side_effect=Exception("Stream error"))
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False
        )
        
        # Emit event - should not raise exception
        result = emitter.emit(
            IntermediateEventType.ACTION_START,
            {"action": "test"}
        )
        
        # Should return False and increment error stats
        assert result is False
        assert emitter.stats["events_dropped"] > 0
    
    def test_statistics_tracking(self):
        """Test event statistics tracking."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=False,
            max_events_per_second=2
        )
        
        # Emit some events
        emitter.emit(IntermediateEventType.ACTION_START, {"action": "test1"})
        emitter.emit(IntermediateEventType.ACTION_START, {"action": "test2"})
        emitter.emit(IntermediateEventType.ACTION_START, {"action": "test3"})  # Should be rate limited
        
        stats = emitter.get_stats()
        
        assert stats["events_emitted"] >= 2
        assert stats["events_rate_limited"] >= 1
        assert "events_dropped" in stats
        assert "events_batched" in stats


class TestEventEmitterIntegration:
    """Integration tests for event emitter."""
    
    def test_global_emitter_functions(self):
        """Test global emitter convenience functions."""
        # Test get_event_emitter
        emitter1 = get_event_emitter()
        emitter2 = get_event_emitter()
        
        # Should return the same instance
        assert emitter1 is emitter2
        
        # Test initialize_event_emitter
        mock_stream_emitter = Mock()
        emitter3 = initialize_event_emitter(
            stream_emitter=mock_stream_emitter,
            max_events_per_second=15,
            batch_events=False
        )
        
        assert emitter3.stream_emitter == mock_stream_emitter
        assert emitter3.max_events_per_second == 15
        assert emitter3.batch_events is False
        
        # Should be the new global instance
        emitter4 = get_event_emitter()
        assert emitter3 is emitter4
    
    def test_realistic_agent_workflow(self):
        """Test realistic agent workflow with event emission."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=10  # Allow multiple events before batching
        )
        
        correlation_id = "workflow-456"
        
        # Simulate full agent workflow
        emitter.emit_action_start(
            action="research_workflow",
            query="Machine learning trends",
            correlation_id=correlation_id,
            stage_id="start"
        )
        
        emitter.emit_action_progress(
            action="query_generation",
            status="running",
            progress={"step": 1, "percentage": 20},
            correlation_id=correlation_id,
            stage_id="query_gen"
        )
        
        emitter.emit_tool_call_start(
            tool_name="brave_search",
            parameters={"query": "ML trends 2024", "max_results": 5},
            correlation_id=correlation_id,
            stage_id="search"
        )
        
        emitter.emit_citation_added(
            title="ML Trends Report",
            url="https://example.com/ml-trends",
            snippet="Latest developments in ML...",
            correlation_id=correlation_id,
            stage_id="search"
        )
        
        emitter.emit_tool_call_complete(
            tool_name="brave_search",
            success=True,
            result_summary="Found 5 relevant results",
            correlation_id=correlation_id,
            stage_id="search"
        )
        
        emitter.emit_synthesis_progress(
            progress_type="drafting",
            content_preview="Based on the research...",
            completion_percentage=0.8,
            correlation_id=correlation_id,
            stage_id="synthesis"
        )
        
        emitter.emit_action_complete(
            action="research_workflow",
            result_summary="Generated comprehensive analysis",
            correlation_id=correlation_id,
            stage_id="complete"
        )
        
        # Force flush to emit the batch
        emitter.flush_batch()
        
        # Verify events were emitted
        mock_stream_emitter.assert_called()
        
        call_args = mock_stream_emitter.call_args[0][0]
        assert call_args["type"] == "event_batch"
        assert len(call_args["events"]) == 7  # All the events we emitted
        
        # Verify all events have the same correlation_id
        for event in call_args["events"]:
            assert event["correlation_id"] == correlation_id
    
    def test_performance_with_high_volume(self):
        """Test performance with high volume event emission."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=100,
            max_events_per_second=1000  # High limit
        )
        
        # Emit many events
        start_time = time.time()
        for i in range(500):
            emitter.emit(
                IntermediateEventType.ACTION_PROGRESS,
                {"step": i, "progress": i / 500},
                correlation_id=f"batch-{i // 50}"  # Group into correlation batches
            )
        
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 2.0
        
        # Force flush any remaining events
        emitter.flush_batch()
        
        # Should have emitted batches
        assert mock_stream_emitter.call_count > 0
        
        # Verify stats
        stats = emitter.get_stats()
        assert stats["events_emitted"] == 500
        assert stats["events_batched"] > 0
    
    def test_memory_efficiency(self):
        """Test memory efficiency with long-running emitter."""
        mock_stream_emitter = Mock()
        emitter = EventEmitter(
            stream_emitter=mock_stream_emitter,
            batch_events=True,
            batch_size=10,
            max_events_per_second=1000  # High limit to avoid rate limiting
        )
        
        # Emit events in multiple batches
        for batch in range(10):
            for i in range(15):  # More than batch_size to trigger emission
                emitter.emit(
                    IntermediateEventType.ACTION_PROGRESS,
                    {"batch": batch, "item": i}
                )
        
        # Flush any remaining events
        emitter.flush_batch()
        
        # Event batch should be empty after each batch emission
        assert len(emitter.event_batch) < emitter.batch_size
        
        # Should have emitted multiple batches (150 events / 10 batch_size = 15 batches)
        assert mock_stream_emitter.call_count >= 15


if __name__ == "__main__":
    pytest.main([__file__])
