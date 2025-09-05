"""
Tests for reasoning tracer functionality.

This module tests the LLM thought capture and emission functionality
for intermediate events.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from deep_research_agent.core.reasoning_tracer import (
    ReasoningTracer,
    ThoughtBuffer,
    get_reasoning_tracer,
    initialize_reasoning_tracer
)
from deep_research_agent.core.types import IntermediateEvent, IntermediateEventType, ReasoningVisibility


class TestThoughtBuffer:
    """Test thought buffer functionality."""
    
    def test_initialization(self):
        """Test thought buffer initialization."""
        buffer = ThoughtBuffer()
        
        assert buffer.content == ""
        assert buffer.token_count == 0
        assert buffer.last_snapshot_time > 0
        assert buffer.last_snapshot_tokens == 0


class TestReasoningTracer:
    """Test reasoning tracer functionality."""
    
    def test_initialization_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = ReasoningTracer()
        
        assert tracer.visibility == ReasoningVisibility.SUMMARIZED
        assert tracer.token_interval == 40
        assert tracer.time_interval_ms == 800
        assert tracer.max_chars_per_step == 1000
        assert not tracer.active
        assert not tracer.paused
        assert tracer.sequence_counter == 0
    
    def test_initialization_custom(self):
        """Test tracer initialization with custom parameters."""
        mock_emitter = Mock()
        custom_patterns = [r'\btest_pattern\b']
        
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.RAW,
            token_interval=20,
            time_interval_ms=500,
            max_chars_per_step=500,
            redaction_patterns=custom_patterns,
            event_emitter=mock_emitter
        )
        
        assert tracer.visibility == ReasoningVisibility.RAW
        assert tracer.token_interval == 20
        assert tracer.time_interval_ms == 500
        assert tracer.max_chars_per_step == 500
        assert tracer.event_emitter == mock_emitter
    
    def test_start_and_end_step(self):
        """Test starting and ending reasoning steps."""
        tracer = ReasoningTracer()
        correlation_id = "test-correlation-123"
        stage_id = "test-stage"
        
        # Start step
        tracer.start_step(correlation_id, stage_id)
        
        assert tracer.active
        assert not tracer.paused
        assert tracer.current_correlation_id == correlation_id
        assert tracer.current_stage_id == stage_id
        assert tracer.thought_buffer.content == ""
        
        # End step
        tracer.end_step()
        
        assert not tracer.active
        assert tracer.current_correlation_id is None
        assert tracer.current_stage_id is None
    
    def test_pause_and_resume(self):
        """Test pausing and resuming thought collection."""
        tracer = ReasoningTracer()
        tracer.start_step("test-correlation")
        
        # Initially not paused
        assert not tracer.paused
        
        # Pause
        tracer.pause()
        assert tracer.paused
        
        # Resume
        tracer.resume()
        assert not tracer.paused
    
    def test_add_thought_when_inactive(self):
        """Test that thoughts are not collected when inactive."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(event_emitter=mock_emitter)
        
        # Add thought when inactive
        tracer.add_thought("This should be ignored")
        
        assert tracer.thought_buffer.content == ""
        mock_emitter.assert_not_called()
    
    def test_add_thought_when_paused(self):
        """Test that thoughts are not collected when paused."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(event_emitter=mock_emitter)
        tracer.start_step("test-correlation")
        tracer.pause()
        
        # Add thought when paused
        tracer.add_thought("This should be ignored")
        
        assert tracer.thought_buffer.content == ""
        mock_emitter.assert_not_called()
    
    def test_add_thought_when_hidden(self):
        """Test that thoughts are not collected when visibility is hidden."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.HIDDEN,
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        # Add thought when hidden
        tracer.add_thought("This should be ignored")
        
        assert tracer.thought_buffer.content == ""
        mock_emitter.assert_not_called()
    
    def test_add_thought_basic(self):
        """Test basic thought addition."""
        tracer = ReasoningTracer(token_interval=100)  # High interval to prevent auto-emission
        tracer.start_step("test-correlation")
        
        # Add thought
        thought_content = "This is a test thought"
        tracer.add_thought(thought_content)
        
        assert tracer.thought_buffer.content == thought_content
        assert tracer.thought_buffer.token_count > 0
    
    def test_add_thought_with_token_count(self):
        """Test thought addition with explicit token count."""
        tracer = ReasoningTracer(token_interval=100)
        tracer.start_step("test-correlation")
        
        # Add thought with specific token count
        thought_content = "Test thought"
        token_count = 10
        tracer.add_thought(thought_content, token_count)
        
        assert tracer.thought_buffer.content == thought_content
        assert tracer.thought_buffer.token_count == token_count
    
    def test_add_reasoning_step(self):
        """Test adding structured reasoning steps."""
        tracer = ReasoningTracer(token_interval=100)
        tracer.start_step("test-correlation")
        
        # Add reasoning step
        step_desc = "Analyzing query"
        details = "Breaking down the user question"
        tracer.add_reasoning_step(step_desc, details)
        
        expected_content = f"\n[REASONING] {step_desc}: {details}\n"
        assert expected_content in tracer.thought_buffer.content
    
    def test_add_decision_point(self):
        """Test adding decision points."""
        tracer = ReasoningTracer(token_interval=100)
        tracer.start_step("test-correlation")
        
        # Add decision point
        decision = "Use web search"
        rationale = "User question requires current information"
        tracer.add_decision_point(decision, rationale)
        
        expected_content = f"\n[DECISION] {decision} | Rationale: {rationale}\n"
        assert expected_content in tracer.thought_buffer.content
    
    def test_thought_snapshot_emission_by_tokens(self):
        """Test thought snapshot emission triggered by token count."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            token_interval=5,  # Low interval for testing
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation", "test-stage")
        
        # Add enough content to trigger token-based emission
        tracer.add_thought("This is a long thought ", token_count=3)
        tracer.add_thought("that should trigger emission", token_count=3)
        
        # Should have called the emitter
        mock_emitter.assert_called()
        
        # Verify the call was for a thought snapshot
        call_args = mock_emitter.call_args[0][0]  # First argument of the call
        assert call_args.event_type == IntermediateEventType.THOUGHT_SNAPSHOT
        assert call_args.correlation_id == "test-correlation"
        assert call_args.stage_id == "test-stage"
    
    def test_thought_snapshot_emission_by_time(self):
        """Test thought snapshot emission triggered by time interval."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            time_interval_ms=50,  # Short interval for testing
            token_interval=1000,  # High token interval
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        # Add some content
        tracer.add_thought("Initial thought", token_count=1)
        
        # Wait for time interval
        time.sleep(0.06)  # 60ms > 50ms interval
        
        # Add more content to trigger time-based check
        tracer.add_thought(" additional content", token_count=1)
        
        # Should have called the emitter
        mock_emitter.assert_called()
    
    def test_visibility_summarized_processing(self):
        """Test thought processing with summarized visibility."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.SUMMARIZED,
            token_interval=5,
            max_chars_per_step=50,
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        # Add content with reasoning markers
        tracer.add_reasoning_step("Step 1", "Important reasoning")
        tracer.add_decision_point("Decision", "Critical choice")
        tracer.add_thought("Regular thought that might be filtered", token_count=10)
        
        # Should emit with summarized content
        mock_emitter.assert_called()
        
        call_args = mock_emitter.call_args[0][0]
        event_data = call_args.data
        
        # Should contain reasoning steps and decisions
        assert "[REASONING]" in event_data["content"] or "[DECISION]" in event_data["content"]
    
    def test_visibility_raw_processing(self):
        """Test thought processing with raw visibility."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.RAW,
            token_interval=5,
            max_chars_per_step=50,
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        long_content = "This is a very long thought that should be truncated when using raw visibility mode"
        tracer.add_thought(long_content, token_count=10)
        
        mock_emitter.assert_called()
        
        call_args = mock_emitter.call_args[0][0]
        event_data = call_args.data
        
        # Should be truncated to max_chars_per_step
        assert len(event_data["content"]) <= 50
        assert event_data["content"].endswith("...")
    
    def test_final_snapshot_on_end_step(self):
        """Test that final snapshot is emitted when ending step."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            token_interval=1000,  # High interval to prevent auto-emission
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        # Add some content
        tracer.add_thought("Final thoughts", token_count=1)
        
        # End the step - should emit final snapshot
        tracer.end_step()
        
        mock_emitter.assert_called()
        
        call_args = mock_emitter.call_args[0][0]
        assert call_args.data["is_final"] is True
    
    def test_sequence_counter_increment(self):
        """Test that sequence counter increments properly."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            token_interval=1,  # Low interval for multiple emissions
            event_emitter=mock_emitter
        )
        tracer.start_step("test-correlation")
        
        # Add multiple thoughts to trigger multiple emissions
        tracer.add_thought("First", token_count=2)
        tracer.add_thought("Second", token_count=2)
        tracer.add_thought("Third", token_count=2)
        
        # Should have multiple calls with incrementing sequence numbers
        assert mock_emitter.call_count >= 2
        
        # Check sequence numbers are incrementing
        call_sequences = [call.args[0].sequence for call in mock_emitter.call_args_list]
        assert len(set(call_sequences)) == len(call_sequences)  # All unique
        assert call_sequences == sorted(call_sequences)  # Ascending order
    
    def test_get_current_thought_summary(self):
        """Test getting current thought summary without emission."""
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.SUMMARIZED,
            token_interval=1000  # Prevent auto-emission
        )
        tracer.start_step("test-correlation")
        
        # Add content
        tracer.add_reasoning_step("Analysis", "Thinking about the problem")
        
        # Get summary without emitting
        summary = tracer.get_current_thought_summary()
        
        assert summary is not None
        assert "[REASONING]" in summary
    
    def test_get_current_thought_summary_when_inactive(self):
        """Test getting summary when tracer is inactive."""
        tracer = ReasoningTracer()
        
        summary = tracer.get_current_thought_summary()
        assert summary is None
    
    def test_event_emitter_setter(self):
        """Test setting event emitter after initialization."""
        tracer = ReasoningTracer()
        new_emitter = Mock()
        
        tracer.set_event_emitter(new_emitter)
        assert tracer.event_emitter == new_emitter


class TestReasoningTracerIntegration:
    """Integration tests for reasoning tracer."""
    
    def test_global_tracer_functions(self):
        """Test global tracer convenience functions."""
        # Test get_reasoning_tracer
        tracer1 = get_reasoning_tracer()
        tracer2 = get_reasoning_tracer()
        
        # Should return the same instance
        assert tracer1 is tracer2
        
        # Test initialize_reasoning_tracer
        mock_emitter = Mock()
        tracer3 = initialize_reasoning_tracer(
            visibility=ReasoningVisibility.RAW,
            token_interval=10,
            event_emitter=mock_emitter
        )
        
        assert tracer3.visibility == ReasoningVisibility.RAW
        assert tracer3.token_interval == 10
        assert tracer3.event_emitter == mock_emitter
        
        # Should be the new global instance
        tracer4 = get_reasoning_tracer()
        assert tracer3 is tracer4
    
    def test_realistic_agent_workflow(self):
        """Test realistic agent workflow with reasoning tracer."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            visibility=ReasoningVisibility.SUMMARIZED,
            token_interval=10,
            time_interval_ms=100,
            event_emitter=mock_emitter
        )
        
        # Simulate agent workflow
        correlation_id = "workflow-123"
        tracer.start_step(correlation_id, "query_analysis")
        
        # Simulate reasoning process
        tracer.add_reasoning_step("Analyze user query", "User asks about machine learning")
        tracer.add_decision_point("Use web search", "Need current ML information")
        tracer.add_thought("The query seems to focus on recent ML developments", token_count=5)
        tracer.add_reasoning_step("Plan search strategy", "Will search for recent ML papers")
        tracer.add_thought("Should include both academic and industry sources", token_count=5)
        
        # End workflow step
        tracer.end_step()
        
        # Verify events were emitted
        assert mock_emitter.call_count > 0
        
        # Verify all events have correct correlation_id
        for call in mock_emitter.call_args_list:
            event = call.args[0]
            assert event.correlation_id == correlation_id
            assert event.event_type == IntermediateEventType.THOUGHT_SNAPSHOT
    
    def test_performance_with_rapid_thoughts(self):
        """Test performance with rapid thought addition."""
        mock_emitter = Mock()
        tracer = ReasoningTracer(
            token_interval=50,
            event_emitter=mock_emitter
        )
        tracer.start_step("perf-test")
        
        # Add many thoughts rapidly
        start_time = time.time()
        for i in range(100):
            tracer.add_thought(f"Thought {i} with some content", token_count=3)
        
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0
        
        # Should have emitted some events
        assert mock_emitter.call_count > 0
    
    def test_memory_usage_with_long_session(self):
        """Test memory usage during long reasoning session."""
        tracer = ReasoningTracer(
            token_interval=1000,  # Prevent frequent emissions
            max_chars_per_step=500
        )
        tracer.start_step("long-session")
        
        # Add lots of content
        for i in range(1000):
            tracer.add_thought(f"Iteration {i} with repeated content ", token_count=1)
        
        # Buffer should be managed/truncated
        assert len(tracer.thought_buffer.content) <= tracer.max_chars_per_step * 2  # Some buffer allowed


if __name__ == "__main__":
    pytest.main([__file__])
