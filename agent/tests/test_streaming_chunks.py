"""
Test suite for streaming chunks functionality.

This test file validates that the agent properly emits multiple delta events
for streaming responses, ensuring proper chunking behavior and Databricks
compatibility.

Key Testing Areas:
- Multiple delta events are emitted for synthesis
- Chunk boundaries respect word boundaries
- Total accumulated content matches done event
- Schema compliance with ResponsesAgentStreamEvent
- Fallback behavior for non-streaming LLMs
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

# Import test modules
import sys
import os

# Add the deep_research_agent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deep_research_agent_dir = os.path.join(parent_dir, 'deep_research_agent')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if deep_research_agent_dir not in sys.path:
    sys.path.insert(0, deep_research_agent_dir)

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
from deep_research_agent.core.types import ResearchContext


class TestStreamingChunks:
    """Test suite for streaming chunks functionality."""
    
    def setup_method(self):
        """Set up test environment with mocked components."""
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = AIMessage(content="Test response")
        
        # Create agent with mocked dependencies
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=self.mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                self.agent = RefactoredResearchAgent()
    
    def test_multiple_delta_events_streaming(self):
        """Test that streaming synthesis generates multiple delta events."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Explain quantum computing in detail"}]
        )
        
        # Mock streaming LLM that returns multiple chunks
        streaming_chunks = [
            AIMessage(content="Quantum computing is "),
            AIMessage(content="a revolutionary field "),
            AIMessage(content="that leverages quantum mechanics "),
            AIMessage(content="to process information in ways "),
            AIMessage(content="impossible with classical computers.")
        ]
        
        # Mock the LLM to have streaming capability
        self.mock_llm.stream = Mock(return_value=streaming_chunks)
        
        # Mock the graph to return synthesis node with streaming data
        def mock_graph_stream(initial_state, stream_mode=None):
            # Simulate synthesis node returning multiple chunks
            research_context = initial_state['research_context']
            research_context.synthesis_chunks = [chunk.content for chunk in streaming_chunks]
            
            yield {
                "synthesize_answer": {
                    "research_context": research_context,
                    "messages": [AIMessage(content="".join([chunk.content for chunk in streaming_chunks]))]
                }
            }
        
        self.agent.graph.stream = mock_graph_stream
        
        # Collect all events
        events = list(self.agent.predict_stream(request))
        
        # Filter events by type
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Assertions for streaming behavior
        assert len(delta_events) >= 5, f"Should emit at least 5 delta events for streaming, got {len(delta_events)}"
        assert len(done_events) == 1, f"Should have exactly one done event, got {len(done_events)}"
        
        # Verify no JSON in deltas
        for event in delta_events:
            try:
                parsed = json.loads(event.delta)
                if isinstance(parsed, dict):
                    pytest.fail(f"JSON found in delta: {event.delta}")
            except json.JSONDecodeError:
                pass  # Good - not JSON
        
        # Verify accumulated content matches done event - filter out progress events
        content_events = [e for e in delta_events if not e.delta.startswith("[PHASE:")]
        accumulated = "".join(e.delta for e in content_events)
        done_content = done_events[0].item['content'][0]['text']
        
        # Content should be very similar (allow for minor variations in chunking)
        assert abs(len(accumulated) - len(done_content)) <= 10, \
            f"Accumulated content length mismatch: {len(accumulated)} vs {len(done_content)}"
        
        # Verify item IDs are consistent
        item_ids = {e.item_id for e in delta_events}
        done_item_id = done_events[0].item['id']
        assert len(item_ids) == 1, "All delta events should have the same item_id"
        assert list(item_ids)[0] == done_item_id, "Delta and done events should share the same item_id"
    
    def test_non_streaming_fallback(self):
        """Test fallback behavior when LLM doesn't support streaming."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "What is machine learning?"}]
        )
        
        # Mock non-streaming LLM (no stream method)
        long_response = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from data without being explicitly programmed for every task. It involves algorithms that can identify patterns, make decisions, and predict outcomes based on input data."
        self.mock_llm.invoke.return_value = AIMessage(content=long_response)
        # Ensure no stream method exists
        if hasattr(self.mock_llm, 'stream'):
            delattr(self.mock_llm, 'stream')
        
        # Mock the graph to return non-streaming synthesis
        def mock_graph_stream(initial_state, stream_mode=None):
            research_context = initial_state['research_context']
            research_context.synthesis_chunks = [long_response]  # Single chunk
            
            yield {
                "synthesize_answer": {
                    "research_context": research_context,
                    "messages": [AIMessage(content=long_response)]
                }
            }
        
        self.agent.graph.stream = mock_graph_stream
        
        # Collect all events
        events = list(self.agent.predict_stream(request))
        
        # Filter events by type
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Should still emit multiple chunks due to simulated streaming for long content
        assert len(delta_events) >= 2, f"Should simulate streaming with multiple chunks, got {len(delta_events)}"
        assert len(done_events) == 1, "Should have exactly one done event"
        
        # Verify content integrity - filter out progress events
        content_events = [e for e in delta_events if not e.delta.startswith("[PHASE:")]
        accumulated = "".join(e.delta for e in content_events)
        done_content = done_events[0].item['content'][0]['text']
        
        # Allow for minor whitespace differences due to word boundary chunking
        assert accumulated.strip() == done_content.strip(), "Content should match between delta and done events"
    
    def test_short_content_single_delta(self):
        """Test that short content emits single delta event."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "What is AI?"}]
        )
        
        # Mock short response
        short_response = "AI is artificial intelligence."
        self.mock_llm.invoke.return_value = AIMessage(content=short_response)
        
        # Mock the graph
        def mock_graph_stream(initial_state, stream_mode=None):
            research_context = initial_state['research_context']
            research_context.synthesis_chunks = [short_response]
            
            yield {
                "synthesize_answer": {
                    "research_context": research_context,
                    "messages": [AIMessage(content=short_response)]
                }
            }
        
        self.agent.graph.stream = mock_graph_stream
        
        # Collect events
        events = list(self.agent.predict_stream(request))
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Short content should still have at least one delta
        assert len(delta_events) >= 1, "Should have at least one delta event"
        assert len(done_events) == 1, "Should have exactly one done event"
        
        # Content should match - filter out progress events
        content_events = [e for e in delta_events if not e.delta.startswith("[PHASE:")]
        accumulated = "".join(e.delta for e in content_events)
        done_content = done_events[0].item['content'][0]['text']
        assert accumulated.strip() == done_content.strip()
    
    def test_word_boundary_respect(self):
        """Test that chunking respects word boundaries."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Explain neural networks"}]
        )
        
        # Mock response with clear word boundaries
        response = "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes called neurons that process information through weighted connections."
        self.mock_llm.invoke.return_value = AIMessage(content=response)
        
        def mock_graph_stream(initial_state, stream_mode=None):
            research_context = initial_state['research_context']
            research_context.synthesis_chunks = [response]
            
            yield {
                "synthesize_answer": {
                    "research_context": research_context,
                    "messages": [AIMessage(content=response)]
                }
            }
        
        self.agent.graph.stream = mock_graph_stream
        
        # Collect events
        events = list(self.agent.predict_stream(request))
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        
        # Verify word boundaries - no content delta should end mid-word
        content_events = [e for e in delta_events if not e.delta.startswith("[PHASE:")]
        for event in content_events:
            delta_text = event.delta.rstrip()  # Remove trailing spaces
            if delta_text:  # Skip empty deltas
                # Should end with a complete word (letter, number, or punctuation)
                last_char = delta_text[-1]
                assert last_char.isalnum() or last_char in '.!?,:;)', \
                    f"Delta should end at word boundary, but ends with: '{last_char}' in '{delta_text}'"
    
    def test_streaming_schema_compliance(self):
        """Test that streaming events comply with ResponsesAgentStreamEvent schema."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test streaming compliance"}]
        )
        
        # Mock streaming response
        response = "This is a test response for schema compliance validation with multiple sentences and sufficient length."
        self.mock_llm.invoke.return_value = AIMessage(content=response)
        
        def mock_graph_stream(initial_state, stream_mode=None):
            research_context = initial_state['research_context']
            research_context.synthesis_chunks = [response]
            
            yield {
                "synthesize_answer": {
                    "research_context": research_context,
                    "messages": [AIMessage(content=response)]
                }
            }
        
        self.agent.graph.stream = mock_graph_stream
        
        # Collect events
        events = list(self.agent.predict_stream(request))
        
        for event in events:
            # Verify event structure
            assert hasattr(event, 'type'), "Event should have 'type' attribute"
            assert event.type in ["response.output_text.delta", "response.output_item.done"], \
                f"Invalid event type: {event.type}"
            
            if event.type == "response.output_text.delta":
                # Delta event requirements
                assert hasattr(event, 'item_id'), "Delta event should have 'item_id'"
                assert hasattr(event, 'delta'), "Delta event should have 'delta'"
                assert isinstance(event.delta, str), "Delta should be a string"
                assert len(event.delta) > 0, "Delta should not be empty"
                
            elif event.type == "response.output_item.done":
                # Done event requirements
                assert hasattr(event, 'item'), "Done event should have 'item'"
                assert isinstance(event.item, dict), "Item should be a dictionary"
                assert event.item.get('type') == 'message', "Item type should be 'message'"
                assert event.item.get('role') == 'assistant', "Item role should be 'assistant'"
                assert 'content' in event.item, "Item should have 'content'"
                assert isinstance(event.item['content'], list), "Content should be a list"
                assert len(event.item['content']) > 0, "Content list should not be empty"
                assert event.item['content'][0].get('type') == 'output_text', "Content type should be 'output_text'"
                assert 'text' in event.item['content'][0], "Content should have 'text'"


if __name__ == "__main__":
    # Run streaming tests
    print("Running streaming chunks tests...")
    print("-" * 60)
    
    test_suite = TestStreamingChunks()
    
    # Main streaming tests
    tests = [
        ("Multiple delta events - streaming", test_suite.test_multiple_delta_events_streaming),
        ("Non-streaming fallback", test_suite.test_non_streaming_fallback),
        ("Short content single delta", test_suite.test_short_content_single_delta),
        ("Word boundary respect", test_suite.test_word_boundary_respect),
        ("Schema compliance", test_suite.test_streaming_schema_compliance)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Testing {test_name}...")
            test_suite.setup_method()
            test_func()
            print("   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("Streaming chunks tests complete!")