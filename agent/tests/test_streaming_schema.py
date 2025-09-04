"""
Comprehensive test for validating streaming response schema.

This test ensures the streaming output follows the correct schema and prevents
regression of the mixed delta/JSON streaming issue.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

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


class TestStreamingSchema:
    """Test suite for validating streaming response schema."""
    
    def test_no_json_objects_in_stream(self):
        """
        Test that no JSON objects from intermediate nodes appear in the stream.
        This prevents regression of the issue where query generation and reflection
        nodes emit their outputs as JSON.
        """
        # Create mock LLM that returns predictable responses
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            # Query generation response
            AIMessage(content='{"queries": ["test query 1", "test query 2"]}'),
            # Reflection response
            AIMessage(content='{"needs_more_research": false, "reflection": "Test complete"}'),
            # Final synthesis (streamed)
            AIMessage(content="This is the final synthesized answer.")
        ]
        
        # Create agent with mocked dependencies
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)  # 6 components
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
        
        # Create test request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "hi"}]
        )
        
        # Collect all stream events
        stream_events = list(agent.predict_stream(request))
        
        # Verify no JSON objects in delta events
        for event in stream_events:
            if event.type == "response.output_text.delta":
                delta_content = event.delta if hasattr(event, 'delta') else None
                
                # Check that delta content is not a JSON object
                if delta_content:
                    try:
                        parsed = json.loads(delta_content)
                        if isinstance(parsed, dict) and ('queries' in parsed or 'needs_more_research' in parsed):
                            pytest.fail(f"Found JSON object in delta event: {parsed}")
                    except json.JSONDecodeError:
                        # This is expected - delta should be plain text
                        pass
    
    def test_delta_event_schema(self):
        """
        Test that delta events have the correct schema.
        """
        # Create a minimal agent
        mock_llm = Mock()
        mock_llm.stream.return_value = [
            Mock(content="Hello "),
            Mock(content="world"),
            Mock(content="!")
        ]
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)  # 6 components
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
        
        # Test delta creation directly
        delta_dict = agent.create_text_delta("test content", "item-123")
        
        # Validate delta schema
        assert delta_dict["type"] == "response.output_text.delta"
        assert delta_dict["item_id"] == "item-123"
        assert delta_dict["delta"] == "test content"
        
        # Verify it can be used with ResponsesAgentStreamEvent
        event = ResponsesAgentStreamEvent(**delta_dict)
        assert event.type == "response.output_text.delta"
        assert event.item_id == "item-123"
        assert event.delta == "test content"
    
    def test_done_event_schema(self):
        """
        Test that done events have the correct schema.
        """
        # Create a minimal agent
        mock_phase2_return = (None, None, None, None, None, None)  # 6 components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
            agent = RefactoredResearchAgent()
        
        # Test done event creation
        done_item = agent.create_text_output_item("Final response", "item-456")
        
        # Validate done item schema
        assert done_item["type"] == "message"
        assert done_item["role"] == "assistant"
        assert done_item["id"] == "item-456"
        assert done_item["content"][0]["type"] == "output_text"
        assert done_item["content"][0]["text"] == "Final response"
        
        # Verify it can be used with ResponsesAgentStreamEvent
        event = ResponsesAgentStreamEvent(type="response.output_item.done", item=done_item)
        assert event.type == "response.output_item.done"
        assert event.item == done_item
    
    def test_stream_integrity(self):
        """
        Test that the stream maintains integrity:
        - Delta events contain only text fragments
        - Final assembled content matches done event
        - No duplicate responses
        """
        # Mock graph stream to return predictable chunks
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            ("messages", [Mock(content="Hello ")]),
            ("messages", [Mock(content="world")]),
            ("messages", [Mock(content="!")]),
        ]
        
        # Create agent with mocked graph
        mock_phase2_return = (None, None, None, None, None, None)  # 6 components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
            agent = RefactoredResearchAgent()
            agent.graph = mock_graph
        
        # Create test request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Collect all stream events
        stream_events = list(agent.predict_stream(request))
        
        # Separate delta and done events
        delta_events = [e for e in stream_events if e.type == "response.output_text.delta"]
        done_events = [e for e in stream_events if e.type == "response.output_item.done"]
        
        # Verify we have delta events
        assert len(delta_events) > 0, "No delta events found"
        
        # Verify we have exactly one done event
        assert len(done_events) == 1, f"Expected 1 done event, got {len(done_events)}"
        
        # Assemble content from deltas
        assembled_content = ""
        item_ids = set()
        for event in delta_events:
            assembled_content += event.delta
            item_ids.add(event.item_id)
        
        # All deltas should have the same item_id
        assert len(item_ids) == 1, "Delta events have different item_ids"
        
        # Get content from done event
        done_content = done_events[0].item["content"][0]["text"]
        
        # Verify assembled content matches done event content
        assert assembled_content == done_content, f"Assembled: '{assembled_content}' != Done: '{done_content}'"
        
        # Verify done event has the same item_id
        assert done_events[0].item["id"] in item_ids, "Done event has different item_id than deltas"
    
    def test_no_duplicate_responses(self):
        """
        Test that the final response is not duplicated in the stream.
        """
        # Mock graph stream to return simple content
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {
                "synthesize_answer": {
                    "messages": [Mock(content="Simple response")]
                }
            }
        ]
        
        # Create agent with mocked graph
        mock_phase2_return = (None, None, None, None, None, None)  # 6 components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
            agent = RefactoredResearchAgent()
            agent.graph = mock_graph
        
        # Create test request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Collect all stream events
        stream_events = list(agent.predict_stream(request))
        
        # Get done events
        done_events = [e for e in stream_events if e.type == "response.output_item.done"]
        
        # Should have exactly one done event
        assert len(done_events) == 1, f"Expected 1 done event, got {len(done_events)}"
        
        # Check that the content appears only once in done events
        done_content = done_events[0].item["content"][0]["text"]
        assert done_content == "Simple response"
    
    def test_regression_hi_query(self):
        """
        Specific regression test for the "hi" query that exposed the original issue.
        This test ensures that querying with "hi" doesn't produce mixed JSON/delta output.
        """
        # Mock the workflow to simulate the problematic scenario
        mock_graph = Mock()
        
        # Simulate the actual problematic flow
        mock_graph.stream.return_value = [
            # This simulates the synthesis streaming
            {"synthesize_answer": {"messages": [Mock(content="H")]}},
            {"synthesize_answer": {"messages": [Mock(content="e")]}},
            {"synthesize_answer": {"messages": [Mock(content="l")]}},
            {"synthesize_answer": {"messages": [Mock(content="l")]}},
            {"synthesize_answer": {"messages": [Mock(content="o")]}},
            {"synthesize_answer": {"messages": [Mock(content="!")]}},
        ]
        
        # Create agent
        mock_phase2_return = (None, None, None, None, None, None)  # 6 components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
            agent = RefactoredResearchAgent()
            agent.graph = mock_graph
        
        # Create the problematic request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "hi"}]
        )
        
        # Collect all stream events
        stream_events = list(agent.predict_stream(request))
        
        # Check for the specific issues from the original problem
        for event in stream_events:
            if event.type == "response.output_text.delta":
                # Delta should not contain JSON objects
                assert not event.delta.startswith("{"), f"Delta starts with JSON: {event.delta}"
                assert "queries" not in event.delta, f"Found 'queries' in delta: {event.delta}"
                assert "needs_more_research" not in event.delta, f"Found 'needs_more_research' in delta: {event.delta}"
        
        # Verify clean streaming
        delta_events = [e for e in stream_events if e.type == "response.output_text.delta"]
        done_events = [e for e in stream_events if e.type == "response.output_item.done"]
        
        # Separate progress events from content events
        progress_events = [e for e in delta_events if e.delta.startswith("[PHASE:")]
        content_events = [e for e in delta_events if not e.delta.startswith("[PHASE:")]
        
        # We should have both progress and content events now
        assert len(progress_events) > 0, f"Expected progress events, got {len(progress_events)}"
        assert len(content_events) == 6, f"Expected 6 content delta events, got {len(content_events)}"
        assert len(done_events) == 1, f"Expected 1 done event, got {len(done_events)}"
        
        # Verify the assembled content from content events only
        assembled = "".join([e.delta for e in content_events])
        assert assembled == "Hello!", f"Unexpected assembled content: {assembled}"
    
    def test_predict_uses_only_done_events(self):
        """
        Test that the predict() method only uses done events, not delta events.
        """
        # Mock graph stream
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [Mock(content="Test ")]}},
            {"synthesize_answer": {"messages": [Mock(content="response")]}},
        ]
        
        # Create agent
        mock_phase2_return = (None, None, None, None, None, None)  # 6 components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
            agent = RefactoredResearchAgent()
            agent.graph = mock_graph
        
        # Create request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Call predict (not predict_stream)
        response = agent.predict(request)
        
        # Verify response structure
        assert isinstance(response, ResponsesAgentResponse)
        assert len(response.output) == 1, f"Expected 1 output item, got {len(response.output)}"
        
        # Verify the output is from done event, not deltas
        output_item = response.output[0]
        # OutputItem has attributes, not dict keys
        assert output_item.type == "message"
        assert output_item.content[0]["text"] == "Test response"  # Should be concatenated content


if __name__ == "__main__":
    # Run specific test for debugging
    test_suite = TestStreamingSchema()
    
    print("Running streaming schema tests...")
    print("-" * 60)
    
    try:
        print("1. Testing no JSON objects in stream...")
        test_suite.test_no_json_objects_in_stream()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("2. Testing delta event schema...")
        test_suite.test_delta_event_schema()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("3. Testing done event schema...")
        test_suite.test_done_event_schema()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("4. Testing stream integrity...")
        test_suite.test_stream_integrity()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("5. Testing no duplicate responses...")
        test_suite.test_no_duplicate_responses()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("6. Testing regression for 'hi' query...")
        test_suite.test_regression_hi_query()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("7. Testing predict uses only done events...")
        test_suite.test_predict_uses_only_done_events()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print("-" * 60)
    print("Streaming schema tests complete!")