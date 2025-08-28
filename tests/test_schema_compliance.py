"""
Comprehensive schema compliance tests for predict and predict_stream methods.

This module tests all schema requirements documented in SCHEMA_REQUIREMENTS.md,
ensuring compliance with MLflow ResponsesAgent interface and Databricks Agent Framework.
"""

import pytest
import json
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Generator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.types.responses import (
    ResponsesAgentRequest, 
    ResponsesAgentResponse, 
    ResponsesAgentStreamEvent
)

# Import test modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
from deep_research_agent.core import ResearchContext, WorkflowMetrics


class TestCustomInputsHandling:
    """Test custom_inputs parameter handling in requests."""
    
    def test_custom_inputs_passthrough(self):
        """Test that custom_inputs are passed through to custom_outputs."""
        agent = RefactoredResearchAgent()
        
        # Create request with custom inputs
        custom_params = {
            "max_iterations": 3,
            "temperature": 0.5,
            "search_provider": "brave",
            "custom_flag": True
        }
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test query"}],
            custom_inputs=custom_params
        )
        
        # Mock the graph to avoid actual execution
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Test response")]}}
        ]
        agent.graph = mock_graph
        
        # Execute predict
        response = agent.predict(request)
        
        # Verify custom_outputs contains custom_inputs
        assert hasattr(response, 'custom_outputs')
        assert response.custom_outputs == custom_params
    
    def test_custom_inputs_in_streaming(self):
        """Test custom_inputs handling in streaming mode."""
        agent = RefactoredResearchAgent()
        
        custom_params = {"streaming_mode": "fast", "batch_size": 10}
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}],
            custom_inputs=custom_params
        )
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Response")]}}
        ]
        agent.graph = mock_graph
        
        # Stream and collect done events
        events = list(agent.predict_stream(request))
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Custom inputs don't appear in stream events, only in predict response
        assert len(done_events) == 1
        
        # But they should be available when using predict
        response = agent.predict(request)
        assert response.custom_outputs == custom_params
    
    def test_missing_custom_inputs(self):
        """Test that missing custom_inputs results in empty custom_outputs."""
        agent = RefactoredResearchAgent()
        
        # Request without custom_inputs
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Response")]}}
        ]
        agent.graph = mock_graph
        
        response = agent.predict(request)
        
        # Should have empty dict for custom_outputs
        assert hasattr(response, 'custom_outputs')
        assert response.custom_outputs == {}


class TestCustomOutputsValidation:
    """Test custom_outputs structure and content validation."""
    
    def test_custom_outputs_structure(self):
        """Test that custom_outputs has the expected structure."""
        from deep_research_agent.components.response_builder import ResponseBuilder
        
        builder = ResponseBuilder()
        
        # Create test data
        research_context = ResearchContext(original_question="test question")
        research_context.add_citation("http://example.com", "Test Source", "Test snippet")
        
        metrics = WorkflowMetrics()
        metrics.total_time = 5.0
        metrics.search_count = 3
        
        # Build custom outputs
        custom_outputs = builder._build_custom_outputs(research_context, metrics)
        
        # Validate structure
        assert isinstance(custom_outputs, dict)
        assert "citations" in custom_outputs
        assert "research_metadata" in custom_outputs
        assert custom_outputs["research_metadata"]["original_question"] == "test question"
        
        # Check web sources if present
        if "web_sources" in custom_outputs:
            assert isinstance(custom_outputs["web_sources"], list)
            for source in custom_outputs["web_sources"]:
                assert "url" in source
                assert "title" in source
    
    def test_custom_outputs_with_citations(self):
        """Test custom_outputs includes citations when available."""
        from deep_research_agent.components.response_builder import ResponseBuilder
        
        builder = ResponseBuilder()
        research_context = ResearchContext(original_question="test")
        
        # Add multiple citations
        research_context.add_citation("http://site1.com", "Site 1", "Snippet 1")
        research_context.add_citation("http://site2.com", "Site 2", "Snippet 2")
        
        custom_outputs = builder._build_custom_outputs(research_context, None)
        
        # Verify citations are included
        assert "citations" in custom_outputs
        assert len(custom_outputs["citations"]) == 2
        assert custom_outputs["citations"][0]["url"] == "http://site1.com"
        assert custom_outputs["citations"][1]["title"] == "Site 2"
    
    def test_custom_outputs_with_metrics(self):
        """Test custom_outputs includes workflow metrics when available."""
        from deep_research_agent.components.response_builder import ResponseBuilder
        
        builder = ResponseBuilder()
        
        metrics = WorkflowMetrics()
        metrics.execution_time_seconds = 10.5
        metrics.total_queries_generated = 5
        metrics.total_web_results = 3
        metrics.error_count = 2
        metrics.success_rate = 0.8
        
        custom_outputs = builder._build_custom_outputs(None, metrics)
        
        # Verify metrics are included
        assert "workflow_metrics" in custom_outputs
        assert custom_outputs["workflow_metrics"]["total_queries_generated"] == 5
        assert custom_outputs["workflow_metrics"]["error_count"] == 2
        assert custom_outputs["workflow_metrics"]["success_rate"] == 0.8
        assert custom_outputs["workflow_metrics"]["total_web_results"] == 3


class TestErrorScenarios:
    """Test error handling and edge cases in schema validation."""
    
    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        agent = RefactoredResearchAgent()
        
        # Test with empty input - should handle gracefully with fallback
        request = ResponsesAgentRequest(input=[])
        response = agent.predict(request)
        assert isinstance(response, ResponsesAgentResponse)
        # Should return fallback response
        assert len(response.output) == 1
        assert "ready to help" in response.output[0].content[0]["text"]
        
        # Test with valid input
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Mock graph to test message conversion
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Response")]}}
        ]
        agent.graph = mock_graph
        
        # Should handle gracefully (invalid roles are skipped or defaulted)
        response = agent.predict(request)
        assert isinstance(response, ResponsesAgentResponse)
    
    def test_empty_content_handling(self):
        """Test handling of empty content in messages."""
        agent = RefactoredResearchAgent()
        
        # Test with empty content string
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": ""}]
        )
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Default response")]}}
        ]
        agent.graph = mock_graph
        
        response = agent.predict(request)
        assert len(response.output) > 0
    
    def test_null_values_handling(self):
        """Test handling of null/None values in request."""
        agent = RefactoredResearchAgent()
        
        # Test extracting text from None
        result = agent._extract_text_content(None)
        assert result == "None"  # Falls back to string conversion
        
        # Test with None in content list
        content_with_none = [
            {"type": "text", "text": "Part 1"},
            None,
            {"type": "text", "text": "Part 2"}
        ]
        result = agent._extract_text_content(content_with_none)
        assert "Part 1" in result
        assert "Part 2" in result
    
    def test_invalid_role_handling(self):
        """Test that invalid roles are handled gracefully."""
        from deep_research_agent.components.message_converter import MessageConverter
        
        converter = MessageConverter()
        
        # Test with invalid role
        messages = [
            {"role": "invalid", "content": "test"},
            {"role": "user", "content": "valid message"}
        ]
        
        # Convert to langchain format
        langchain_msgs = converter.responses_to_langchain(messages)
        
        # Should skip invalid role or convert to default
        assert len(langchain_msgs) >= 1
        # Valid message should be converted
        assert any(isinstance(m, HumanMessage) for m in langchain_msgs)
    
    def test_json_in_delta_prevention(self):
        """Test that JSON objects are never sent in delta events."""
        agent = RefactoredResearchAgent()
        
        # Mock graph that returns JSON-like content
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            ("query_generation", {"queries": ["q1", "q2"]}),  # Should be filtered
            {"synthesize_answer": {"messages": [AIMessage(content="Actual response")]}}
        ]
        agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Collect all delta events
        events = list(agent.predict_stream(request))
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        
        # Check no JSON in deltas
        for event in delta_events:
            if hasattr(event, 'delta'):
                try:
                    parsed = json.loads(event.delta)
                    if isinstance(parsed, dict):
                        pytest.fail(f"JSON object found in delta: {parsed}")
                except json.JSONDecodeError:
                    pass  # Expected - should not be valid JSON


class TestAdvancedStreaming:
    """Test advanced streaming scenarios."""
    
    def test_concurrent_stream_requests(self):
        """Test handling multiple concurrent streaming requests."""
        agent = RefactoredResearchAgent()
        
        # Mock graph for consistent responses
        def create_mock_graph(response_text):
            mock = Mock()
            mock.stream.return_value = [
                {"synthesize_answer": {"messages": [AIMessage(content=response_text)]}}
            ]
            return mock
        
        # Create multiple requests
        requests = [
            ResponsesAgentRequest(input=[{"role": "user", "content": f"Query {i}"}])
            for i in range(3)
        ]
        
        # Execute concurrently
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            def process_request(req, idx):
                agent.graph = create_mock_graph(f"Response {idx}")
                events = list(agent.predict_stream(req))
                done_events = [e for e in events if e.type == "response.output_item.done"]
                return done_events
            
            futures = [
                executor.submit(process_request, req, idx) 
                for idx, req in enumerate(requests)
            ]
            
            for future in futures:
                results.append(future.result())
        
        # Verify all requests completed
        assert len(results) == 3
        for result in results:
            assert len(result) == 1  # Each should have one done event
    
    def test_large_content_streaming(self):
        """Test streaming with very large content."""
        agent = RefactoredResearchAgent()
        
        # Create large content (10KB of text)
        large_text = "A" * 10000
        
        # Mock graph with large response
        mock_graph = Mock()
        # Simulate chunked streaming
        chunks = [large_text[i:i+100] for i in range(0, len(large_text), 100)]
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content=chunk)]}}
            for chunk in chunks
        ]
        agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Stream and verify
        events = list(agent.predict_stream(request))
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Should have multiple delta events for large content
        assert len(delta_events) > 10
        
        # Verify content integrity
        assembled = "".join(e.delta for e in delta_events if hasattr(e, 'delta'))
        done_text = done_events[0].item["content"][0]["text"]
        assert assembled == done_text
        assert len(done_text) == 10000
    
    def test_multi_message_conversation(self):
        """Test handling multi-turn conversation in request."""
        agent = RefactoredResearchAgent()
        
        # Create conversation history
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
                {"role": "user", "content": "Tell me more about machine learning."},
                {"role": "assistant", "content": "Machine learning is a subset of AI."},
                {"role": "user", "content": "What are neural networks?"}
            ]
        )
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Neural networks are...")]}}
        ]
        agent.graph = mock_graph
        
        # Process request
        response = agent.predict(request)
        
        # Should handle conversation context
        assert len(response.output) > 0
        assert response.output[0].role == "assistant"
    
    def test_stream_interruption_recovery(self):
        """Test that streaming handles interruptions gracefully."""
        agent = RefactoredResearchAgent()
        
        # Mock graph that simulates interruption
        mock_graph = Mock()
        call_count = 0
        
        def stream_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails partway through
                yield {"synthesize_answer": {"messages": [AIMessage(content="Part 1")]}}
                raise ConnectionError("Stream interrupted")
            else:
                # Recovery attempt
                yield {"synthesize_answer": {"messages": [AIMessage(content="Recovery response")]}}
        
        mock_graph.stream = Mock(side_effect=stream_with_error)
        agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # First attempt should return error response (not raise exception)
        events = list(agent.predict_stream(request))
        done_events = [e for e in events if e.type == "response.output_item.done"]
        assert len(done_events) == 1
        # Should contain error message
        assert "Error:" in done_events[0].item["content"][0]["text"]
        
        # Agent should be reusable for next request  
        events = list(agent.predict_stream(request))
        done_events = [e for e in events if e.type == "response.output_item.done"]
        assert len(done_events) == 1


class TestDatabricksCompatibility:
    """Test DatabricksCompatibleAgent wrapper compliance."""
    
    def test_compatibility_wrapper_initialization(self):
        """Test that DatabricksCompatibleAgent properly wraps the agent."""
        compat_agent = DatabricksCompatibleAgent()
        
        # Should have the wrapped agent
        assert hasattr(compat_agent, 'agent')
        assert isinstance(compat_agent.agent, RefactoredResearchAgent)
        
        # Should expose required methods
        assert hasattr(compat_agent, 'predict')
        assert hasattr(compat_agent, 'predict_stream')
        assert hasattr(compat_agent, '_responses_to_cc')
        assert hasattr(compat_agent, 'create_text_output_item')
    
    def test_compatibility_request_handling(self):
        """Test that compatibility wrapper handles various request formats."""
        compat_agent = DatabricksCompatibleAgent()
        
        # Mock the wrapped agent
        mock_response = ResponsesAgentResponse(
            output=[{"type": "message", "role": "assistant", "id": "123", 
                    "content": [{"type": "output_text", "text": "Response"}]}],
            custom_outputs={}
        )
        compat_agent.agent.predict = Mock(return_value=mock_response)
        
        # Test with standard format
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        response = compat_agent.predict(request)
        assert isinstance(response, ResponsesAgentResponse)
        assert len(response.output) > 0
    
    def test_compatibility_streaming(self):
        """Test that compatibility wrapper properly forwards streaming."""
        compat_agent = DatabricksCompatibleAgent()
        
        # Mock streaming from wrapped agent
        mock_events = [
            ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                item_id="123",
                delta="Test "
            ),
            ResponsesAgentStreamEvent(
                type="response.output_text.delta", 
                item_id="123",
                delta="response"
            ),
            ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "role": "assistant", 
                    "id": "123",
                    "content": [{"type": "output_text", "text": "Test response"}]
                }
            )
        ]
        
        compat_agent.agent.predict_stream = Mock(return_value=iter(mock_events))
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Stream through compatibility layer
        events = list(compat_agent.predict_stream(request))
        
        assert len(events) == 3
        assert events[0].type == "response.output_text.delta"
        assert events[-1].type == "response.output_item.done"


class TestSchemaValidation:
    """Test schema validation against documented requirements."""
    
    def test_request_schema_requirements(self):
        """Test that requests meet all schema requirements."""
        # Valid request
        valid_request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"}
            ],
            custom_inputs={"param": "value"}
        )
        
        # Validate structure
        assert hasattr(valid_request, 'input')
        assert isinstance(valid_request.input, list)
        # After MLflow processing, input items are Message objects, not dicts
        assert all(hasattr(msg, 'role') and hasattr(msg, 'content') for msg in valid_request.input)
    
    def test_response_schema_requirements(self):
        """Test that responses meet all schema requirements."""
        agent = RefactoredResearchAgent()
        
        # Create a proper response
        output_item = agent.create_text_output_item("Test response", "msg-123")
        
        response = ResponsesAgentResponse(
            output=[output_item],
            custom_outputs={"key": "value"}
        )
        
        # Validate structure
        assert hasattr(response, 'output')
        assert hasattr(response, 'custom_outputs')
        assert isinstance(response.output, list)
        assert isinstance(response.custom_outputs, dict)
        
        # Validate output item structure
        item = response.output[0]
        assert item.type == "message"
        assert item.role == "assistant"
        assert hasattr(item, 'id')
        assert hasattr(item, 'content')
        assert isinstance(item.content, list)
        assert item.content[0]["type"] == "output_text"
        assert "text" in item.content[0]
    
    def test_stream_event_schema_requirements(self):
        """Test that stream events meet all schema requirements."""
        agent = RefactoredResearchAgent()
        
        # Test delta event creation
        delta_dict = agent.create_text_delta("Test chunk", "msg-456")
        delta_event = ResponsesAgentStreamEvent(**delta_dict)
        
        assert delta_event.type == "response.output_text.delta"
        assert delta_event.item_id == "msg-456"
        assert delta_event.delta == "Test chunk"
        
        # Test done event creation
        done_item = agent.create_text_output_item("Complete text", "msg-456")
        done_event = ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=done_item
        )
        
        assert done_event.type == "response.output_item.done"
        assert done_event.item["id"] == "msg-456"
        assert done_event.item["content"][0]["text"] == "Complete text"
    
    def test_id_consistency_requirement(self):
        """Test that item IDs are consistent between delta and done events."""
        agent = RefactoredResearchAgent()
        
        # Mock graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"synthesize_answer": {"messages": [AIMessage(content="Part 1 ")]}},
            {"synthesize_answer": {"messages": [AIMessage(content="Part 2")]}}
        ]
        agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "test"}]
        )
        
        # Collect events
        events = list(agent.predict_stream(request))
        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        done_events = [e for e in events if e.type == "response.output_item.done"]
        
        # Extract IDs
        delta_ids = {e.item_id for e in delta_events if hasattr(e, 'item_id')}
        done_ids = {e.item["id"] for e in done_events if "id" in e.item}
        
        # Should have exactly one ID used consistently
        assert len(delta_ids) == 1
        assert len(done_ids) == 1
        assert delta_ids == done_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])