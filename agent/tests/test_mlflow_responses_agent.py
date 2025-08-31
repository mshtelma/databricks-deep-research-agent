"""Tests for MLflow ResponsesAgent functionality."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from mlflow.types.responses import (
    ResponsesAgentRequest, 
    ResponsesAgentResponse, 
    ResponsesAgentStreamEvent
)
from mlflow.types.llm import ChatMessage

# Import the classes we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
# Backward compatibility alias  
DatabricksResearchAgent = RefactoredResearchAgent


class TestTextContentExtraction:
    """Test the _extract_text_content helper method."""
    
    def test_extract_string_content(self):
        """Test extracting text from string content."""
        agent = DatabricksResearchAgent()
        
        content = "Simple string content"
        result = agent._extract_text_content(content)
        
        assert result == "Simple string content"
    
    def test_extract_list_content_with_dicts(self):
        """Test extracting text from list of dictionaries."""
        agent = DatabricksResearchAgent()
        
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world!"}
        ]
        result = agent._extract_text_content(content)
        
        assert result == "Hello world!"
    
    def test_extract_mixed_list_content(self):
        """Test extracting text from mixed list content."""
        agent = DatabricksResearchAgent()
        
        content = [
            "Direct string",
            {"type": "text", "text": " and structured text"},
            {"type": "other", "text": " with type filter"}
        ]
        result = agent._extract_text_content(content)
        
        assert result == "Direct string and structured text with type filter"
    
    def test_extract_empty_list(self):
        """Test extracting text from empty list."""
        agent = DatabricksResearchAgent()
        
        content = []
        result = agent._extract_text_content(content)
        
        assert result == ""
    
    def test_extract_dict_without_text(self):
        """Test extracting from dict without 'text' key."""
        agent = DatabricksResearchAgent()
        
        content = [{"type": "image", "url": "http://example.com"}]
        result = agent._extract_text_content(content)
        
        assert result == ""
    
    def test_extract_fallback_conversion(self):
        """Test fallback string conversion."""
        agent = DatabricksResearchAgent()
        
        content = 12345
        result = agent._extract_text_content(content)
        
        assert result == "12345"


class TestResponsesAgentBaseFunctionality:
    """Test that the base ResponsesAgent methods work correctly."""
    
    def test_create_text_output_item_inherited(self):
        """Test that inherited create_text_output_item works."""
        agent = DatabricksResearchAgent()
        
        result = agent.create_text_output_item(text="Test message", id="test-id")
        
        # Should have the correct structure from base class
        assert isinstance(result, dict)
        assert result["id"] == "test-id"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert isinstance(result["content"], list)
        assert result["content"][0]["text"] == "Test message"
        assert result["content"][0]["type"] == "output_text"
    
    def test_create_function_call_item_inherited(self):
        """Test that inherited create_function_call_item works."""
        agent = DatabricksResearchAgent()
        
        result = agent.create_function_call_item(
            id="func-id",
            call_id="call-123",
            name="test_function",
            arguments='{"param": "value"}'
        )
        
        # Should have the correct structure from base class
        assert isinstance(result, dict)
        assert result["id"] == "func-id"
        assert result["call_id"] == "call-123"
        assert result["name"] == "test_function"
        assert result["arguments"] == '{"param": "value"}'


# NOTE: TestLangchainToResponsesConversion class removed
# The _langchain_to_responses method no longer exists in RefactoredResearchAgent
# Message conversion is handled internally through MessageConverter class

'''
class TestLangchainToResponsesConversion:
    """Test the _langchain_to_responses method."""
    
    @pytest.mark.skip(reason="_langchain_to_responses method not available in RefactoredResearchAgent")
    def test_convert_ai_message_string_content(self):
        """Test converting AI message with string content."""
        mock_combined_agent = Mock()
        mock_agent.return_value = mock_combined_agent
        
        agent = LangGraphResponsesAgent(mock_combined_agent)
        
        message = AIMessage(content="Simple response text", id="msg-123")
        messages = [message]
        
        result = agent._langchain_to_responses(messages)
        
        assert len(result) == 1
        item = result[0]
        assert item["id"] == "msg-123"
        assert item["type"] == "message"
        assert item["content"][0]["text"] == "Simple response text"
    
    @pytest.mark.skip(reason="_langchain_to_responses method not available in RefactoredResearchAgent")
    def test_convert_ai_message_structured_content(self):
        """Test converting AI message with structured content."""
        mock_combined_agent = Mock()
        mock_agent.return_value = mock_combined_agent
        
        agent = LangGraphResponsesAgent(mock_combined_agent)
        
        message = AIMessage(
            content=[
                {"type": "text", "text": "Research shows "},
                {"type": "text", "text": "AI is advancing rapidly."}
            ],
            id="msg-456"
        )
        messages = [message]
        
        result = agent._langchain_to_responses(messages)
        
        assert len(result) == 1
        item = result[0]
        assert item["id"] == "msg-456"
        assert item["content"][0]["text"] == "Research shows AI is advancing rapidly."
    
    @pytest.mark.skip(reason="_langchain_to_responses method not available in RefactoredResearchAgent")
    def test_convert_ai_message_with_tool_calls(self):
        """Test converting AI message with tool calls."""
        mock_combined_agent = Mock()
        mock_agent.return_value = mock_combined_agent
        
        agent = LangGraphResponsesAgent(mock_combined_agent)
        
        message = AIMessage(
            content="I'll search for information.",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "web_search", 
                    "args": {"query": "AI trends"}
                }
            ],
            id="msg-789"
        )
        messages = [message]
        
        result = agent._langchain_to_responses(messages)
        
        assert len(result) == 1
        item = result[0]
        assert item["call_id"] == "call_123"
        assert item["name"] == "web_search"
        assert json.loads(item["arguments"])["query"] == "AI trends"
'''


class TestResponsesAgentPredictMethod:
    """Test the predict method functionality."""
    
    def test_predict_with_structured_content(self):
        """Test predict method with structured message content."""
        
        # Mock the graph execution
        from deep_research_agent.core import ResearchContext, WorkflowMetrics
        mock_final_state = {
            "messages": [
                AIMessage(content="Based on research, here's what I found.")
            ],
            "research_context": ResearchContext(original_question="Test query"),
            "workflow_metrics": WorkflowMetrics(),
        }
        
        def mock_stream(initial_state, stream_mode=None):
            # Return streaming events in synthesize_answer node format
            from unittest.mock import Mock
            content = "Based on research, here's what I found."
            # Yield as a dictionary with the node name as key
            yield {
                "synthesize_answer": {
                    "messages": [Mock(content=content)]
                }
            }

        with patch('deep_research_agent.research_agent_refactored.RefactoredResearchAgent._build_workflow_graph') as mock_build:
            # Create a mock graph
            mock_graph = Mock()
            mock_graph.stream.side_effect = lambda initial_state, stream_mode=None: mock_stream(initial_state, stream_mode)
            mock_build.return_value = mock_graph
            
            agent = DatabricksResearchAgent()
            
            request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "Test query"}]
            )
            
            response = agent.predict(request)
            
            assert isinstance(response, ResponsesAgentResponse)
            assert len(response.output) == 1
            
            output_item = response.output[0]
            assert output_item.type == "message"
            assert output_item.role == "assistant"
            assert output_item.content[0]["text"] == "Based on research, here's what I found."


class TestResponsesAgentStreamEvent:
    """Test ResponsesAgentStreamEvent creation and validation."""
    
    def test_stream_event_creation_with_valid_item(self):
        """Test creating ResponsesAgentStreamEvent with valid output item."""
        item = {
            "id": str(uuid4()),
            "content": [{"text": "Test response", "type": "output_text"}],
            "role": "assistant",
            "type": "message"
        }
        
        event = ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
        
        assert event.type == "response.output_item.done"
        assert event.item == item
    
    def test_stream_event_validation_error_with_invalid_content(self):
        """Test that invalid content structure raises validation error."""
        # This should fail validation - content.0.text should be string, not list
        invalid_item = {
            "id": str(uuid4()),
            "content": [{"text": [{"text": "nested", "type": "output_text"}], "type": "output_text"}],
            "role": "assistant", 
            "type": "message"
        }
        
        with pytest.raises(Exception):  # Should raise pydantic ValidationError
            ResponsesAgentStreamEvent(type="response.output_item.done", item=invalid_item)


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_predict_handles_graph_exception(self):
        """Test that predict method handles graph execution exceptions."""
        
        agent = DatabricksResearchAgent()
        
        def mock_stream_with_exception(initial_state, stream_mode=None):
            raise Exception("Graph execution failed")

        with patch.object(agent, 'graph') as mock_graph:
            mock_graph.stream.side_effect = mock_stream_with_exception
            
            request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "Test query"}]
            )
            
            response = agent.predict(request)
            
            assert isinstance(response, ResponsesAgentResponse)
            assert len(response.output) == 1
            
            output_item = response.output[0]
            response_text = output_item.content[0]["text"].lower()
            assert "streaming error" in response_text or "error" in response_text
    
    def test_extract_text_content_handles_none(self):
        """Test that _extract_text_content handles None input."""
        agent = DatabricksResearchAgent()
        
        result = agent._extract_text_content(None)
        
        assert result == "None"  # Fallback string conversion


# NOTE: TestLangGraphResponsesAgentStreaming class removed  
# Streaming tests are properly implemented in test_streaming_schema.py
# for the RefactoredResearchAgent architecture

'''
class TestLangGraphResponsesAgentStreaming:
    """Test streaming functionality in LangGraphResponsesAgent."""
    
    @pytest.mark.skip(reason="Streaming test requires refactoring for RefactoredResearchAgent architecture")
    def test_predict_stream_with_structured_content(self):
        """Test streaming with structured content from LangGraph."""
        
        # Mock the combined agent
        mock_agent = Mock()
        
        # Mock streaming events with structured content
        mock_events = [
            ("updates", {
                "node1": {
                    "messages": [
                        AIMessage(content=[
                            {"type": "text", "text": "Streaming "},
                            {"type": "text", "text": "response"}
                        ])
                    ]
                }
            }),
            ("messages", [
                AIMessageChunk(content="chunk text", id="chunk-1")
            ])
        ]
        
        mock_agent.stream.return_value = iter(mock_events)
        mock_agent_creator.return_value = mock_agent
        
        agent = LangGraphResponsesAgent(mock_agent)
        
        request = ResponsesAgentRequest(
            input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "output_text", "text": "Test query"}]
                    }
                ]
        )
        
        events = list(agent.predict_stream(request))
        
        # Should have events from both updates and messages
        assert len(events) >= 1
        
        # Check that text was properly extracted
        for event in events:
            if event.type == "response.output_item.done":
                assert isinstance(event.item, dict)
                assert "content" in event.item
'''