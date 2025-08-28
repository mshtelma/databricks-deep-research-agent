"""
Tests for the refactored research agent with improved architecture.

This module provides comprehensive tests for the new agent implementation
using the core libraries and components.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.types.llm import ChatMessage

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

from deep_research_agent.core import (
    ConfigManager,
    AgentConfiguration,
    SearchResult,
    Citation,
    ResearchContext,
    WorkflowMetrics,
    SearchResultType,
    get_logger
)
from deep_research_agent.core.model_manager import ModelConfig, ModelRole, ModelManager
from deep_research_agent.components import (
    message_converter,
    content_extractor,
    create_tool_registry,
    response_builder
)


class TestConfigManager:
    """Test the configuration management system."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        # Patch load_yaml_config to return empty dict, ensuring we test true defaults
        with patch.object(ConfigManager, 'load_yaml_config', return_value={}):
            config_manager = ConfigManager()
            agent_config = config_manager.get_agent_config()
            
            assert isinstance(agent_config, AgentConfiguration)
            assert agent_config.llm_endpoint == "databricks-claude-3-7-sonnet"
            assert agent_config.max_research_loops == 2
            assert agent_config.temperature == 0.7
    
    def test_config_override(self):
        """Test configuration override functionality."""
        override_config = {
            "llm_endpoint": "test-endpoint",
            "max_research_loops": 5,
            "temperature": 0.5
        }
        
        config_manager = ConfigManager(override_config)
        agent_config = config_manager.get_agent_config()
        
        assert agent_config.llm_endpoint == "test-endpoint"
        assert agent_config.max_research_loops == 5
        assert agent_config.temperature == 0.5
    
    def test_config_validation(self):
        """Test configuration validation."""
        invalid_config = {
            "max_research_loops": -1,  # Invalid
            "temperature": 5.0,  # Invalid
            "max_tokens": -100  # Invalid
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            config_manager = ConfigManager(invalid_config)
            config_manager.get_agent_config()


class TestMessageConverter:
    """Test message conversion functionality."""
    
    def test_extract_text_content_string(self):
        """Test extracting text from string content."""
        content = "Simple text content"
        result = message_converter.extract_text_content(content)
        assert result == "Simple text content"
    
    def test_extract_text_content_list(self):
        """Test extracting text from list content."""
        content = [
            {"type": "text", "text": "Part 1 "},
            {"type": "text", "text": "Part 2 "},
            "Part 3"
        ]
        result = message_converter.extract_text_content(content)
        assert result == "Part 1 Part 2 Part 3"
    
    def test_extract_text_content_dict(self):
        """Test extracting text from dict content."""
        content = {"text": "Dictionary content"}
        result = message_converter.extract_text_content(content)
        assert result == "Dictionary content"
    
    def test_responses_to_langchain_conversion(self):
        """Test converting ResponsesAgent messages to LangChain format."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "You are helpful"}
        ]
        
        result = message_converter.responses_to_langchain(messages)
        
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], SystemMessage)
        assert result[0].content == "Hello"
    
    def test_langchain_to_responses_conversion(self):
        """Test converting LangChain messages to ResponsesAgent format."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            SystemMessage(content="You are helpful")
        ]
        
        result = message_converter.langchain_to_responses(messages)
        
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "system"


class TestContentExtractor:
    """Test content extraction functionality."""
    
    def test_extract_search_results_web(self):
        """Test extracting web search results."""
        raw_results = [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "content": "Article content here",
                "score": 0.9
            },
            {
                "title": "Another Article",
                "url": "https://example.com/another",
                "content": "More content",
                "score": 0.8
            }
        ]
        
        results = content_extractor.extract_search_results(raw_results, SearchResultType.WEB)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].title == "Test Article"
        assert results[0].url == "https://example.com/article"
        assert results[0].result_type == SearchResultType.WEB
    
    def test_extract_citations(self):
        """Test extracting citations from search results."""
        search_results = [
            SearchResult(
                content="Content 1",
                source="example.com",
                url="https://example.com/1",
                title="Article 1",
                result_type=SearchResultType.WEB
            ),
            SearchResult(
                content="Content 2",
                source="internal",
                title="Internal Doc",
                result_type=SearchResultType.VECTOR
            )
        ]
        
        citations = content_extractor.extract_citations(search_results)
        
        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)
        assert citations[0].url == "https://example.com/1"
        assert citations[1].url is None
    
    def test_clean_content(self):
        """Test content cleaning functionality."""
        dirty_content = "  Content with   extra   spaces  [Advertisement] and unwanted stuff  "
        clean_content = content_extractor.clean_content(dirty_content)
        
        assert "Advertisement" not in clean_content
        # Check that multiple spaces are reduced (allow for some multiple spaces after cleaning)
        assert clean_content.count("   ") == 0  # No triple spaces
        assert clean_content.strip() == clean_content  # No leading/trailing spaces


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_tool_registry_creation(self):
        """Test creating tool registry."""
        config_manager = ConfigManager()
        registry = create_tool_registry(config_manager)
        
        assert registry is not None
        assert len(registry.factories) > 0
    
    @patch('deep_research_agent.components.tool_manager.DATABRICKS_AVAILABLE', False)
    def test_tool_registry_without_databricks(self):
        """Test tool registry when Databricks is not available."""
        config_manager = ConfigManager()
        registry = create_tool_registry(config_manager)
        
        # Should still work but some tools may not be available
        status = registry.get_tool_status()
        assert "total_registered" in status
    
    def test_tool_validation(self):
        """Test tool configuration validation."""
        config_manager = ConfigManager()
        registry = create_tool_registry(config_manager)
        
        validation_results = registry.validate_all_configurations()
        
        assert isinstance(validation_results, dict)
        # Should have validation results for each tool type


class TestResponseBuilder:
    """Test response builder functionality."""
    
    # Note: test_create_text_output_item and test_create_text_delta removed
    # These methods were removed from ResponseBuilder as they should be
    # inherited from ResponsesAgent in classes that need them
    
    def test_build_research_response(self):
        """Test building complete research response."""
        final_answer = "This is the final answer based on research."
        
        research_context = ResearchContext(
            original_question="What is AI?",
            citations=[
                Citation(source="example.com", url="https://example.com", title="AI Article")
            ],
            research_loops=1
        )
        
        metrics = WorkflowMetrics(
            total_queries_generated=2,
            total_web_results=5,
            execution_time_seconds=10.5,
            success_rate=1.0
        )
        
        response = response_builder.build_research_response(
            final_answer, research_context, metrics
        )
        
        assert isinstance(response, ResponsesAgentResponse)
        assert len(response.output) == 1
        # Check that response has the expected structure
        output_item = response.output[0]
        # Output items are OutputItem objects, not dicts
        assert hasattr(output_item, 'type')
        assert output_item.type == 'message'
        assert hasattr(output_item, 'content')
        assert response.custom_outputs is not None
        assert "citations" in response.custom_outputs
        assert "research_metadata" in response.custom_outputs
    
    def test_build_error_response(self):
        """Test building error responses."""
        error_message = "Something went wrong"
        response = response_builder.build_error_response(error_message)
        
        assert isinstance(response, ResponsesAgentResponse)
        assert len(response.output) == 1
        # Check that error response has expected structure
        output_item = response.output[0]
        assert hasattr(output_item, 'type')
        assert output_item.type == 'message'
        assert response.custom_outputs is not None
        assert response.custom_outputs["error"] == True
    
    def test_format_citations_text(self):
        """Test formatting citations as text."""
        citations = [
            Citation(source="example.com", url="https://example.com", title="Article 1"),
            Citation(source="internal", title="Internal Doc")
        ]
        
        citations_text = response_builder.format_citations_text(citations)
        
        assert "Sources:" in citations_text
        assert "1. example.com" in citations_text
        assert "2. internal" in citations_text
        assert "https://example.com" in citations_text


class TestRefactoredAgentIntegration:
    """Integration tests for the refactored agent."""
    
    @patch('deep_research_agent.agent_initialization.ChatDatabricks')
    @patch('deep_research_agent.agent_initialization.DATABRICKS_AVAILABLE', True)
    def test_agent_initialization(self, mock_chat):
        """Test agent initialization with mocks."""
        mock_llm = Mock()
        mock_chat.return_value = mock_llm
        
        from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
        
        config = {
            "llm_endpoint": "test-endpoint",
            "max_research_loops": 1,
            "tavily_api_key": "test-key"
        }
        
        agent = RefactoredResearchAgent(config)
        
        assert agent is not None
        assert agent.agent_config.llm_endpoint == "test-endpoint"
        assert agent.agent_config.max_research_loops == 1
    
    @patch('deep_research_agent.agent_initialization.ChatDatabricks')
    @patch('deep_research_agent.agent_initialization.DATABRICKS_AVAILABLE', True)
    def test_agent_predict_simple(self, mock_chat):
        """Test simple prediction with mocked components."""
        # Setup mocks
        mock_llm = Mock()
        mock_response = AIMessage(content="Test response")
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
        
        # Create agent
        config = {"max_research_loops": 1}
        agent = RefactoredResearchAgent(config)
        
        # Mock the graph to return a simple response
        with patch.object(agent, 'graph') as mock_graph:
            # Mock stream to yield messages in the expected format
            def mock_stream(initial_state, stream_mode=None):
                # Return message chunks as tuples in messages mode format
                from unittest.mock import Mock
                yield ("messages", [Mock(content="Test response")])
            
            mock_graph.stream.return_value = mock_stream(None)
            
            # Create request using dict format instead of ChatMessage
            request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "What is AI?"}]
            )
            
            # Test prediction
            response = agent.predict(request)
            
            assert isinstance(response, ResponsesAgentResponse)
            assert len(response.output) == 1
    
    @patch('deep_research_agent.agent_initialization.ChatDatabricks')
    @patch('deep_research_agent.agent_initialization.DATABRICKS_AVAILABLE', True)
    def test_model_manager_integration(self, mock_chat):
        """Test ModelManager integration to prevent get_node_config AttributeError."""
        mock_llm = Mock()
        mock_response = AIMessage(content="query1\nquery2\nquery3")
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
        
        # Create agent with model manager
        agent = RefactoredResearchAgent()
        agent.model_manager = ModelManager()
        
        # Test that _generate_search_queries works with model_manager
        # This should NOT raise AttributeError about get_node_config
        queries = agent._generate_search_queries("What is machine learning?")
        
        # Verify the method completed successfully
        assert isinstance(queries, list)
        assert len(queries) > 0
        
        # Verify the model manager config is accessible via correct API
        model_config = agent.model_manager.config.get_model_for_role(ModelRole.QUERY_GENERATION)
        assert model_config is not None
        assert isinstance(model_config, ModelConfig)
        assert hasattr(model_config, 'endpoint')
        assert isinstance(model_config.endpoint, str)


class TestHealthChecksAndRobustness:
    """Test health checks and robustness features."""
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker behavior."""
        from core.utils import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Function that always fails
        def failing_function():
            raise Exception("Always fails")
        
        # First two calls should fail and increment failure count
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)
        
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)
        
        # Third call should trigger circuit breaker
        from core.exceptions import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(failing_function)
    
    def test_retry_with_backoff(self):
        """Test retry with exponential backoff."""
        from core.utils import retry_with_exponential_backoff
        
        call_count = 0
        
        @retry_with_exponential_backoff(max_retries=2)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        from core.config import validate_configuration
        
        # Valid configuration
        valid_config = {
            "llm_endpoint": "test-endpoint",
            "max_research_loops": 2,
            "temperature": 0.7
        }
        
        issues = validate_configuration(valid_config)
        assert isinstance(issues, list)
        
        # Invalid configuration
        invalid_config = {
            "max_research_loops": -1,
            "temperature": 10.0
        }
        
        issues = validate_configuration(invalid_config)
        assert len(issues) > 0


# Fixtures for test data
@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return [
        SearchResult(
            content="AI is transforming industries",
            source="tech.com",
            url="https://tech.com/ai-article",
            title="AI Revolution",
            score=0.9,
            result_type=SearchResultType.WEB
        ),
        SearchResult(
            content="Machine learning basics",
            source="edu.org",
            url="https://edu.org/ml-basics",
            title="ML Fundamentals",
            score=0.8,
            result_type=SearchResultType.WEB
        )
    ]


@pytest.fixture
def sample_research_context():
    """Provide sample research context for testing."""
    return ResearchContext(
        original_question="What is artificial intelligence?",
        research_loops=1,
        max_loops=2
    )


@pytest.fixture
def mock_config_manager():
    """Provide mocked configuration manager."""
    config = {
        "llm_endpoint": "test-endpoint",
        "max_research_loops": 1,
        "initial_query_count": 2,
        "tavily_api_key": "test-key"
    }
    return ConfigManager(config)


# Test utilities
def create_mock_responses_agent_request(content: str) -> ResponsesAgentRequest:
    """Create a mock ResponsesAgentRequest for testing."""
    return ResponsesAgentRequest(
        input=[ChatMessage(role="user", content=content)]
    )


def assert_valid_response(response: ResponsesAgentResponse):
    """Assert that a response is valid."""
    assert isinstance(response, ResponsesAgentResponse)
    assert response.output is not None
    assert len(response.output) > 0
    assert "content" in response.output[0] or "type" in response.output[0]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])