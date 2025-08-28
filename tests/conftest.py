"""
Pytest configuration for refactored agent tests.

This module provides common fixtures and configuration for testing
the refactored research agent with improved architecture.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Add parent directory to path for imports  
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
    ToolType
)
from deep_research_agent.components import create_tool_registry


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests."""
    return {
        "llm_endpoint": "test-databricks-endpoint",
        "max_research_loops": 1,
        "initial_query_count": 2,
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout_seconds": 10,
        "max_retries": 1,
        "tavily_api_key": "test-tavily-key",
        "vector_search_index": "test.schema.index"
    }


@pytest.fixture
def config_manager(test_config):
    """Provide configured ConfigManager instance."""
    return ConfigManager(test_config)


@pytest.fixture
def agent_config(config_manager):
    """Provide AgentConfiguration instance."""
    return config_manager.get_agent_config()


@pytest.fixture
def mock_llm():
    """Provide mocked LLM for testing."""
    mock = Mock()
    
    # Default responses for different types of prompts
    def mock_invoke(messages):
        last_message = messages[-1] if messages else None
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        if "generate" in content.lower() and "queries" in content.lower():
            # Query generation response
            from langchain_core.messages import AIMessage
            return AIMessage(content='{"queries": ["query 1", "query 2"]}')
        elif "reflect" in content.lower() or "sufficient" in content.lower():
            # Reflection response
            from langchain_core.messages import AIMessage
            return AIMessage(content='{"needs_more_research": false, "reflection": "Good research coverage"}')
        else:
            # Default synthesis response
            from langchain_core.messages import AIMessage
            return AIMessage(content="Based on the research, here is a comprehensive answer.")
    
    mock.invoke.side_effect = mock_invoke
    return mock


@pytest.fixture
def mock_tavily_tool():
    """Provide mocked Tavily search tool."""
    mock = Mock()
    
    def mock_search(query, max_results=5):
        return [
            {
                "title": f"Result for {query}",
                "url": f"https://example.com/search/{query.replace(' ', '-')}",
                "content": f"Content about {query} with detailed information.",
                "score": 0.9
            },
            {
                "title": f"Another result for {query}",
                "url": f"https://example.org/article/{query.replace(' ', '-')}",
                "content": f"Additional content about {query}.",
                "score": 0.8
            }
        ]
    
    mock.search.side_effect = mock_search
    return mock


@pytest.fixture
def mock_vector_tool():
    """Provide mocked vector search tool."""
    from langchain_core.documents import Document
    
    mock = Mock()
    
    def mock_get_relevant_documents(query):
        return [
            Document(
                page_content=f"Internal document about {query}",
                metadata={
                    "source": "internal_db",
                    "title": f"Internal: {query}",
                    "score": 0.85
                }
            ),
            Document(
                page_content=f"Another internal document related to {query}",
                metadata={
                    "source": "knowledge_base", 
                    "title": f"KB: {query}",
                    "score": 0.75
                }
            )
        ]
    
    mock.get_relevant_documents.side_effect = mock_get_relevant_documents
    return mock


@pytest.fixture
def mock_tool_registry(mock_tavily_tool, mock_vector_tool):
    """Provide mocked tool registry."""
    mock_registry = Mock()
    
    def mock_get_tool(tool_type):
        if tool_type == ToolType.TAVILY_SEARCH:
            return mock_tavily_tool
        elif tool_type == ToolType.VECTOR_SEARCH:
            return mock_vector_tool
        else:
            return None
    
    mock_registry.get_tool.side_effect = mock_get_tool
    mock_registry.get_all_tools.return_value = {
        ToolType.TAVILY_SEARCH: mock_tavily_tool,
        ToolType.VECTOR_SEARCH: mock_vector_tool
    }
    
    return mock_registry


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return [
        SearchResult(
            content="Artificial Intelligence (AI) is transforming industries worldwide.",
            source="tech-news.com",
            url="https://tech-news.com/ai-revolution",
            title="The AI Revolution",
            score=0.95,
            result_type=SearchResultType.WEB,
            metadata={"domain": "tech-news.com"}
        ),
        SearchResult(
            content="Machine learning is a subset of AI that enables systems to learn.",
            source="edu-portal.org",
            url="https://edu-portal.org/ml-basics",
            title="Machine Learning Fundamentals",
            score=0.87,
            result_type=SearchResultType.WEB,
            metadata={"domain": "edu-portal.org"}
        ),
        SearchResult(
            content="Internal documentation about AI implementation guidelines.",
            source="internal_docs",
            title="AI Implementation Guide",
            score=0.82,
            result_type=SearchResultType.VECTOR,
            metadata={"source": "internal", "department": "engineering"}
        )
    ]


@pytest.fixture
def sample_citations():
    """Provide sample citations for testing."""
    return [
        Citation(
            source="tech-news.com",
            url="https://tech-news.com/ai-revolution",
            title="The AI Revolution",
            snippet="AI is transforming industries worldwide..."
        ),
        Citation(
            source="edu-portal.org", 
            url="https://edu-portal.org/ml-basics",
            title="Machine Learning Fundamentals",
            snippet="Machine learning is a subset of AI..."
        ),
        Citation(
            source="internal_docs",
            title="AI Implementation Guide",
            snippet="Internal documentation about AI implementation..."
        )
    ]


@pytest.fixture
def sample_research_context(sample_search_results, sample_citations):
    """Provide sample research context."""
    context = ResearchContext(
        original_question="What is artificial intelligence and how does it work?",
        research_loops=1,
        max_loops=2
    )
    
    # Add search results
    for result in sample_search_results:
        if result.result_type == SearchResultType.WEB:
            context.web_results.append(result)
        else:
            context.vector_results.append(result)
    
    # Add citations
    context.citations = sample_citations
    context.reflection = "Research provides good coverage of AI fundamentals and applications."
    
    return context


@pytest.fixture
def sample_workflow_metrics():
    """Provide sample workflow metrics."""
    return WorkflowMetrics(
        total_queries_generated=3,
        total_web_results=5,
        total_vector_results=2,
        total_research_loops=1,
        execution_time_seconds=12.5,
        error_count=0,
        success_rate=1.0
    )


@pytest.fixture
def mock_databricks_available():
    """Mock Databricks availability for testing."""
    with patch('core.config.DATABRICKS_AVAILABLE', True):
        with patch('components.tool_manager.DATABRICKS_AVAILABLE', True):
            yield


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('core.logging.MLFLOW_AVAILABLE', True):
        mock_mlflow = Mock()
        with patch('core.logging.mlflow', mock_mlflow):
            yield mock_mlflow


# Test utilities
def create_test_messages(content: str):
    """Create test messages for LangChain format."""
    from langchain_core.messages import HumanMessage, AIMessage
    return [
        HumanMessage(content=content),
        AIMessage(content="Test response")
    ]


def create_test_request(content: str):
    """Create test request in ResponsesAgent format."""
    from mlflow.types.responses import ResponsesAgentRequest
    
    return ResponsesAgentRequest(
        input=[{"role": "user", "content": content}]
    )


def assert_valid_search_result(result):
    """Assert that a search result is valid."""
    assert isinstance(result, SearchResult)
    assert result.content is not None
    assert result.source is not None
    assert isinstance(result.score, (int, float))
    assert result.result_type in [SearchResultType.WEB, SearchResultType.VECTOR]


def assert_valid_citation(citation):
    """Assert that a citation is valid."""
    assert isinstance(citation, Citation)
    assert citation.source is not None
    # URL is optional for internal sources


def assert_valid_research_context(context):
    """Assert that a research context is valid."""
    assert isinstance(context, ResearchContext)
    assert context.original_question is not None
    assert isinstance(context.research_loops, int)
    assert isinstance(context.max_loops, int)
    assert isinstance(context.web_results, list)
    assert isinstance(context.vector_results, list)
    assert isinstance(context.citations, list)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup logic here
    pass


# Error simulation utilities
class MockError:
    """Utility class for simulating various error conditions."""
    
    @staticmethod
    def network_error():
        """Simulate network error."""
        import requests
        raise requests.exceptions.ConnectionError("Network connection failed")
    
    @staticmethod
    def timeout_error():
        """Simulate timeout error."""
        import requests
        raise requests.exceptions.Timeout("Request timed out")
    
    @staticmethod
    def api_error(status_code=500):
        """Simulate API error."""
        import requests
        response = Mock()
        response.status_code = status_code
        response.text = "Internal Server Error"
        error = requests.exceptions.HTTPError()
        error.response = response
        raise error


@pytest.fixture
def mock_error():
    """Provide mock error utilities."""
    return MockError


@pytest.fixture
def sample_request():
    """Provide sample request for testing."""
    return create_test_request("What are the latest AI trends in 2024?")