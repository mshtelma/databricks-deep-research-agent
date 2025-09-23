"""
Pytest configuration for research agent tests.

NEW TESTING APPROACH (Post-TEST_MODE removal):
- Uses real services (Databricks LLMs, Brave Search) with test-optimized configurations
- Dependency injection through config_override parameter instead of environment flags
- Integration tests that validate actual functionality rather than mock behavior
- Significantly reduced mock usage - only for specific component isolation
- TEST_MODE environment variable is no longer used or respected

Legacy mock fixtures are maintained for backward compatibility but new tests
should use the integration_config fixtures for real service testing.
"""

import pytest
import os
import sys
import yaml
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional, Union

# Add parent directory to path for imports  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deep_research_agent_dir = os.path.join(parent_dir, 'deep_research_agent')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if deep_research_agent_dir not in sys.path:
    sys.path.insert(0, deep_research_agent_dir)

# No longer set TEST_MODE - agents now use dependency injection for testing
# Tests provide configuration overrides instead of environment flags

from deep_research_agent.core import (
    AgentConfiguration,
    SearchResult,
    Citation,
    ResearchContext,
    WorkflowMetrics,
    SearchResultType,
    ToolType
)
from deep_research_agent.components import create_tool_registry

# Import integration test fixtures
from tests.integration_config.test_fixtures import (
    enhanced_agent, databricks_agent, coordinator_agent, 
    planner_agent, researcher_agent, fact_checker_agent, reporter_agent,
    assert_real_search_results, assert_real_report_content,
    skip_if_no_api_key, get_simple_test_query, get_complex_test_query
)

# Import modern agent classes
try:
    from deep_research_agent.enhanced_research_agent import EnhancedResearchAgent
    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
    from deep_research_agent.databricks_helper import get_workspace_client, create_mock_workspace_client
    MODERN_AGENTS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Modern agents not available: {e}")
    MODERN_AGENTS_AVAILABLE = False


def load_test_config():
    """Load test configuration optimized for real services with test settings."""
    test_config_path = Path(__file__).parent / "integration_config" / "test_agent_config.yaml"
    if test_config_path.exists():
        with open(test_config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return minimal real test config - no TEST_MODE
        return {
            "llm": {
                "primary_endpoint": "databricks-gpt-oss-120b",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "search": {
                "providers": [{"type": "brave", "enabled": True}],
                "rate_limiting": {"max_concurrent_searches": 2}
            },
            "research": {
                "max_research_loops": 1,
                "initial_query_count": 2
            },
            "workflow": {
                "enable_background_investigation": True,
                "auto_accept_plan": True
            }
        }


@pytest.fixture(scope="session")
def test_config_yaml():
    """Provide test configuration from YAML file."""
    return load_test_config()


@pytest.fixture(scope="session")
def test_config(test_config_yaml):
    """Provide test configuration for all tests - uses real services with test settings."""
    return test_config_yaml


@pytest.fixture
def config_manager(test_config):
    """Provide configured ConfigManager instance."""
    from deep_research_agent.core.unified_config import get_config_manager
    return get_config_manager(override_config=test_config)


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


# New fixtures for modern agent architecture

@pytest.fixture(scope="session")
def mock_workspace_client():
    """Provide mock Databricks workspace client."""
    return create_mock_workspace_client()


@pytest.fixture
def enhanced_agent_config():
    """Provide configuration for EnhancedResearchAgent."""
    return {
        "multi_agent": {"enabled": True},
        "planning": {
            "enable_iterative_planning": True,
            "max_plan_iterations": 2,
            "plan_quality_threshold": 0.7,
            "auto_accept_plan": True
        },
        "background_investigation": {"enabled": True},
        "grounding": {
            "enabled": True,
            "verification_level": "moderate"
        },
        "report": {"default_style": "professional"},
        "reflexion": {"enabled": True}
    }


@pytest.fixture
def mock_enhanced_agent(enhanced_agent_config):
    """Provide mock EnhancedResearchAgent for testing."""
    if not MODERN_AGENTS_AVAILABLE:
        pytest.skip("Modern agents not available")
    
    # Create mock LLM
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Test response"))
    
    # Create mock tools
    mock_tools = {
        "search": [Mock(execute=Mock(return_value=[]))]
    }
    
    # Create agent with mocks
    agent = EnhancedResearchAgent(
        config_path=None,
        llm=mock_llm,
        tool_registry=mock_tools,
        **enhanced_agent_config
    )
    
    return agent


@pytest.fixture
def mock_databricks_agent():
    """Provide mock DatabricksCompatibleAgent for testing."""
    if not MODERN_AGENTS_AVAILABLE:
        pytest.skip("Modern agents not available")
    
    # Create a simple test config
    test_config = {
        "models": {"default": {"endpoint": "test-endpoint"}},
        "research": {"max_research_loops": 1}
    }
    
    agent = DatabricksCompatibleAgent(config=test_config)
    
    # Mock the underlying agent's graph to avoid actual execution
    agent.agent.graph = Mock()
    agent.agent.graph.ainvoke = AsyncMock(return_value={
        "final_report": "Test report content",
        "citations": [],
        "factuality_score": 0.8
    })
    
    return agent


@pytest.fixture
def agent_factory():
    """Factory fixture to create different types of agents."""
    def _create_agent(agent_type="databricks", **kwargs):
        if agent_type == "enhanced":
            return mock_enhanced_agent(**kwargs) if MODERN_AGENTS_AVAILABLE else None
        elif agent_type == "databricks":
            return mock_databricks_agent(**kwargs) if MODERN_AGENTS_AVAILABLE else None
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    return _create_agent


@pytest.fixture
def deprecation_warner():
    """Fixture to check for deprecation warnings."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


# High-level mocking patterns for multi-agent tests
@pytest.fixture
def mock_multi_agent_response():
    """Provide mock response for multi-agent workflow."""
    return {
        "final_report": "# Test Report\n\nThis is a test report with findings.",
        "factuality_score": 0.85,
        "citations": [
            {"source": "example.com", "title": "Test Source", "url": "https://example.com"}
        ],
        "plan": {
            "steps": [
                {"description": "Research step 1", "status": "completed"},
                {"description": "Research step 2", "status": "completed"}
            ]
        },
        "observations": [
            "Found relevant information about the topic",
            "Verified facts with multiple sources"
        ],
        "grounding_report": {
            "verification_level": "moderate",
            "verified_claims": 5,
            "contradictions": 0
        }
    }