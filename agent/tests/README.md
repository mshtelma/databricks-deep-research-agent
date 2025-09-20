# Testing Guide for Databricks Deep Research Agent

This guide provides comprehensive information about testing the Deep Research Agent, including patterns, best practices, and migration guidance.

## Overview

The test suite has been fully migrated to use the modern agent architecture:

- **Primary Agent**: `DatabricksCompatibleAgent` (recommended for most tests)
- **Alternative**: `EnhancedResearchAgent` (for multi-agent specific tests)
- **Legacy**: `RefactoredResearchAgent` (deprecated, marked for removal)

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit tests/         # Unit tests only
pytest -m integration tests/  # Integration tests only
pytest -m external tests/     # External service tests (requires API keys)

# Run with real APIs (requires environment variables)
BRAVE_API_KEY=your_key pytest tests/test_agent_brave_integration.py -v

# Run in CI mode (full mocking)
CI=true pytest tests/ -v
```

## Test Architecture

### Test Categories

1. **Unit Tests** (`-m unit`)
   - Fast, isolated tests with full mocking
   - No external dependencies
   - Test individual components and functions

2. **Integration Tests** (`-m integration`) 
   - Test component interactions
   - Mock external services but allow internal communication
   - Test agent workflows and state management

3. **External Tests** (`-m external`)
   - Require real external services (Brave API, Databricks)
   - Run only when explicitly requested
   - Test end-to-end functionality

### Agent Testing Patterns

#### DatabricksCompatibleAgent (Recommended)

```python
def test_with_databricks_agent():
    """Standard pattern for testing with DatabricksCompatibleAgent."""
    agent = DatabricksCompatibleAgent(config={
        "models": {"default": {"endpoint": "test-endpoint"}},
        "research": {"max_research_loops": 1}
    })
    
    # High-level mocking at graph level
    with patch.object(agent.agent, 'graph') as mock_graph:
        mock_graph.ainvoke.return_value = {
            "final_report": "Test response",
            "citations": [],
            "factuality_score": 0.8
        }
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test query"}]
        )
        response = agent.predict(request)
        
        assert isinstance(response, ResponsesAgentResponse)
```

#### EnhancedResearchAgent

```python
def test_with_enhanced_agent():
    """Pattern for testing multi-agent specific functionality."""
    agent = EnhancedResearchAgent(
        config_path=None,
        llm=mock_llm,
        tool_registry=mock_tools,
        **enhanced_agent_config
    )
    
    # Mock at graph level for multi-agent workflows
    with patch.object(agent, 'graph') as mock_graph:
        mock_graph.ainvoke.return_value = mock_multi_agent_response
        # Test multi-agent specific functionality
```

### Configuration Management

#### Test Configuration Files

1. **tests/test_config.yaml** - Central test configuration
2. **conftest.py** - Global fixtures and test utilities
3. **Environment Variables** - Runtime overrides

#### Configuration Priorities

1. Environment variables (highest)
2. Test-specific overrides
3. `tests/test_config.yaml`
4. Default test configuration (lowest)

#### Key Environment Variables

```bash
# Test mode (enables mocking)
TEST_MODE=true

# API keys for external tests  
BRAVE_API_KEY=your_brave_key

# Databricks configuration
DATABRICKS_HOST=https://your.databricks.com
DATABRICKS_TOKEN=your_token

# CI detection (enables full mocking)
CI=true
GITHUB_ACTIONS=true
```

## Common Test Patterns

### 1. High-Level Mocking (Recommended)

```python
# ✅ Correct: Mock at the graph level
with patch.object(agent.agent, 'graph') as mock_graph:
    mock_graph.ainvoke.return_value = expected_response
    result = agent.predict(request)
```

### 2. Multi-Agent Response Mocking

```python
# Use the mock_multi_agent_response fixture
def test_multi_agent_workflow(mock_multi_agent_response):
    with patch.object(agent.agent, 'graph') as mock_graph:
        mock_graph.ainvoke.return_value = mock_multi_agent_response
        # Test multi-agent functionality
```

### 3. Streaming Tests

```python
def test_streaming():
    async def mock_astream(initial_state, stream_mode=None):
        for event in mock_stream_events:
            yield event
    
    mock_graph.astream = mock_astream
    events = list(agent.predict_stream(request))
    assert len(events) > 0
```

### 4. Error Handling Tests

```python
def test_error_handling():
    with patch.object(agent.agent, 'graph') as mock_graph:
        mock_graph.ainvoke.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            agent.predict(request)
```

## Fixtures and Utilities

### Global Fixtures (from conftest.py)

- `mock_databricks_agent()` - Ready-to-use DatabricksCompatibleAgent
- `mock_enhanced_agent()` - Ready-to-use EnhancedResearchAgent  
- `mock_workspace_client()` - Mock Databricks workspace client
- `mock_multi_agent_response()` - Standard multi-agent response
- `test_config_yaml()` - Test configuration from YAML
- `agent_factory()` - Factory for creating different agent types

### Test Utilities

```python
# Create test requests
request = create_test_request("Test query")

# Validate responses
assert_valid_search_result(search_result)
assert_valid_citation(citation)
assert_valid_research_context(context)

# Error simulation
mock_error.network_error()  # Simulate network issues
mock_error.api_error(500)   # Simulate API errors
```

## Migration from Legacy Tests

### Deprecated Patterns

```python
# ❌ Old: RefactoredResearchAgent (deprecated)
from deep_research_agent.research_agent_refactored import RefactoredResearchAgent

# ❌ Old: Low-level LLM mocking (causes ordering issues)
with patch.object(ChatDatabricks, 'ainvoke') as mock_llm:
    mock_llm.side_effect = [response1, response2]  # Wrong order!

# ❌ Old: create_combined_agent (doesn't exist)
@patch('deep_research_agent.research_agent_refactored.create_combined_agent')
```

### Modern Patterns

```python
# ✅ New: DatabricksCompatibleAgent
from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent

# ✅ New: High-level graph mocking
with patch.object(agent.agent, 'graph') as mock_graph:
    mock_graph.ainvoke.return_value = complete_response

# ✅ New: Direct agent instantiation
agent = DatabricksCompatibleAgent(config=test_config)
```

## Troubleshooting

### Common Issues

1. **Tests Skip Due to Missing Databricks Connection**
   ```bash
   # Solution: Enable test mode
   TEST_MODE=true pytest tests/
   ```

2. **Tests Skip Due to Missing API Keys**
   ```bash
   # Solution: Provide keys or run without external tests
   pytest tests/ -m "not external"
   ```

3. **Mock Ordering Issues**
   ```python
   # Problem: Low-level mocking with side_effect
   # Solution: Use high-level graph mocking instead
   ```

4. **Import Errors for Modern Agents**
   ```python
   # Problem: Agent classes not found
   # Solution: Check that enhanced/databricks agents are available
   if not MODERN_AGENTS_AVAILABLE:
       pytest.skip("Modern agents not available")
   ```

### Running Tests in Different Environments

#### Local Development
```bash
# Full test suite with some real services
pytest tests/ -v
```

#### Continuous Integration
```bash
# Full mocking, no external dependencies
CI=true pytest tests/ -v --tb=short
```

#### Integration Testing
```bash
# Real APIs, requires credentials
BRAVE_API_KEY=key DATABRICKS_TOKEN=token pytest tests/ -m integration -v
```

#### Performance Testing
```bash
# Longer timeouts, performance metrics
pytest tests/ -m slow -v --timeout=300
```

## Writing New Tests

### Template for New Tests

```python
"""Test module for new functionality."""

import pytest
from unittest.mock import patch
from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
from mlflow.types.responses import ResponsesAgentRequest


class TestNewFunctionality:
    """Test suite for new functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality with DatabricksCompatibleAgent."""
        agent = DatabricksCompatibleAgent(config={
            "models": {"default": {"endpoint": "test-endpoint"}},
            "research": {"max_research_loops": 1}
        })
        
        with patch.object(agent.agent, 'graph') as mock_graph:
            mock_graph.ainvoke.return_value = {
                "final_report": "Expected result",
                "citations": [],
                "factuality_score": 0.8
            }
            
            request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "Test input"}]
            )
            
            response = agent.predict(request)
            
            # Add your assertions here
            assert response is not None
    
    @pytest.mark.integration
    def test_integration_scenario(self, mock_databricks_agent):
        """Test integration scenario using fixture."""
        # Use pre-configured agent from fixture
        response = mock_databricks_agent.predict(test_request)
        
        # Add integration-specific assertions
        assert response is not None
    
    @pytest.mark.external
    def test_with_real_services(self):
        """Test with real external services (requires API keys)."""
        # This test will be skipped unless external marker is included
        # and required environment variables are set
        pass
```

### Best Practices

1. **Use High-Level Mocking**: Mock at the graph level, not individual components
2. **Test Edge Cases**: Include error conditions and boundary cases
3. **Use Fixtures**: Leverage existing fixtures for common setup
4. **Mark Tests Appropriately**: Use `@pytest.mark.unit|integration|external`
5. **Test Both Sync and Async**: Test both `predict()` and `predict_stream()`
6. **Validate Response Structure**: Check MLflow ResponsesAgent compliance
7. **Clean Up**: Use appropriate fixtures for cleanup
8. **Document Intent**: Clear docstrings explaining what each test validates

## Performance and Reliability

### Test Performance Guidelines

- Unit tests: < 100ms each
- Integration tests: < 5s each  
- External tests: < 30s each
- Full test suite: < 5 minutes

### Reliability Patterns

1. **Deterministic Mocking**: Avoid random behavior in mocks
2. **Cleanup After Tests**: Use `autouse` fixtures for cleanup
3. **Isolation**: Tests should not depend on each other
4. **Timeout Handling**: Set appropriate timeouts for external calls
5. **Retry Logic**: Test retry behavior under failure conditions

## Contributing

When adding new tests:

1. Follow the established patterns in this guide
2. Add appropriate markers (`unit`, `integration`, `external`)
3. Update this documentation if introducing new patterns
4. Ensure tests pass in both local and CI environments
5. Add fixtures to `conftest.py` if they'll be reused

## See Also

- [Multi-Agent Architecture Documentation](../docs/MULTI_AGENT_ARCHITECTURE.md)
- [Testing Architecture](../docs/TESTING_ARCHITECTURE.md)
- [Development Setup](../README.md)