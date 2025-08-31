# Testing Guide for LangGraph MLflow ResponsesAgent

This directory contains comprehensive tests for the MLflow ResponsesAgent integration with LangGraph workflows.

## Overview

The test suite validates:
- **MLflow ResponsesAgent functionality** - Content extraction, response validation, streaming
- **LangGraph workflow execution** - Graph construction, node execution, state management  
- **End-to-end integration** - Complete pipelines, error handling, performance

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Shared fixtures and mocks
├── test_mlflow_responses_agent.py  # MLflow ResponsesAgent unit tests
├── test_refactored_agent.py        # RefactoredResearchAgent unit tests
├── test_streaming_schema.py        # Streaming schema validation tests
├── test_schema_compliance.py       # Comprehensive schema compliance tests (NEW)
├── test_integration.py             # End-to-end integration tests
├── test_brave_search.py            # Brave search provider tests
├── pytest.ini                      # Pytest configuration
└── README.md                       # This file
```

## Setup

### 1. Install Test Dependencies

```bash
# From the project root directory
pip install -r requirements-test.txt
```

### 2. Install Main Dependencies

```bash
# Install the main project dependencies
pip install -r requirements.txt
```

## Running Tests

### Fix the Import Issue First

Before running tests, ensure all dependencies are installed and the PYTHONPATH includes the project root directory.

### Run All Tests

```bash
# From the project root directory
pytest tests/ -v
```

### Run Integration Tests with External Services

To run tests that interact with real APIs (marked with `@pytest.mark.external`):

```bash
# Set environment variables for API keys
export BRAVE_API_KEY="your-brave-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Run external tests
pytest tests/ -m external -v

# Run all tests including external ones
pytest tests/ -v
```

**Prerequisites for external API tests:**
- Valid API keys for search providers (Brave, Tavily)
- Configured environment variables or MLflow secrets
- Network access to external APIs

### Run Specific Test Files

```bash
# Test MLflow functionality only
pytest tests/test_mlflow_responses_agent.py -v

# Test refactored agent functionality
pytest tests/test_refactored_agent.py -v

# Test integration scenarios only
pytest tests/test_integration.py -v
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest tests/test_mlflow_responses_agent.py::TestTextContentExtraction -v

# Run specific test method
pytest tests/test_integration.py::TestEndToEndIntegration::test_research_agent_full_pipeline -v
```

## Test Categories

### Unit Tests (`test_mlflow_responses_agent.py`)

- **Text Content Extraction**: Tests `_extract_text_content()` helper method with various input formats
- **Base ResponsesAgent Methods**: Validates inherited MLflow methods work correctly
- **Response Validation**: Ensures ResponsesAgentResponse and ResponsesAgentStreamEvent validation
- **Error Handling**: Tests graceful error recovery
- **Note**: Outdated tests for old architecture have been commented out

### Schema Compliance Tests (`test_schema_compliance.py`) - NEW

- **Custom Inputs Handling**: Tests custom_inputs parameter processing
- **Custom Outputs Validation**: Validates custom_outputs structure with citations and metrics
- **Error Scenarios**: Tests malformed requests, empty content, null values, JSON prevention
- **Advanced Streaming**: Tests concurrent requests, large content, multi-message conversations
- **Databricks Compatibility**: Tests DatabricksCompatibleAgent wrapper
- **Schema Validation**: Ensures compliance with all documented requirements

### Streaming Schema Tests (`test_streaming_schema.py`)

- **No JSON in Deltas**: Prevents regression of JSON leak issue
- **Event Schema Validation**: Tests delta and done event structures
- **Stream Integrity**: Validates content consistency between deltas and done events
- **ID Consistency**: Ensures item_id consistency across related events
- **Regression Tests**: Specific tests for known issues like "hi" query

### Agent Tests (`test_refactored_agent.py`)

- **Configuration Management**: Tests agent configuration and overrides
- **Message Conversion**: Tests conversion between different message formats
- **Tool Management**: Tests tool creation and validation
- **Response Building**: Tests response construction with citations
- **Circuit Breaker**: Tests error recovery mechanisms

### Integration Tests (`test_integration.py`)

- **End-to-End Pipelines**: Complete request→processing→response flows
- **Streaming Integration**: Tests streaming with content extraction
- **Query Classification**: Validates research vs. simple query routing
- **MLflow Validation**: Tests request/response validation edge cases
- **Error Recovery**: Tests system-wide error handling
- **Performance**: Tests concurrent requests and large contexts

## Key Testing Features

### Comprehensive Mocking

Tests use extensive mocking to avoid external dependencies:

- **Databricks LLMs**: Mocked ChatDatabricks responses
- **Search Tools**: Mocked Tavily search results  
- **Unity Catalog**: Mocked UC function toolkit
- **Environment Variables**: Mocked API keys and endpoints

### Realistic Test Data

- **Structured Content**: Tests both string and list-based message content
- **Tool Calls**: Validates function calling scenarios
- **Large Contexts**: Tests performance with extensive conversation history
- **Error Scenarios**: Covers API failures, network issues, malformed responses

### Content Extraction Validation

Critical tests for the content extraction fix:

```python
# Tests string content
content = "Simple string content"
result = agent._extract_text_content(content)
assert result == "Simple string content"

# Tests structured content  
content = [
    {"type": "text", "text": "Hello "},
    {"type": "text", "text": "world!"}
]
result = agent._extract_text_content(content)
assert result == "Hello world!"
```

### Schema Validation Testing

The test suite includes comprehensive schema validation to ensure compliance with MLflow ResponsesAgent and Databricks requirements:

#### Test File: `test_streaming_schema.py`

This test module validates critical schema requirements:

1. **No JSON in Delta Events**: Ensures delta events contain only plain text
2. **Event Type Validation**: Verifies correct event type strings
3. **Stream Integrity**: Validates consistent item IDs and proper done events
4. **Content Structure**: Tests proper message format in done events

```python
# Example: Testing for no JSON in delta events
def test_no_json_objects_in_stream():
    """Ensure no JSON objects from intermediate nodes appear in stream."""
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "test"}]
    )
    
    stream_events = list(agent.predict_stream(request))
    
    for event in stream_events:
        if event.type == "response.output_text.delta":
            # Verify delta is not JSON
            try:
                parsed = json.loads(event.delta)
                if isinstance(parsed, dict):
                    pytest.fail(f"JSON object found in delta: {parsed}")
            except json.JSONDecodeError:
                pass  # Expected - delta should be plain text
```

#### Running Schema Tests

```bash
# Run schema validation tests specifically
pytest tests/test_streaming_schema.py -v

# Run with detailed output
pytest tests/test_streaming_schema.py -v -s

# Check coverage for schema validation
pytest --cov=deep_research_agent.research_agent_refactored tests/test_streaming_schema.py
```

#### Schema Requirements Documentation

For complete schema specifications, see `SCHEMA_REQUIREMENTS.md` which covers:
- Request/Response schemas (ResponsesAgentRequest, ResponsesAgentResponse)
- Stream event schemas (ResponsesAgentStreamEvent)
- Critical requirements and common pitfalls
- Implementation examples and best practices

## Environment Variables for Testing

Tests use mocked environment variables, but you can set real ones for integration testing:

```bash
export GEMINI_API_KEY="your-key-here"
export TAVILY_API_KEY="your-key-here" 
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token-here"
```

## Common Issues and Solutions

### Import Errors

If you get import errors:

```bash
# Ensure you're in the project root directory
cd /path/to/databricks-deep-research-agent

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies

```bash
# Install all test dependencies
pip install -r requirements-test.txt

# If you get MLflow import errors
pip install mlflow>=2.10.0

# If you get LangGraph import errors  
pip install langgraph>=0.6.6
```

### Databricks Import Errors

Tests mock Databricks imports, but if you get import errors:

```bash
# Install databricks packages (optional for testing)
pip install databricks-langchain
pip install unitycatalog-ai[databricks]
```

## Testing Best Practices

### Writing New Tests

1. **Use Fixtures**: Leverage fixtures in `conftest.py` for common test data
2. **Mock External Services**: Always mock API calls and external dependencies
3. **Test Edge Cases**: Include error scenarios and malformed inputs
4. **Validate Structure**: Test both content and structure of responses
5. **Use Descriptive Names**: Make test names clearly describe what they test

### Example Test Pattern

```python
@patch('deep_research_agent.research_agent_refactored.ChatDatabricks')
@patch('deep_research_agent.tools_brave.requests')
def test_specific_functionality(self, mock_requests, mock_chat):
    """Test description of what this validates."""
    # Setup mocks
    mock_requests.get.return_value.json.return_value = {"results": []}
    mock_chat.return_value = Mock()
    
    # Create agent
    agent = RefactoredResearchAgent()
    
    # Execute test
    result = agent.predict(test_request)
    
    # Validate results
    assert isinstance(result, ResponsesAgentResponse)
    assert result.content is not None
```

## Continuous Integration

These tests are designed to run in CI environments:

- **No External Dependencies**: All external services are mocked
- **Fast Execution**: Tests complete in under 60 seconds
- **Cross-Platform**: Compatible with Linux, macOS, Windows
- **Python Versions**: Tested with Python 3.11+

## Contributing

When adding new features:

1. **Add Unit Tests**: Test individual components
2. **Add Integration Tests**: Test feature end-to-end
3. **Update Fixtures**: Add new test data to `conftest.py`
4. **Document Changes**: Update this README for new test patterns

## Debugging Tests

### Verbose Output

```bash
# See detailed test output
pytest tests/ -v -s

# See print statements
pytest tests/ -v -s --capture=no
```

### Debug Individual Tests

```bash
# Run with debugger
pytest tests/test_integration.py::test_name --pdb

# Run with coverage to see untested code
pytest --cov=deep_research_agent --cov-report=term-missing tests/
```

This comprehensive test suite ensures the MLflow + LangGraph integration works reliably across various scenarios and edge cases.