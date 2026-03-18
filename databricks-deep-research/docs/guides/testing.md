# Testing

> Unit and integration testing patterns for framework workflows.

## Overview
Testing multi-agent workflows requires strategies for mocking LLM calls, verifying state mutations, and testing tool interactions.

## Test Setup
```bash
pip install "databricks-deep-research[dev]"
uv run pytest tests/ -v
```

## Unit Testing Patterns

### Testing Workflow Definitions
```python
from databricks_deep_research import load_workflow

def test_workflow_loads():
    wf = load_workflow("examples/simple_research.yaml")
    assert wf.name == "simple-research"
    assert wf.root.type == NodeType.SEQUENCE
```

### Testing State Mutations
```python
from databricks_deep_research import WorkflowState

def test_state_append_and_get():
    state = WorkflowState(query="test")
    state.append("node1", "key", "value")
    assert state.get("key") == "value"
    assert len(state.log) == 1
```

### Mocking the LLM Client
```python
from unittest.mock import AsyncMock

mock_llm = AsyncMock()
mock_llm.complete.return_value = LLMResponse(
    content='{"steps": ["Step 1", "Step 2"]}',
    tool_calls=[],
    usage={"total_tokens": 100},
    model="test",
    finish_reason="stop",
)
```

### Testing Tools
```python
async def test_custom_tool():
    tool = MyCustomTool(config=...)
    result = await tool.execute(arguments={"query": "test"}, context=mock_context)
    assert result.success
    assert len(result.sources) > 0
```

## Integration Testing
- Requires real LLM credentials
- Test complete workflow execution
- Verify event stream structure
- Check token usage bounds

## Testing Workflows End-to-End
```python
async def test_simple_research():
    runner = WorkflowRunner(workflow, client, model_mapping=mapping)
    result = await runner.run(query="What is quantum computing?")
    assert result.output is not None
    assert result.token_usage > 0
```

## Best Practices
- Unit test state mutations and workflow loading without LLM
- Mock LLM for agent behavior tests
- Integration test the full pipeline sparingly
- Test error handling (skip, retry) with mock failures
- Verify event types emitted during execution

## See Also
- [Installation](../getting-started/installation.md)
- [Architecture](../concepts/architecture.md)
