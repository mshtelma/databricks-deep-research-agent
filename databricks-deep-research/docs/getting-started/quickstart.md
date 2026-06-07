# Quick Start

Get a research workflow running in 5 minutes.

## Setup

```bash
# Install
uv pip install -e ".[all]"

# Set up credentials (.env or environment)
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
# OR
export OPENAI_API_KEY="your-key"  # for standalone OpenAI usage
export BRAVE_API_KEY="your-brave-key"  # only when using the Brave web-search backend (databricks/brave/jina) — see guides/search-providers.md
```

## Option 1: Run a Built-in Workflow (3 lines)

```python
import asyncio
from openai import AsyncOpenAI
from databricks_deep_research import WorkflowRunner, load_workflow

async def main():
    client = AsyncOpenAI()  # or configured for Databricks
    workflow = load_workflow("examples/simple_research.yaml")
    runner = WorkflowRunner(workflow, client, model_mapping={
        "simple": "gpt-4o-mini",
        "analytical": "gpt-4o",
        "complex": "gpt-4o",
    })

    result = await runner.run(query="What are the latest advances in quantum computing?")
    print(result.output)  # Final synthesized report
    print(f"Tokens used: {result.token_usage}")

asyncio.run(main())
```

## Option 2: Stream Events

```python
async def main_streaming():
    client = AsyncOpenAI()
    workflow = load_workflow("examples/simple_research.yaml")
    runner = WorkflowRunner(workflow, client, model_mapping={
        "simple": "gpt-4o-mini",
        "analytical": "gpt-4o",
        "complex": "gpt-4o",
    })

    async for event in runner.stream(query="What is the current state of fusion energy?"):
        match event.event_type:
            case "node_started":
                print(f"Starting: {event.node_id}")
            case "plan_created":
                print(f"Research plan: {event.steps}")
            case "reflection_decision":
                print(f"Reflector says: {event.decision}")
            case "agent_stream_chunk":
                print(event.chunk, end="", flush=True)
            case "workflow_completed":
                print(f"\nDone! Tokens: {event.total_tokens}")

asyncio.run(main_streaming())
```

## Option 3: Use the Low-Level API

```python
from databricks_deep_research import (
    WorkflowExecutor, WorkflowState, ExecutionContext,
    FrameworkLLMClient, load_workflow,
)

async def main_low_level():
    workflow = load_workflow("examples/simple_research.yaml")
    llm = FrameworkLLMClient(
        client=AsyncOpenAI(),
        model_mapping={"simple": "gpt-4o-mini", "analytical": "gpt-4o", "complex": "gpt-4o"},
    )
    state = WorkflowState(query="Explain CRISPR gene editing advances in 2024")
    ctx = ExecutionContext(llm=llm, state=state)
    executor = WorkflowExecutor(workflow, ctx)

    async for event in executor.execute():
        print(f"[{event.event_type}] {getattr(event, 'node_id', '')}")

    # Access results
    print(state.get("synthesis"))

asyncio.run(main_low_level())
```

## What Just Happened?

When you ran any of the examples above, the framework executed a multi-agent research pipeline:

1. **Coordinator** classified the query complexity (simple vs complex)
2. **Background** ran a quick web search for initial context
3. **Planner** created a multi-step research plan
4. For each step, **Researcher** searched the web and crawled sources
5. **Reflector** decided after each step: CONTINUE, ADJUST, or COMPLETE
6. **Synthesizer** generated a final report with citations from collected evidence

See [Architecture](../concepts/architecture.md) for a deeper understanding of the execution model.

## Next Steps

- [Authentication](authentication.md) -- Configure Databricks workspace auth
- [Architecture](../concepts/architecture.md) -- Understand the execution model
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md) -- Write your own workflows
- [Builtin Agents](../guides/builtin-agents.md) -- Learn what each agent does
