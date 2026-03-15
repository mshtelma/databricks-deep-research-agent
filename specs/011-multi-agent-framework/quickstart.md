# Quickstart: databricks-deep-research Framework

## Installation

```bash
pip install databricks-deep-research
# With all builtin tool dependencies:
pip install "databricks-deep-research[all]"
```

## 1. Define a Workflow in YAML

Create `my_workflow.yaml`:

```yaml
id: "simple-research"
name: "Simple Research Pipeline"
version: 1
required_inputs: [query]
output_keys: [report]

pools:
  - name: observations
    item_type: text
  - name: sources
    item_type: source
    dedup_key: url

root:
  id: root
  type: sequence
  label: "Research Pipeline"
  children:
    - id: coordinator
      type: agent
      label: "Query Coordinator"
      config:
        subtype: coordinator
        input_keys: [query]
        output_key: coordination

    - id: planner
      type: agent
      label: "Research Planner"
      config:
        subtype: planner
        input_keys: [query, coordination]

    - id: research_loop
      type: loop
      label: "Research Loop"
      config:
        min_iterations: 2
        max_iterations: 5
        until:
          type: state
          key: reflection.decision
          operator: eq
          value: "COMPLETE"
      children:
        - id: research_body
          type: sequence
          label: "Research Step"
          children:
            - id: researcher
              type: agent
              label: "Researcher"
              config:
                subtype: researcher
                pool_writes:
                  - pool: observations
                  - pool: sources
                pool_tools: [observations, sources]

            - id: reflector
              type: agent
              label: "Reflector"
              config:
                subtype: reflector
                pool_tools: [observations]

    - id: synthesizer
      type: agent
      label: "Report Synthesizer"
      config:
        subtype: synthesizer
        pool_tools: [observations, sources]
        output_key: report
        output_format: markdown
```

## 2. Configure the LLM Client

The framework uses `AsyncOpenAI` directly -- no protocol to implement.

```python
from openai import AsyncOpenAI
from databricks_deep_research.llm.client import FrameworkLLMClient

# Option A: Direct API key
client = FrameworkLLMClient(
    openai_client=AsyncOpenAI(api_key="sk-..."),
    model_mapping={
        "simple": "gpt-4o-mini",
        "analytical": "gpt-4o",
        "complex": "o1",
    },
)

# Option B: Databricks model serving endpoints
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(profile="my-profile")
openai_client = w.serving_endpoints.get_open_ai_client()

client = FrameworkLLMClient(
    openai_client=openai_client,
    model_mapping={
        "simple": "databricks-meta-llama-3-1-8b-instruct",
        "analytical": "databricks-meta-llama-3-1-70b-instruct",
        "complex": "databricks-meta-llama-3-1-405b-instruct",
    },
)
```

## 3. Execute the Workflow

```python
import asyncio
from databricks_deep_research import run_workflow_from_yaml

async def main():
    client = FrameworkLLMClient(
        openai_client=AsyncOpenAI(api_key="sk-..."),
        model_mapping={
            "simple": "gpt-4o-mini",
            "analytical": "gpt-4o",
            "complex": "o1",
        },
    )

    async for event in run_workflow_from_yaml(
        yaml_path="my_workflow.yaml",
        llm_client=client,
        query="What are the latest advances in quantum computing?",
    ):
        print(f"[{event.node_id}] {event.event_type}")

        # Check for the final report
        if event.event_type == "agent_output" and event.output_key == "report":
            print(f"\nReport preview: {event.output_preview[:200]}...")

asyncio.run(main())
```

## 4. Use with Databricks (Deep Research App Pattern)

```python
from databricks.sdk import WorkspaceClient
from databricks_deep_research import WorkflowDefinition, WorkflowExecutor, ExecutionContext
from databricks_deep_research.llm.client import FrameworkLLMClient

# Use WorkspaceClient for auth
w = WorkspaceClient(profile="my-profile")
openai_client = w.serving_endpoints.get_open_ai_client()

# Build LLM client with model tier mapping
llm_client = FrameworkLLMClient(
    openai_client=openai_client,
    model_mapping={
        "simple": "databricks-meta-llama-3-1-8b-instruct",
        "analytical": "databricks-meta-llama-3-1-70b-instruct",
        "complex": "databricks-meta-llama-3-1-405b-instruct",
    },
)

# Load workflow
definition = WorkflowDefinition.from_yaml("workflows/deep_research.yaml")

# Build execution context
context = ExecutionContext(
    llm_client=llm_client,
)

# Execute
executor = WorkflowExecutor(context)
async for event in executor.execute(definition):
    # Map events to SSE, save to DB, etc.
    pass
```

## 5. Customize Agent Prompts

Override the main prompt pair for any agent subtype in YAML:

```yaml
- id: custom_researcher
  type: agent
  label: "Domain-Specific Researcher"
  config:
    subtype: researcher
    system_prompt: |
      You are a medical research specialist. Focus exclusively on
      peer-reviewed sources and clinical trials. Always cite DOI numbers.
    user_prompt_template: |
      Research the following medical topic: {query}
      Focus area: {domain_filter}
    pool_writes:
      - pool: observations
      - pool: sources
    pool_tools: [observations, sources]
```

When `system_prompt` or `user_prompt_template` is omitted, the subtype's built-in default is used.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **8 Node Types** | agent, tool, sequence, parallel, loop, conditional, subworkflow, plan_and_execute |
| **Append-Only State** | `state.get(key)` for latest, `state.get_all(key)` for accumulated |
| **Shared Pools** | Multi-producer accumulation with dedup and search |
| **Pool Tools** | Auto-generated search/retrieval tools for each pool |
| **6 Agent Subtypes** | coordinator, researcher, planner, reflector, synthesizer, background (with defaults) |
| **Conditions** | StateCondition, LLMCondition, CompositeCondition |
| **AsyncOpenAI Client** | Framework wraps `openai.AsyncOpenAI` directly -- no protocol to implement |
| **Streaming Events** | Pydantic BaseModel with `event_type` discriminator for pattern matching |
| **YAML First-Class** | Load and save workflow definitions via `from_yaml()` / `to_yaml()` |
