# Custom Agents

> Register custom agent subtypes with the framework.

## Overview
Beyond the 6 builtin subtypes, you can register custom agent subtypes with their own default configurations, output models, and prompts.

## How Subtypes Work
When an agent node has `subtype: my_custom`, the harness:
1. Looks up defaults for "my_custom" in the subtype registry
2. Merges defaults with node config (node config wins)
3. Executes with the merged config

## Registering a Custom Subtype

The `register_builtin()` function accepts a subtype name and optional hook functions. Each hook targets a different phase of the agent execution lifecycle inside `execute_agent()`.

```python
from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.state import WorkflowState
from pydantic import BaseModel
from typing import Any


class FactCheckerOutput(BaseModel):
    claims: list[str]
    verdicts: list[str]
    confidence: float


def _enrich_config(
    config: AgentNodeConfig, state: WorkflowState
) -> AgentNodeConfig:
    """Fill in fact-checker defaults when not specified in the YAML node."""
    updates: dict[str, Any] = {}

    if not config.system_prompt:
        updates["system_prompt"] = "You are a fact-checking agent..."

    if not config.user_prompt_template:
        updates["user_prompt_template"] = (
            "Verify these claims against the provided sources:\n"
            "{{ query }}"
        )

    if config.output_model is None:
        updates["output_model"] = FactCheckerOutput

    if updates:
        return config.model_copy(update=updates)
    return config


def _post_process(
    node_id: str,
    output: Any,
    config: AgentNodeConfig,
    state: WorkflowState,
) -> list[StreamEvent]:
    """Emit domain events after the agent runs."""
    # Return custom StreamEvent instances, or an empty list.
    return []


register_builtin(
    "fact_checker",
    enrich_config=_enrich_config,
    post_process=_post_process,
    output_model=FactCheckerOutput,
)
```

### `register_builtin()` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `subtype` | `str` | Unique name for the subtype (positional). |
| `post_process` | `PostProcessFn \| None` | Called after output is written to state. Receives `(node_id, output, config, state)` and returns a list of `StreamEvent` to emit. |
| `enrich_config` | `ConfigEnrichFn \| None` | Called before input construction. Receives `(config, state)` and returns a (possibly updated) `AgentNodeConfig`. Use this to inject default prompts, output models, or model tier. |
| `execute` | `ExecuteFn \| None` | Completely replaces the default LLM call + ReAct loop. Receives `(node_id, config, state, llm_client, tools, pools, agent_input, messages, tool_ctx)` and returns an `AgentOutput` or `None`. When it returns `None`, the harness falls through to the standard execution path. |
| `default_system_prompt` | `str` | Stored on the `BuiltinSubtype` entry (not automatically applied -- use `enrich_config` to apply it). |
| `default_user_prompt` | `str` | Same as above. |
| `output_model` | `Any` | Pydantic model class stored on the entry. Typically also set via `enrich_config`. |

All keyword parameters are optional. A minimal registration only needs the subtype name:

```python
register_builtin("simple_agent")
```

## Using Custom Subtypes in YAML

Once registered, reference the subtype by name in any agent node:

```yaml
root:
  id: pipeline
  type: sequence
  children:
    - id: research
      type: agent
      config:
        subtype: researcher
    - id: verify
      type: agent
      config:
        subtype: fact_checker  # Custom subtype
        model_tier: analytical
        output_key: fact_check
        output_mode: structured
        pool_inject:
          - pool: observations
            format: text
            max_items: 20
```

Node-level config fields always override any defaults set by `enrich_config`. For example, if the YAML specifies `system_prompt`, the enrichment function should skip overriding it (as shown in the `if not config.system_prompt` pattern above).

## Custom Output Models

When `output_mode` is `"structured"`, the harness passes the `output_model` to the LLM client for structured output parsing.

Requirements:
- Must be a `pydantic.BaseModel` subclass
- Set via `enrich_config` (preferred) or directly in the YAML node's `output_model` field
- The LLM response is parsed into this model automatically

```python
from pydantic import BaseModel, Field


class FactCheckerOutput(BaseModel):
    claims: list[str] = Field(description="The claims that were checked")
    verdicts: list[str] = Field(
        description="SUPPORTED, REFUTED, or NOT_ENOUGH_INFO for each claim"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the verification"
    )
```

Field descriptions are passed to the LLM as part of the structured output schema, so they directly influence output quality.

## Hooks

The registry supports three hooks, called at different points in the `execute_agent()` lifecycle:

### `enrich_config` -- Before Execution

Called before the harness builds the `AgentInput`. Use it to fill in default prompts, output models, or any config field the YAML node did not specify.

```python
def _enrich_config(
    config: AgentNodeConfig, state: WorkflowState
) -> AgentNodeConfig:
    updates: dict[str, Any] = {}
    if not config.system_prompt:
        updates["system_prompt"] = MY_SYSTEM_PROMPT
    if config.output_model is None:
        updates["output_model"] = MyOutputModel
    if updates:
        return config.model_copy(update=updates)
    return config
```

The pattern of checking each field before overriding ensures that YAML-level config always takes precedence.

### `execute` -- Replace Standard Execution

When provided and returning a non-`None` `AgentOutput`, this completely replaces the standard LLM call (or ReAct loop). The harness still handles state writes, pool writes, and event emission around it.

```python
from databricks_deep_research.agents.isolation import AgentInput, AgentOutput

async def _execute(
    node_id, config, state, llm_client, tools, pools,
    agent_input, messages, tool_ctx,
) -> AgentOutput | None:
    # Custom execution logic here.
    # Return None to fall through to standard execution.
    response = await llm_client.complete(messages, config.model_tier)
    return AgentOutput(
        content=response.content,
        output_key=config.output_key,
        pool_writes={},
        sources=[],
        token_usage=response.usage,
        events=[],
    )
```

The synthesizer builtin uses this hook for its multi-phase citation pipeline.

### `post_process` -- After Execution

Called after the output has been written to state. Use it to emit domain-specific `StreamEvent` instances (e.g., `ReflectionDecisionEvent`).

```python
def _post_process(
    node_id: str,
    output: Any,
    config: AgentNodeConfig,
    state: WorkflowState,
) -> list[StreamEvent]:
    if isinstance(output, FactCheckerOutput):
        # Return custom events to stream to the frontend
        return [MyVerificationEvent(
            node_id=node_id,
            confidence=output.confidence,
        )]
    return []
```

## Override vs Custom

**Override defaults** when you want the same behavior but different config. Set fields directly in the YAML node -- no registration needed:

```yaml
- id: deep_reflect
  type: agent
  config:
    subtype: reflector          # Uses builtin reflector
    model_tier: reasoning       # Override default model tier
    system_prompt: "Custom reflector prompt..."
```

**Create a new subtype** when you need fundamentally different behavior:
- A new output model with different fields
- Custom post-processing events
- A completely custom `execute` function
- Domain-specific `enrich_config` logic that reads state

## Querying the Registry

```python
from databricks_deep_research.agents.builtins.registry import (
    get_builtin,
    list_builtins,
)

# List all registered subtypes (builtin + custom)
names = list_builtins()
# => ["coordinator", "planner", "researcher", "reflector", "synthesizer", "background", "fact_checker"]

# Look up a specific subtype
entry = get_builtin("fact_checker")
if entry:
    print(entry.subtype)       # "fact_checker"
    print(entry.output_model)  # <class 'FactCheckerOutput'>
```

## See Also
- [Agent System](../concepts/agent-system.md)
- [Builtin Agents](builtin-agents.md)
- [Agent Config Reference](../reference/agent-config-reference.md)
