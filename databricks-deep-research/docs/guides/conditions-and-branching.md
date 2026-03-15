# Conditions and Branching

> Control workflow execution flow with conditional nodes and state conditions.

## Overview

Conditional nodes evaluate state conditions to choose which branch to execute. This enables dynamic workflows that adapt based on intermediate results -- for example, routing simple questions to a fast synthesizer while sending complex queries through a full research pipeline.

Conditions are also used in loop nodes to control when iteration should stop.

## Condition Types

The framework supports three condition types, defined as a discriminated union on the `type` field.

### StateCondition

Evaluates a value from the workflow state against a comparison.

```yaml
key: coordination.is_simple
operator: eq
value: true
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"state"` | Discriminator (optional in YAML shorthand) |
| `key` | `str` | Dot-path into workflow state |
| `operator` | `str` | Comparison operator (see table below) |
| `value` | `Any` | Right-hand side of the comparison (omit for `exists`/`not_exists`) |

### LLMCondition

Asks an LLM a yes/no question to decide the branch. Useful when the routing decision cannot be reduced to a simple state check.

```yaml
type: llm
prompt_template: "Is the following query answerable with a single web search? Query: {query}"
model_tier: simple
expected_output: "yes"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"llm"` | | Discriminator |
| `prompt_template` | `str` | | Template string (may reference state keys) |
| `model_tier` | `str` | `"simple"` | Which model tier to use for evaluation |
| `expected_output` | `str` | `"yes"` | The LLM response that makes the condition true |

### CompositeCondition

Combines multiple conditions with boolean logic.

```yaml
type: composite
operator: all   # all | any | not
conditions:
  - key: research.source_count
    operator: gt
    value: 0
  - key: research.confidence
    operator: gte
    value: 0.8
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"composite"` | Discriminator |
| `operator` | `str` | Boolean combinator: `all` (AND), `any` (OR), or `not` (negation) |
| `conditions` | `list[Condition]` | Nested conditions (recursive -- composites can contain composites) |

## Condition Operators

All operators supported by `StateCondition`:

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equal | `operator: eq, value: true` |
| `neq` | Not equal | `operator: neq, value: "skip"` |
| `gt` | Greater than | `operator: gt, value: 5` |
| `gte` | Greater than or equal | `operator: gte, value: 3` |
| `lt` | Less than | `operator: lt, value: 10` |
| `lte` | Less than or equal | `operator: lte, value: 100` |
| `in` | Value is a member of the list | `operator: in, value: [complex, analytical]` |
| `contains` | String/list on the left contains value on the right | `operator: contains, value: "error"` |
| `exists` | Key exists in state (no `value` needed) | `operator: exists` |
| `not_exists` | Key does not exist in state (no `value` needed) | `operator: not_exists` |

When using `exists` or `not_exists`, the `value` field is ignored. For all other operators, if the key is missing from state the condition evaluates to `false`.

## State Key Resolution

Condition keys use dot-path notation resolved by `resolve_dot_path()`.

- **Simple keys** -- `"plan"` resolves to the latest value stored under the `plan` key in workflow state.
- **Nested keys** -- `"coordination.complexity"` first resolves `coordination` (which should be a dict), then looks up `complexity` within it.

Resolution walks each segment left to right. If the value at any segment is a `dict`, it uses `dict.get()`; otherwise it falls back to `getattr()`. If any segment is missing, the resolution returns a sentinel indicating the key does not exist.

```
coordination.is_simple
    ^              ^
    |              |
    state key      nested field within the dict stored at "coordination"
```

## Conditional Node Structure

A conditional node declares a list of conditions and a set of child branches. Each condition maps (by index) to the child that should execute if the condition is true.

```yaml
- id: route
  type: conditional
  label: Complexity Branch
  config:
    conditions:
      - key: coordination.is_simple
        operator: eq
        value: true
    default_branch: 1
  children:
    # Branch 0 -- selected when coordination.is_simple == true
    - id: simple_answer
      type: agent
      config:
        subtype: synthesizer
        model_tier: simple
        output_key: report

    # Branch 1 (default) -- full research path
    - id: deep_research
      type: sequence
      children:
        - id: background
          type: agent
          config:
            subtype: background
            model_tier: simple
            output_key: background
            tools: [web_search]
        - id: synthesizer
          type: agent
          config:
            subtype: synthesizer
            model_tier: analytical
            output_key: report
```

### Evaluation Order

Conditions are evaluated sequentially, in the order they appear in the `conditions` list. The **first match wins** -- its index determines which child branch to execute. If no condition matches, the `default_branch` index is used (defaults to `0`).

### Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conditions` | `list[dict]` | required | Serialized conditions, evaluated in order |
| `default_branch` | `int` | `0` | Index of the child to execute if no condition matches |

## Loop Conditions

Loop nodes use the `until` field to specify an exit condition. The condition is checked **after each iteration**, starting from `min_iterations`.

```yaml
- id: research_loop
  type: loop
  label: Research Loop
  config:
    until:
      key: reflection.decision
      operator: eq
      value: COMPLETE
    min_iterations: 1
    max_iterations: 10
  children:
    - id: researcher
      type: agent
      config:
        subtype: researcher
        model_tier: analytical
        output_key: findings
        tools: [web_search, web_crawl]

    - id: reflector
      type: agent
      config:
        subtype: reflector
        model_tier: analytical
        output_key: reflection
```

### Loop Exit Logic

1. Execute all children (one full iteration).
2. If the current iteration count is below `min_iterations`, go to step 1.
3. Evaluate the `until` condition against the current state.
4. If the condition is `true`, emit a `LoopExitEvent` with reason `"condition_met"` and stop.
5. If the condition fails to parse, emit a `LoopExitEvent` with reason `"parse_failure"` and stop.
6. If `max_iterations` is reached, emit a `LoopExitEvent` with reason `"max_iterations"` and stop.
7. Otherwise, go to step 1.

### Loop Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `until` | `dict` | required | Serialized condition that triggers loop exit |
| `min_iterations` | `int` | `1` | Minimum iterations before the exit condition is checked |
| `max_iterations` | `int` | `10` | Hard upper bound on iterations |

## Composite Conditions

Composite conditions let you combine multiple conditions using boolean logic. They are recursive -- a composite can contain other composites.

### AND (all)

All child conditions must be true.

```yaml
type: composite
operator: all
conditions:
  - key: research.source_count
    operator: gt
    value: 0
  - key: research.confidence
    operator: gte
    value: 0.8
```

### OR (any)

At least one child condition must be true.

```yaml
type: composite
operator: any
conditions:
  - key: coordination.complexity
    operator: eq
    value: complex
  - key: coordination.complexity
    operator: eq
    value: analytical
```

### NOT (not)

Negates a single child condition. The `conditions` list must contain exactly one entry.

```yaml
type: composite
operator: not
conditions:
  - key: coordination.is_simple
    operator: eq
    value: true
```

### Python Helper Constructors

For programmatic workflow building, the `conditions` module provides shorthand functions:

```python
from databricks_deep_research.workflow.conditions import (
    StateCondition,
    all_of,
    any_of,
    negate,
)

# AND: both must be true
cond = all_of(
    StateCondition(key="research.source_count", operator="gt", value=0),
    StateCondition(key="research.confidence", operator="gte", value=0.8),
)

# OR: at least one must be true
cond = any_of(
    StateCondition(key="coordination.complexity", operator="eq", value="complex"),
    StateCondition(key="coordination.complexity", operator="eq", value="analytical"),
)

# NOT: negate a condition
cond = negate(
    StateCondition(key="coordination.is_simple", operator="eq", value=True),
)
```

## Practical Patterns

### Route by Complexity

A coordinator agent classifies the query, then a conditional node routes simple queries to a fast synthesizer and complex queries to a full research pipeline.

```yaml
# Step 1: Classify
- id: coordinator
  type: agent
  config:
    subtype: coordinator
    model_tier: simple
    output_key: coordination

# Step 2: Branch
- id: branch
  type: conditional
  config:
    conditions:
      - key: coordination.is_simple
        operator: eq
        value: true
    default_branch: 1
  children:
    - id: simple_synth
      type: agent
      config:
        subtype: synthesizer
        model_tier: simple
        output_key: report
    - id: full_research
      type: sequence
      children: [...]
```

### Early Exit on Confidence

A reflector agent evaluates research quality after each step. When it outputs `COMPLETE`, the loop exits early instead of running all remaining iterations.

```yaml
- id: research_loop
  type: loop
  config:
    until:
      key: reflection.decision
      operator: eq
      value: COMPLETE
    min_iterations: 1
    max_iterations: 5
  children:
    - id: researcher
      type: agent
      config:
        subtype: researcher
        output_key: findings
        tools: [web_search, web_crawl]
    - id: reflector
      type: agent
      config:
        subtype: reflector
        output_key: reflection
```

### Source-Dependent Research

Route to different research strategies based on what data sources are available.

```yaml
- id: source_router
  type: conditional
  config:
    conditions:
      - key: coordination.preferred_source
        operator: eq
        value: enterprise
      - key: coordination.preferred_source
        operator: eq
        value: web
    default_branch: 2
  children:
    # Branch 0: Enterprise-only research
    - id: enterprise_research
      type: agent
      config:
        subtype: researcher
        tools: [vector_search, genie]
        output_key: findings

    # Branch 1: Web-only research
    - id: web_research
      type: agent
      config:
        subtype: researcher
        tools: [web_search, web_crawl]
        output_key: findings

    # Branch 2 (default): Mixed sources
    - id: mixed_research
      type: agent
      config:
        subtype: researcher
        tools: [web_search, web_crawl, vector_search]
        output_key: findings
```

## Full Example

See [`examples/conditional_research.yaml`](../../examples/conditional_research.yaml) for a complete workflow that combines coordinator classification, conditional branching, plan-and-execute research, and report synthesis.

## See Also

- [Workflow Engine](../concepts/workflow-engine.md)
- [Node Types Reference](../reference/node-types-reference.md)
- [YAML Workflow Authoring](yaml-workflow-authoring.md)
