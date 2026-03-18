# Node Types Reference

> Complete reference for all 8 workflow node types.

## Overview

Each node in a workflow tree has a `type` that determines how it executes. The executor dispatches to a dedicated handler per type, walks the tree depth-first, and yields `StreamEvent` objects as an async generator.

Every node -- regardless of type -- emits `NodeStartedEvent` on entry and `NodeCompletedEvent` on success. If a node fails, it emits `NodeErrorEvent` and may emit `NodeSkippedEvent` when `error_handling.on_error` is `"skip"`.

---

## 1. `agent`

The primary computation node. Calls an LLM via the agent harness (`execute_agent`), optionally with tools (ReAct loop) and pool access.

### Config (`AgentNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `subtype` | `str` | *(required)* | Agent subtype: `coordinator`, `planner`, `researcher`, `reflector`, `synthesizer`, `evaluator`, or custom. |
| `model_tier` | `str` | `"analytical"` | LLM tier to use: `simple`, `analytical`, or `complex`. |
| `system_prompt` | `str` | `""` | System prompt sent to the LLM. |
| `user_prompt_template` | `str` | `""` | User prompt template with `{placeholder}` variables resolved from state. |
| `input_keys` | `list[str]` | `[]` | State keys to inject into the prompt. Auto-detected from template if omitted. |
| `output_key` | `str` | `"output"` | State key where the agent's output is stored. |
| `output_mode` | `str` | `"text"` | Output parsing mode: `text`, `json`, `structured`. |
| `output_format` | `str` | `"text"` | Output format hint: `text`, `markdown`, `json`. |
| `output_schema` | `dict \| None` | `None` | JSON schema for structured output validation. |
| `grounding_mode` | `str \| None` | `None` | Citation grounding mode: `none`, `classical_lite`, `reclaim`. |
| `tools` | `list[str \| dict]` | `[]` | Tool references (by name or `{type, name}` dict). Resolved via `ToolResolver`. |
| `pool_writes` | `list[PoolWriteConfig]` | `[]` | Pools to write results into (pool name + extraction path). |
| `pool_tools` | `list[str]` | `[]` | Pool names whose search tools should be available to this agent. |
| `max_tool_calls` | `int` | `0` | Maximum tool calls in the ReAct loop. `0` means no tool calling. |
| `max_retries` | `int` | `2` | Maximum retries on LLM call failure. |
| `max_result_chars` | `int` | `4000` | Truncation limit for old tool results in conversation. `0` = unlimited. |
| `conversation_budget` | `int \| None` | `None` | Max tokens for the agent's conversation context. |
| `pool_inject` | `list[PoolInjectConfig]` | `[]` | Pools whose contents are injected into the prompt before LLM call. |

### Children

**Leaf node** -- must have no children.

### Execution Behavior

1. Resolves tool references from the `ToolResolver` (logs warnings for missing tools; raises if `strict_tool_resolution` is enabled).
2. Adds pool search tools for each pool name listed in `pool_tools`.
3. Calls `execute_agent()` which runs the prompt-LLM-parse cycle (with optional ReAct tool-calling loop).
4. Writes output to state under `output_key`.
5. Writes to pools per `pool_writes` configuration.
6. Tracks token usage and source counts.
7. Yields all events produced by the harness.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- `AgentOutputEvent` -- final output with key and preview
- `AgentStreamChunkEvent` -- token-by-token streaming (synthesizer subtype)
- `ToolCallEvent`, `ToolResultEvent`, `ToolCacheHitEvent` -- when tools are used
- Domain events per subtype:
  - **coordinator**: `CoordinatorClassifiedEvent`
  - **planner**: `PlanCreatedEvent`
  - **reflector**: `ReflectionDecisionEvent`
  - **background**: `BackgroundCompletedEvent`
  - **synthesizer**: `SynthesisStartedEvent`

### YAML Example

```yaml
- id: research_step
  type: agent
  label: "Research current topic"
  config:
    subtype: researcher
    model_tier: analytical
    tools:
      - web_search
      - web_crawl
    pool_writes:
      - pool: sources
        extract: sources
    pool_tools:
      - sources
    max_tool_calls: 15
    output_key: findings
```

---

## 2. `tool`

Direct tool execution without an LLM. Calls a registered tool with arguments mapped from state.

### Config (`ToolNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref` | `dict` | *(required)* | Tool reference: `{type: "builtin", name: "web_search"}`. |
| `input_mapping` | `dict[str, str]` | `{}` | Maps tool argument names to state keys. |
| `output_key` | `str` | `"tool_result"` | State key where the tool result is stored. |

### Children

**Leaf node** -- must have no children.

### Execution Behavior

1. Parses `ref` into a `ToolRef` and resolves the tool from the `ToolRegistry`.
2. Reads arguments from state via `input_mapping`.
3. Validates arguments against the tool's schema.
4. Executes the tool with a `ToolContext` containing the current query and URL registry.
5. Stores the result content in state under `output_key`.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`

> Note: Unlike agent tool calls, direct tool nodes do not emit `ToolCallEvent`/`ToolResultEvent` in the current implementation. They execute silently and write results to state.

### YAML Example

```yaml
- id: fetch_page
  type: tool
  label: "Fetch source page"
  config:
    ref:
      type: builtin
      name: web_crawl
    input_mapping:
      url: target_url
    output_key: page_content
```

---

## 3. `sequence`

Executes children in order, one at a time. The simplest composite node.

### Config

None -- sequence nodes have no type-specific configuration.

### Children

**1 or more required.** Children execute in order from first to last.

### Execution Behavior

1. Iterates through `node.children` in order.
2. For each child, calls `_exec_node()` and yields all child events.
3. If a child fails (and error handling does not suppress it), execution stops and the error propagates.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- Plus all events from each child node (in order)

### YAML Example

```yaml
- id: research_pipeline
  type: sequence
  label: "Research pipeline"
  children:
    - id: background
      type: agent
      label: "Background investigation"
      config:
        subtype: background
        model_tier: simple
    - id: research
      type: plan_and_execute
      label: "Main research"
      config: { ... }
    - id: synthesis
      type: agent
      label: "Write report"
      config:
        subtype: synthesizer
        model_tier: complex
```

---

## 4. `parallel`

Executes children concurrently via `asyncio.gather` with a merged event queue.

### Config

None -- parallel nodes have no type-specific configuration.

### Children

**2 or more recommended.** (1 child is valid but offers no concurrency benefit.)

### Execution Behavior

1. Creates an `asyncio.Queue` for merged events.
2. Launches all children as concurrent `asyncio.Task` instances.
3. Each child pushes its events into the shared queue, followed by a `None` sentinel on completion.
4. The parallel handler dequeues and yields events as they arrive (interleaved).
5. Waits until all child sentinels are received, then gathers tasks.
6. If any child raises an exception, the first error is re-raised after all tasks complete.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- Plus all events from child nodes (interleaved in real-time arrival order)

### Notes

- Parallel writes to the same pool are safe because `PoolState` uses `asyncio.Lock` internally.
- Event ordering across children is non-deterministic -- consumers should not depend on child event order.

### YAML Example

```yaml
- id: parallel_research
  type: parallel
  label: "Research in parallel"
  children:
    - id: web_researcher
      type: agent
      label: "Web research"
      config:
        subtype: researcher
        tools: [web_search, web_crawl]
    - id: internal_researcher
      type: agent
      label: "Internal data research"
      config:
        subtype: researcher
        tools: [vector_search]
```

---

## 5. `loop`

Repeats its children until an exit condition is met or `max_iterations` is reached.

### Config (`LoopNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `until` | `dict` | *(required)* | Exit condition (serialized `StateCondition`, `LLMCondition`, or `CompositeCondition`). |
| `min_iterations` | `int` | `1` | Minimum iterations before the exit condition is checked. |
| `max_iterations` | `int` | `10` | Hard upper bound on iterations. |

### Condition Types

The `until` field accepts these condition types:

- **`state`**: Check a state key against a value. Operators: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `contains`, `in`, `exists`, `not_exists`.
- **`llm`**: Ask the LLM a yes/no question (deferred implementation).
- **`composite`**: Boolean combination (`all`, `any`, `not`) of other conditions.

### Children

**1 or more required** (the loop body). All children are executed sequentially per iteration.

### Execution Behavior

1. Enters the loop, incrementing iteration count from 1.
2. Emits `LoopIterationEvent` at the start of each iteration.
3. Executes all children sequentially (the loop body).
4. After the body completes, if `iteration >= min_iterations`, evaluates the `until` condition against current state.
5. If the condition is true, emits `LoopExitEvent` with `reason="condition_met"` and exits.
6. If the condition cannot be parsed, emits `LoopExitEvent` with `reason="parse_failure"` and exits.
7. If `max_iterations` is reached without the condition being met, emits `LoopExitEvent` with `reason="max_iterations"`.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- `LoopIterationEvent` -- at the start of each iteration (`iteration`, `max_iterations`)
- `LoopExitEvent` -- on termination (`reason`: `"condition_met"`, `"max_iterations"`, or `"parse_failure"`; `total_iterations`)
- Plus all events from child nodes (repeated per iteration)

### YAML Example

```yaml
- id: iterative_refinement
  type: loop
  label: "Refine until quality threshold"
  config:
    min_iterations: 2
    max_iterations: 5
    until:
      type: state
      key: quality_score
      operator: gte
      value: 0.8
  children:
    - id: refine
      type: agent
      label: "Refine draft"
      config:
        subtype: researcher
        output_key: quality_score
```

---

## 6. `conditional`

Branches execution based on state conditions. Evaluates conditions in order and executes the first matching branch (or the default).

### Config (`ConditionalNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conditions` | `list[dict]` | *(required)* | Ordered list of serialized conditions. Each condition maps to the child at the same index. |
| `default_branch` | `int` | `0` | Index of the child to execute if no condition matches. |

### Children

**2 or more required** (one per branch). Each child index corresponds to a condition at the same index.

### Execution Behavior

1. Materializes a snapshot of the latest state values for condition evaluation.
2. Iterates through `conditions` in order, evaluating each against the state snapshot.
3. The first condition that evaluates to `true` selects the child at that index.
4. If no condition matches, the child at `default_branch` index is selected.
5. Emits `BranchSelectedEvent` with the selected branch index.
6. Executes the selected child node.
7. Conditions that raise exceptions during evaluation are silently skipped (execution continues to the next condition).

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- `BranchSelectedEvent` -- which branch was selected (`branch_index`, `condition_summary`)
- Plus all events from the selected child node

### YAML Example

```yaml
- id: route_by_complexity
  type: conditional
  label: "Route by query complexity"
  config:
    default_branch: 1
    conditions:
      - type: state
        key: coordination.is_simple
        operator: eq
        value: true
      - type: state
        key: coordination.complexity
        operator: eq
        value: "complex"
  children:
    - id: simple_path
      type: agent
      label: "Quick answer"
      config:
        subtype: synthesizer
        model_tier: simple
    - id: full_research
      type: sequence
      label: "Full research pipeline"
      children: [...]
```

---

## 7. `subworkflow`

Executes a nested workflow definition. Allows workflow composition and reuse.

### Config (`SubworkflowNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref` | `str` | *(required)* | Workflow name or file path to load. |
| `params` | `dict[str, Any]` | `{}` | Parameters passed to the subworkflow. |
| `input_mapping` | `dict[str, str]` | `{}` | Maps parent state keys to subworkflow input keys. |
| `output_mapping` | `dict[str, str]` | `{}` | Maps subworkflow output keys back to parent state keys. |
| `output_key` | `str` | `"subworkflow_result"` | State key for the subworkflow's main result. |
| `pool_mode` | `str` | `"inherit"` | Pool sharing strategy: `inherit` (share parent pools), `isolate` (fresh pools), `merge` (merge on completion). |

### Children

**Leaf node** -- must have no children. The subworkflow has its own internal tree.

### Execution Behavior

> **Status: Deferred to P2.** The current implementation raises `NotImplementedError`.

When implemented, the expected behavior is:
1. Load or resolve the referenced workflow definition.
2. Create a child `WorkflowState` with mapped inputs.
3. Execute the subworkflow tree, yielding all events.
4. Map outputs back to the parent state.

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- Plus all events from the subworkflow execution (when implemented)

### YAML Example

```yaml
- id: run_sub
  type: subworkflow
  label: "Run specialized analysis"
  config:
    ref: analysis_workflow.yaml
    input_mapping:
      query: analysis_question
    output_mapping:
      report: analysis_result
    pool_mode: merge
```

---

## 8. `plan_and_execute`

The most complex node type. Implements a plan-execute-evaluate loop: a planner agent creates a list of items, a body node executes each item, an optional evaluator decides whether to continue, complete, or replan.

### Config (`PlanAndExecuteNodeConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `planner` | `dict` | *(required)* | Serialized `AgentNodeConfig` for the planner agent. |
| `items_path` | `str` | `"steps"` | Dot-path into planner output to extract the iterable list. |
| `item_state_key` | `str` | `"current_step"` | State key where the current item is written before body execution. |
| `body` | `dict` | `{}` | Serialized `WorkflowNode` to execute per item. |
| `evaluator` | `dict \| None` | `None` | Serialized `AgentNodeConfig` for the evaluator agent (optional). |
| `max_iterations` | `int` | `10` | Maximum total items to process across all replan cycles. |
| `min_iterations` | `int` | `1` | Minimum items to process before evaluator can return `"complete"`. |
| `max_replan_cycles` | `int` | `3` | Maximum times the planner can be re-invoked. |
| `complete_on_exhaustion` | `bool` | `True` | Whether to exit when all planned items are processed (vs. continuing to next cycle). |
| `planner_guidance` | `str` | `""` | Free-text guidance injected into the planner prompt template. |
| `synthesis_metadata` | `dict[str, str]` | `{}` | Key-value pairs written to state for downstream synthesizer use. |

### Children

**Leaf node** -- must have no children. Uses `planner`, `body`, and `evaluator` from config instead.

### Execution Behavior

1. Writes `min_steps`, `max_steps`, `step_prompt_guidance`, and `synthesis_metadata` to state.
2. Creates a shared `ToolCallCache` for deduplication across all steps.
3. **Outer loop** (up to `max_replan_cycles + 1` cycles):
   a. **Plan phase**: Runs the planner agent, extracts items from output via `items_path`.
   b. Emits `ItemsExtractedEvent` with total items and cycle number.
   c. **Item loop**: For each extracted item (up to `max_iterations` total):
      - Writes the item to state under `item_state_key` (plus individual fields like `step_title`, `step_description`).
      - Emits `ItemStartedEvent`.
      - Executes the `body` node.
      - Tracks item health (blocked steps, source counts).
      - Emits `ItemCompletedEvent`.
      - **Evaluation** (if evaluator configured):
        - Injects progress metadata (remaining steps, source counts, observations) into state.
        - Runs the evaluator agent.
        - Normalizes the decision to `"continue"`, `"complete"`, or `"replan"`.
        - Blocked steps override `"continue"` to `"replan"`.
        - `"complete"` is gated: downgraded to `"continue"` if `items_processed < min_iterations`.
        - Emits `EvaluationDecisionEvent`.
        - On `"complete"`: populates synthesis state and exits.
        - On `"replan"`: emits `ReplanTriggeredEvent`, breaks to outer loop for replanning.
      - **Blocked step without evaluator**: triggers replan if cycles remain, otherwise exits.
   d. **All items exhausted**: if `complete_on_exhaustion` is true (or no evaluator), exits. Otherwise continues to next cycle.
4. Emits `PlanAndExecuteExitEvent` on any exit path.

### Exit Reasons

| Reason | Description |
|--------|-------------|
| `evaluator_complete` | Evaluator decided research is sufficient. |
| `max_iterations` | Reached the maximum item count. |
| `items_exhausted` | All planned items processed with `complete_on_exhaustion=True`. |
| `max_replan_cycles` | Exhausted all replan cycles. |
| `blocked_step` | A step was blocked and no replan cycles remain. |

### Events Emitted

- `NodeStartedEvent`, `NodeCompletedEvent`
- `ItemsExtractedEvent` -- after planner output is parsed (`total_items`, `items_path`, `cycle`)
- `ItemStartedEvent` -- before each item's body executes (`item_index`, `item_summary`, `total_items`)
- `ItemCompletedEvent` -- after each item's body completes (`item_index`, `items_processed`)
- `EvaluationDecisionEvent` -- after evaluator runs (`decision`, `reasoning`, `items_processed`)
- `ReplanTriggeredEvent` -- when replanning is triggered (`cycle`, `reason`, `items_remaining`)
- `PlanAndExecuteExitEvent` -- on exit (`reason`, `total_items_processed`, `replan_cycles`, `total_planned`)
- Plus all events from the planner agent, body node, and evaluator agent

### YAML Example

```yaml
- id: main_research
  type: plan_and_execute
  label: "Plan and execute research"
  config:
    max_iterations: 8
    min_iterations: 3
    max_replan_cycles: 2
    items_path: steps
    item_state_key: current_step
    planner_guidance: "Focus on primary sources and recent data."
    synthesis_metadata:
      research_depth: extended
      max_words: "3000"
    planner:
      subtype: planner
      model_tier: analytical
      output_key: plan
      output_mode: json
    body:
      id: step_body
      type: sequence
      label: "Execute research step"
      children:
        - id: researcher
          type: agent
          label: "Research"
          config:
            subtype: researcher
            tools: [web_search, web_crawl]
            pool_writes:
              - pool: sources
                extract: sources
            max_tool_calls: 15
    evaluator:
      subtype: reflector
      model_tier: analytical
      output_key: reflection
      output_mode: json
```

---

## Node Type Classification

| Type | Leaf/Composite | Requires Children | Config Model |
|------|---------------|-------------------|--------------|
| `agent` | Leaf | No | `AgentNodeConfig` |
| `tool` | Leaf | No | `ToolNodeConfig` |
| `sequence` | Composite | Yes (1+) | *(none)* |
| `parallel` | Composite | Yes (2+) | *(none)* |
| `loop` | Composite | Yes (1+) | `LoopNodeConfig` |
| `conditional` | Composite | Yes (2+) | `ConditionalNodeConfig` |
| `subworkflow` | Leaf | No | `SubworkflowNodeConfig` |
| `plan_and_execute` | Leaf | No | `PlanAndExecuteNodeConfig` |

## Error Handling

Every node supports an optional `error_handling` field (`ErrorConfig`):

| Policy | Behavior |
|--------|----------|
| `fail` (default) | Propagate the exception upward. Emits `NodeErrorEvent`. |
| `skip` | Emit `NodeSkippedEvent` and continue to the next sibling. |
| `retry` | Retry up to `max_retries` times with exponential back-off (`retry_delay_seconds * 2^attempt`). Emits `NodeErrorEvent` with `will_retry=True` per attempt. |

```yaml
- id: fragile_step
  type: agent
  label: "Might fail"
  error_handling:
    on_error: retry
    max_retries: 3
    retry_delay_seconds: 2.0
  config:
    subtype: researcher
```

## See Also

- [Workflow Engine](../concepts/workflow-engine.md)
- [Workflow Definition Schema](workflow-definition-schema.md)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
