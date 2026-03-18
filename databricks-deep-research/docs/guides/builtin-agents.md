# Builtin Agents

> In-depth guide to all 6 builtin agent subtypes.

## Overview

The framework ships with 6 agent subtypes that form a complete research pipeline. Each subtype is registered via `register_builtin()` in `agents/builtins/registry.py` and provides:

- **Default prompts** (system and user) loaded lazily from `agents/prompts/`
- **Output models** (Pydantic) for structured LLM output
- **Config enrichment** that fills in sensible defaults when YAML omits them
- **Post-processing** that emits domain-specific `StreamEvent` instances
- **Optional execute hook** for custom execution logic (used by synthesizer for citation verification)

All subtypes are auto-discovered when you import the builtins package. The harness (`execute_agent()`) calls the registered `enrich_config` before execution and `post_process` after the LLM returns.

---

## 1. Coordinator

Classifies incoming queries by complexity, detects simple queries that can be answered directly, and recommends research depth. Acts as the entry gate to the pipeline.

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `simple` |
| `output_key` | `coordination` |
| `output_format` | `json` |
| `input_keys` | `[query]` |
| `tools` | `[]` (none) |
| `pool_writes` | `[]` (none) |
| `pool_tools` | `[]` (none) |

### Output Model -- `CoordinatorOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `complexity` | `str` | -- | Complexity classification (e.g. `"simple"`, `"moderate"`, `"complex"`) |
| `is_simple` | `bool` | `False` | Whether the query can be answered without research |
| `recommended_depth` | `str` | `"standard"` | Suggested research depth (e.g. `"light"`, `"standard"`, `"extended"`) |
| `direct_response` | `str \| None` | `None` | Direct answer if `is_simple` is true |
| `follow_up_type` | `str \| None` | `None` | Type of follow-up detected (e.g. continuation of previous query) |

### Key Prompts

- **System prompt**: Instructs the LLM to act as a query classifier, analyzing complexity and deciding whether direct answering or full research is needed.
- **User prompt**: Provides the raw query and asks for a JSON classification with complexity, depth, and optional direct response.

### Events Emitted

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `CoordinatorClassifiedEvent` | `coordinator_classified` | `complexity`, `recommended_depth`, `is_simple`, `direct_response`, `follow_up_type`, `reasoning` |

### YAML Example

```yaml
- id: coordinator
  type: agent
  label: Query Classifier
  config:
    subtype: coordinator
    model_tier: simple
    output_key: coordination
```

### Usage Tips

- Place at the very start of the pipeline to route between quick answers and full research.
- When `is_simple` is true, a downstream conditional node can skip the entire research cycle and return `direct_response` immediately.
- Uses the `simple` model tier for fast classification (low latency).
- No tools or pool access needed -- operates purely on the query text.

---

## 2. Background

Performs quick context discovery before planning. Decomposes the query into sub-questions, assesses the data landscape, and discovers relevant sources via web search.

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `simple` (from YAML convention; no subtype default in code) |
| `output_key` | `background` (typical YAML usage) |
| `output_format` | `json` (enforced by enrichment if set to `text`) |
| `input_keys` | `[query]` |
| `tools` | `[web_search]` (typical YAML usage) |
| `pool_writes` | `[]` (none by default) |
| `pool_tools` | `[]` (none) |

### Output Model -- `BackgroundOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_landscape` | `dict[str, Any]` | `{}` | Structured map of discovered data sources and their relevance |
| `summary` | `str` | `""` | Brief context summary for downstream agents |
| `query_decomposition` | `list[str]` | `[]` | Sub-questions derived from the original query |
| `discovered_sources` | `list[dict[str, Any]]` | `[]` | Individual source records found during background search |

### Key Prompts

- **System prompt**: Instructs the LLM to act as a background investigator -- decompose the query, assess what data is available, and create a landscape of relevant sources.
- **User prompt**: Provides the query and asks for structured output including sub-questions, source landscape, and summary.

### State Writes

The background post-processor writes directly to workflow state:

- `background_summary` -- the summary text
- `data_landscape` -- the full landscape dict
- `query_decomposition` -- list of sub-questions
- `discovered_sources` -- list of discovered source records

These are used by the planner to create source-aware research plans.

### Events Emitted

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `BackgroundCompletedEvent` | `background_completed` | `sources_discovered`, `data_landscape_summary`, `data_landscape`, `query_decomposition` |

### YAML Example

```yaml
- id: background
  type: agent
  label: Background Investigator
  config:
    subtype: background
    model_tier: simple
    output_key: background
    tools: [web_search]
    max_tool_calls: 5
```

### Usage Tips

- Run after the coordinator to prime the sources pool and data landscape before planning.
- The planner automatically switches to source-aware prompts when `data_landscape` or `discovered_sources` exist in state.
- Typically uses `web_search` tool with a low `max_tool_calls` (3-5) to keep latency down.
- Feeds context to the planner so it can create better-targeted research steps.

---

## 3. Planner

Creates a numbered research plan with ordered steps. Supports depth-aware step counts and source-aware planning when background discovery data is available.

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `analytical` |
| `output_key` | `plan` |
| `output_format` | `json` (enforced by enrichment) |
| `input_keys` | `[query, background]` |
| `tools` | `[]` (none) |
| `pool_writes` | `[]` (none) |
| `pool_tools` | `[]` (none) |

### Output Model -- `PlanOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | -- | Plan title summarizing the research approach |
| `thought` | `str` | -- | Reasoning behind the plan structure |
| `steps` | `list[PlanStepOutput]` | -- | Ordered research steps |
| `has_enough_context` | `bool` | `False` | Whether current context is sufficient to skip research |
| `iteration` | `int` | `1` | Plan iteration number (increments on replan) |

**`PlanStepOutput` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | -- | Unique step identifier |
| `title` | `str` | -- | Step title |
| `description` | `str` | `""` | Detailed step description |
| `step_type` | `Literal["research", "analysis"]` | `"research"` | Step category |
| `needs_search` | `bool` | `True` | Whether the step requires search tools |
| `source_hints` | `list[SourceHintOutput]` | `[]` | Hints for which sources to query |
| `exclude_sources` | `list[str]` | `[]` | Sources to skip for this step |

### Key Prompts

The planner dynamically selects prompts based on available state:

- **With `data_landscape`**: Uses source-aware system and user prompts that incorporate the landscape data for targeted planning.
- **With `discovered_sources` only**: Uses source-aware system prompt with a no-landscape user prompt variant.
- **Without either**: Uses standard planner prompts focused on query decomposition and research step ordering.

### Events Emitted

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `PlanCreatedEvent` | `plan_created` | `plan_id`, `title`, `thought`, `steps`, `iteration`, `has_enough_context` |

### YAML Example

```yaml
# Typically used inside plan_and_execute as the planner_node:
- id: research_cycle
  type: plan_and_execute
  label: Research Cycle
  config:
    planner:
      subtype: planner
      model_tier: analytical
      output_key: plan
    items_path: steps
    item_state_key: current_step
    body:
      # ... researcher config ...
    evaluator:
      # ... reflector config ...
    max_iterations: 10
    min_iterations: 2
    max_replan_cycles: 3
```

### Usage Tips

- Typically embedded in a `plan_and_execute` node as the planner, not as a standalone node.
- The `items_path: steps` setting tells plan-and-execute to iterate over the plan's `steps` list.
- Source hints in `PlanStepOutput` let the researcher know which enterprise sources to prioritize for each step.
- On replan cycles, the planner receives updated context including what has already been researched.

---

## 4. Researcher

Executes a single research step using tools. Uses the ReAct loop (LLM-controlled tool calls) to investigate a step from the plan, performing web searches, crawling pages, and querying pools.

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `analytical` |
| `output_key` | `findings` |
| `output_format` | `json` (enforced by enrichment) |
| `input_keys` | `[query, current_step, plan]` |
| `tools` | `[web_search, web_crawl]` |
| `pool_writes` | `[{pool: sources, extract: sources}]` |
| `pool_tools` | `[pool_search]` |
| `max_tool_calls` | `15` (default set by enrichment when 0) |

### Output Model -- `ResearcherOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_queries` | `list[str]` | `[]` | Queries used during research |
| `observation` | `str` | `""` | Synthesized observation from findings |
| `key_points` | `list[str]` | `[]` | Key points discovered |
| `sources_used` | `list[str]` | `[]` | URLs of sources accessed |
| `research_status` | `Literal["ok", "blocked", "insufficient_data"]` | `"ok"` | Status of the research step |
| `blocking_reason` | `str \| None` | `None` | Reason if status is `blocked` |
| `findings` | `str` | `""` | Full text findings |
| `sources_found` | `int` | `0` | Number of sources discovered |

### Key Prompts

- **System prompt**: Instructs the LLM to act as a research investigator, using available tools to find information relevant to the current step. Emphasizes source diversity and thoroughness.
- **User prompt**: Provides the original query, current plan step details, and asks the researcher to investigate using available tools.

### Events Emitted

The researcher itself emits no additional domain events. The ReAct loop automatically emits:

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `ToolCallEvent` | `tool_call` | `tool_name`, `arguments` |
| `ToolResultEvent` | `tool_result` | `tool_name`, `result_summary`, `source_count`, `tool_success` |
| `ToolCacheHitEvent` | `tool_cache_hit` | `tool_name`, `cache_key` |

### YAML Example

```yaml
- id: researcher
  type: agent
  label: Researcher
  config:
    subtype: researcher
    model_tier: analytical
    output_key: findings
    tools: [web_search, web_crawl]
    pool_writes:
      - pool: observations
        extract: findings
      - pool: sources
        extract: sources
    pool_inject:
      - pool: observations
        threshold: 0
        max_items: 10
        max_item_chars: 500
    max_tool_calls: 15
```

### Usage Tips

- The `max_tool_calls` default of 15 balances thoroughness with latency. Override lower (8-10) for faster runs or higher (20+) for extended research.
- Pool writes automatically deduplicate sources by URL and observations by content hash.
- `pool_inject` lets the researcher see what has already been found in previous steps, avoiding redundant searches.
- Enterprise tools (vector search, Genie, knowledge assistant) can be added to the `tools` list for hybrid web + enterprise research.
- The enrichment enforces `output_format: json` even if set to `text` or `markdown` in YAML.

---

## 5. Reflector

Evaluates research progress after each step and decides whether to CONTINUE, ADJUST, or COMPLETE. Provides step-by-step reflection that is central to the framework's adaptive research approach.

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `analytical` |
| `output_key` | `reflection` |
| `output_format` | `json` |
| `input_keys` | `[query, plan_summary, findings, current_step, remaining_steps, total_steps, steps_completed, min_steps, step_title, iteration, observation, all_observations, sources_count, source_topics, source_quality]` |
| `tools` | `[]` (none) |
| `pool_writes` | `[]` (none) |
| `pool_tools` | `[]` (none) |

### Output Model -- `ReflectionOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decision` | `Literal["continue", "adjust", "complete"]` | -- | Research continuation decision |
| `reasoning` | `str` | -- | Explanation for the decision |
| `suggested_changes` | `list[str] \| None` | `None` (normalized to `[]`) | Suggested modifications when decision is `adjust` |

### Key Prompts

- **System prompt**: Instructs the LLM to evaluate coverage, source quality, and research depth relative to the original query. Defines the three decision options and when each is appropriate.
- **User prompt**: Provides the query, plan summary, current step findings, accumulated observations, source counts, and asks for a structured decision.

### Events Emitted

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `ReflectionDecisionEvent` | `reflection_decision` | `decision`, `reasoning` |

### YAML Example

```yaml
# Typically used as the evaluator in a plan_and_execute node:
- id: research_cycle
  type: plan_and_execute
  label: Research Cycle
  config:
    planner:
      subtype: planner
      model_tier: analytical
      output_key: plan
    items_path: steps
    item_state_key: current_step
    body:
      id: researcher
      type: agent
      label: Researcher
      config:
        subtype: researcher
        # ...
    evaluator:
      subtype: reflector
      model_tier: analytical
      output_key: evaluation
      pool_inject:
        - pool: observations
          threshold: 0
    max_iterations: 10
    min_iterations: 2
    max_replan_cycles: 3
```

### Usage Tips

- When used as the evaluator in `plan_and_execute`, a `complete` decision ends the research cycle, `continue` moves to the next step, and `adjust` (mapped to `replan` in plan-and-execute) triggers replanning.
- The reflector has the richest set of `input_keys` among all subtypes -- it receives comprehensive context about research progress.
- Use `pool_inject` to give the reflector access to accumulated observations for better coverage assessment.
- The `min_iterations` setting on plan-and-execute prevents premature completion by the reflector.

---

## 6. Synthesizer

Generates the final markdown report from accumulated research. Supports three grounding modes for citation verification: `none` (no verification), `classical_lite` (draft then verify), and `reclaim` (interleaved generation with inline citations).

### Default Configuration

| Setting | Default Value |
|---------|--------------|
| `model_tier` | `complex` |
| `output_key` | `report` |
| `output_format` | `markdown` |
| `input_keys` | `[query, plan]` |
| `tools` | `[]` (none) |
| `pool_writes` | `[]` (none) |
| `pool_tools` | `[pool_search]` |
| `max_tool_calls` | `10` (default); `5` in reclaim mode |
| `grounding_mode` | `classical_lite` (resolved default) |

### Output Model -- `SynthesizerOutput`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `report` | `str` | -- | Final markdown report |
| `structured_output` | `Any \| None` | `None` | Optional structured data (verification details in reclaim mode) |

### Grounding Modes

| Mode | Description |
|------|-------------|
| `none` | No citation verification. The LLM generates the report directly. |
| `classical_lite` | Generates a draft report first, then runs the 7-stage citation verification pipeline on it. |
| `reclaim` | Interleaved generation where the LLM produces claims with inline citation markers, verified as they are generated. Specialized prompts enforce `[N]` citation format. |

Set the mode explicitly via `grounding_mode` in the YAML config, or let the synthesizer resolve it from `output_schema` settings.

### Key Prompts

- **Standard mode**: System prompt instructs the LLM to synthesize all research findings into a well-structured markdown report. User prompt provides the query, plan summary, and all observations and sources.
- **Reclaim mode**: Specialized system prompt enforces strict citation rules (`[N]` markers for every factual claim), brevity, and grounding. User prompt includes word count targets and available evidence indices.

### Events Emitted

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `SynthesisStartedEvent` | `synthesis_started` | `total_observations`, `total_sources` |

When grounding mode is `classical_lite` or `reclaim`, additional verification events are emitted:

| Event | `event_type` | Key Fields |
|-------|-------------|------------|
| `ClaimGeneratedEvent` | `claim_generated` | `claim_text`, `claim_index`, `citation_keys`, `claim_role` |
| `ClaimVerifiedEvent` | `claim_verified` | `claim_index`, `verdict`, `confidence`, `verification_method` |
| `CitationCorrectedEvent` | `citation_corrected` | `claim_index`, `action`, `original_key`, `corrected_key` |
| `NumericClaimDetectedEvent` | `numeric_claim_detected` | `claim_index`, `numeric_value`, `verification_status` |
| `VerificationSummaryEvent` | `verification_summary` | `total_claims`, `verified_claims`, `corrected_citations`, `removed_claims`, `softened_claims`, `overall_confidence` |

### YAML Example

```yaml
# Standard synthesizer (classical_lite grounding by default)
- id: synthesizer
  type: agent
  label: Report Synthesizer
  config:
    subtype: synthesizer
    model_tier: complex
    output_key: report
    pool_inject:
      - pool: observations
        threshold: 0
      - pool: sources
        threshold: 0
    pool_tools:
      - observations
      - sources
    max_tool_calls: 10
```

```yaml
# Synthesizer with explicit reclaim mode
- id: synthesizer
  type: agent
  label: Verified Report
  config:
    subtype: synthesizer
    model_tier: complex
    output_key: report
    grounding_mode: reclaim
    output_schema:
      target_word_count: 600
      max_tokens: 2000
      generation_mode: strict
      enable_are_retrieval: false
    pool_inject:
      - pool: observations
        threshold: 0
      - pool: sources
        threshold: 0
```

### Usage Tips

- Always place last in the pipeline, after all research is complete.
- Use `pool_inject` to feed accumulated observations and sources into the synthesis prompt.
- The `complex` model tier is recommended for best report quality, but `analytical` works for faster runs.
- In reclaim mode, the synthesizer has a custom `_execute` hook that runs the full 7-stage citation verification pipeline (evidence selection, interleaved generation, confidence classification, NLI verification, citation correction, numeric QA, ARE retrieval).
- The synthesizer collects sources from the `sources` pool and observations from both state `findings` entries and the `observations` pool, with deduplication.

---

## Pipeline Composition

The typical pipeline order follows four phases:

```
Coordinator --> Background --> Plan-and-Execute(Planner --> Researcher --> Reflector) --> Synthesizer
```

In a YAML workflow, this maps to a `sequence` root with:

1. **Coordinator** agent node -- classifies the query
2. **Background** agent node -- gathers initial context
3. **Plan-and-execute** meta-node containing:
   - **Planner** as `planner` -- creates research steps
   - **Researcher** as `body` -- executes each step
   - **Reflector** as `evaluator` -- decides continue/adjust/complete after each step
4. **Synthesizer** agent node -- produces the final report

The plan-and-execute node handles the iteration loop, replan cycles, and early completion logic automatically.

---

## Overriding Defaults

Every default can be overridden in YAML. The `enrich_config` hook only fills in values that are not already set, so explicit YAML values always take precedence.

```yaml
- id: my-researcher
  type: agent
  label: Custom Researcher
  config:
    subtype: researcher
    model_tier: complex          # Override default 'analytical'
    max_tool_calls: 10           # Override default 15
    tools: [web_search]          # Remove web_crawl, keep only search
    system_prompt: |             # Override default prompt entirely
      You are a specialized financial researcher.
      Focus exclusively on SEC filings and financial reports.
      Always cite specific document sections.
    pool_writes:
      - pool: observations
        extract: findings
      - pool: sources
        extract: sources
```

Override examples for other subtypes:

```yaml
# Override coordinator to always route to full research
- id: coordinator
  type: agent
  config:
    subtype: coordinator
    system_prompt: |
      Always classify queries as complex requiring full research.
      Never return a direct response.

# Override synthesizer grounding mode
- id: synthesizer
  type: agent
  config:
    subtype: synthesizer
    grounding_mode: none         # Skip citation verification
    model_tier: analytical       # Override default 'complex'
```

When overriding prompts, note that user prompts use Jinja-style `{variable}` template placeholders. The harness auto-detects required input keys from the template, so your custom prompt's variables determine what state values are injected.

---

## See Also

- [Agent System](../concepts/agent-system.md)
- [Custom Agents](custom-agents.md)
- [Agent Config Reference](../reference/agent-config-reference.md)
- [YAML Workflow Authoring](yaml-workflow-authoring.md)
