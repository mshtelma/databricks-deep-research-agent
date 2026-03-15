# 011 — Workflow Orchestration Architecture

> Multi-agent orchestration via composable workflow trees with 8 node types,
> immutable append-only state, declarative conditions, tool-accessible
> research pools, standard agent subtypes, verification/grounding integration,
> parameterized workflow templates, and UC tool integration.

---

> **Revision Notes (2026-03-08 Planning Session)**
>
> The following sections of this architecture document are superseded by decisions
> made during the planning session. See `plan.md` and `research.md` for the
> authoritative versions.
>
> | Section | Change |
> |---------|--------|
> | §9.3 (Agent Delegation Pattern) | Removed — agents are fully ported to framework builtins, no "simplified builtin + app override" pattern. |
> | §14.2 (Gradual Migration) | Removed — full replacement, no feature flag, no gradual migration path. |
> | §15.1 (Role Registries / Dynamic Subtype Registration) | Deferred beyond P0. |
> | §19 (Implementation Phases) | Replaced by P0a/P0b/P0c/P0d phasing in `plan.md`. |
> | LLM Client sections | `LLMClient` Protocol replaced by `FrameworkLLMClient` wrapping `AsyncOpenAI` directly. See `contracts/llm_client.py`. |
> | Tool Protocol sections | Updated: single `definition` property, `validate_arguments()`, `ToolResult.success`, `data` not `metadata`. See `contracts/tool_protocol.py`. |
> | Event sections | Events now Pydantic `BaseModel` with `event_type: Literal[...]` discriminator (not dataclass). See `contracts/events.py`. |
> | Agent Subtypes | 6 subtypes: coordinator, researcher, planner, reflector, synthesizer, background. Classic researcher and simple synthesizer dropped. Coordinator and background added as framework builtins. |
> | §2 (Node Types) | 8 types (plan_and_execute added). |
> | §4.6 | New: PlanAndExecuteNodeConfig (replaces ResearchLoopNodeConfig). |
> | §5.3 (Pool Access) | pool_inject for small pools. |
> | §7.6 | New: Typed output models per subtype. |
> | §8.6 | New: UrlRegistry on ToolContext. |
> | §9 (Subtypes) | 6 subtypes (background added). Domain events per subtype. |
> | §9.5 | New: Domain events from builtin subtypes. |
> | §15.3 | New: DomainContextTracker replaces event_mapper + state_bridge. |
> | LLM Client | FrameworkLLMClient gains rate limiting (EndpointHealth, ModelTierConfig, fallback). |
> | Templates (§12) | BestOfN, SelfCritique, Debate, MajorityVote deferred beyond P0. |
> | Gates (§8) | HITL gates deferred beyond P0. |
> | Subworkflows (§2.7) | Subworkflow node type defined but implementation deferred beyond P0. |
> | DataFlowGraph (§6) | Static data flow analysis deferred beyond P0. |

---

## 1. Core Idea

A workflow is a **tree of nodes**. The tree structure IS the control flow — no
explicit edges, no graph traversal. Each node type determines how its children
execute. Users compose 8 primitive node types to express any multi-agent pattern.

```
sequence                           <- root
├── agent: "Planner"               <- child 1 (leaf)
├── loop                           <- child 2
│   └── sequence                   <- loop body
│       ├── agent: "Researcher"    <- does web_search, web_crawl
│       └── agent: "Reflector"     <- writes "reflection_decision"
└── agent: "Synthesizer"           <- child 3 (leaf)
```

Indentation = hierarchy. Reading top-to-bottom = execution order. A user can
look at this tree and immediately understand what happens.

### Why a tree and not a graph

The user asked for **linear/hierarchical** orchestration. A tree gives both:
- **Linear**: `sequence` node — children run top to bottom
- **Hierarchical**: nesting — any node can contain other nodes

A graph with explicit edges adds complexity without expressiveness. Every useful
pattern from SOTA research (critique, voting, MoA, debate) decomposes into a
tree of 8 primitives. Arbitrary edges (`goto`-style) make workflows harder to
reason about, debug, visualize, and cancel.

The only construct a graph enables that a tree doesn't is cross-level cycles
(e.g., "jump from step 5 back to step 2"). But the `loop` node handles all
iteration needs, and the `conditional` node handles all branching needs. No
`goto` required.

---

## 2. The 8 Node Types

| #  | Type            | Purpose                          | Has children? | State interaction                |
|----|-----------------|----------------------------------|---------------|----------------------------------|
| 1  | `agent`         | Run an LLM (with optional tools) | No (leaf)     | Reads `input_keys`, writes `output_key` |
| 2  | `tool`          | Run a function deterministically | No (leaf)     | Reads `input_mapping`, writes `output_key` |
| 3  | `sequence`      | Execute children in order        | Yes           | Pass-through                     |
| 4  | `parallel`      | Execute children concurrently    | Yes           | Each child writes its own key    |
| 5  | `loop`          | Repeat single child              | Yes (exactly one) | Checks `until` condition after each iteration |
| 6  | `conditional`   | Pick one child to execute        | Yes           | Evaluates `conditions` to select branch |
| 7  | `subworkflow`   | Delegate to a saved workflow     | No (leaf-like)| Isolated child state with I/O mapping |
| 8  | `plan_and_execute` | Plan-execute-evaluate cycle      | Yes (inline configs) | Manages plan-item-evaluate cycle with continue/replan/complete |

**That's it.** Every SOTA pattern (critique, best-of-N, debate, MoA, dynamic
planning) is a specific arrangement of these 8 types. Convenience patterns like
BestOfN and SelfCritique are **parameterized subworkflow templates** (see §12),
NOT additional node types.

### Why not more types?

The first draft had 15+ types (`CritiqueNode`, `VotingNode`, `DebateNode`,
`MixtureNode`, etc.). Analysis showed most were compositions of primitives:

- "Best-of-N" = `parallel` + `agent(judge)` in a `sequence`
- "Majority vote" = `parallel` + `tool(vote_counter)` in a `sequence`
- "Critique" = `loop(sequence(generator, critic))`
- "Debate" = `loop(sequence(side_a, side_b))` -> `agent(judge)`
- "MoA" = `sequence(parallel(layer1), parallel(layer2), agent(final))`

Adding dedicated types for these would:
1. Increase the learning curve (15 concepts vs 8)
2. Create semantic overlap (when to use `VotingNode` vs `parallel` + judge?)
3. Complicate the executor (15 code paths vs 8)

The `plan_and_execute` type is the one exception — it encapsulates the plan →
execute → evaluate cycle with continue/replan/complete semantics that
cannot be cleanly expressed with generic `loop` + `conditional` composition
(the replan counter, item-aware iteration, and evaluation-driven control flow
require specialized logic).

Instead of more types, we provide **parameterized workflow templates** (§12) —
builtin generators that take parameters and produce trees of 8 primitives.
Templates are not special; they're just trees. Adding new patterns is purely
additive (template + params schema) — no framework changes, no new NodeType
values.

---

## 3. Data Structures

### 3.1 WorkflowNode — The universal building block

```python
class WorkflowNode(BaseModel):
    """Single node in a workflow tree. Recursive structure."""

    id: str                                 # Unique within the workflow
    type: NodeType                          # One of the 8 types
    label: str                              # Human-readable: "Research Loop"
    config: dict[str, Any] = {}             # Type-specific (see §4)
    children: list["WorkflowNode"] = []     # Empty for leaf nodes
    gate: "GateConfig | None" = None        # Optional HITL gate (see §13.9)
```

Validation rules (checked at load time):
- `agent`, `tool`, `subworkflow`: `children` MUST be empty
- `sequence`, `parallel`: `children` MUST have >= 1 entry
- `loop`: `children` MUST have exactly 1 entry
- `conditional`: `children` MUST have >= 1 entry (gate pattern: 1 condition + 1 child = execute-or-skip)
- `plan_and_execute`: `children` MUST be empty; config MUST have `planner`, `body` inline configs; `evaluator` is optional
- All `id` values MUST be unique within the tree

### 3.2 WorkflowDefinition — A complete workflow

```python
class PoolConfig(BaseModel):
    """Declaration of a shared research pool."""

    name: str                              # "observations", "sources", "evidence"
    item_type: Literal["text", "source", "evidence", "claim", "any"] = "text"
    dedup_key: str | None = None           # Field for dedup (e.g., "url" for sources)
    max_items: int | None = None           # Capacity limit (oldest evicted)


class CheckpointConfig(BaseModel):
    """Configuration for workflow state checkpointing (see §13.7)."""
    enabled: bool = True
    granularity: Literal["every_node", "every_n"] = "every_node"
    checkpoint_interval: int = 1            # Checkpoint every N leaf nodes


class WorkflowDefinition(BaseModel):
    """A complete, self-contained workflow definition."""

    id: str                                 # UUID
    name: str                               # "Default Deep Research"
    description: str | None = None
    version: int = 1
    root: WorkflowNode                      # The tree
    required_inputs: list[str] = ["query"]  # State keys this workflow expects
    output_keys: list[str] = ["report"]     # State keys this workflow produces
    pools: list[PoolConfig] = []            # Shared research pools (see §5.7)
    checkpoint: CheckpointConfig = CheckpointConfig()    # State checkpointing (§13.7)
    token_budget: int = 0                   # Max total tokens, 0 = unlimited (§13.8)

    @cached_property
    def data_flow_graph(self) -> "DataFlowGraph":
        """Computed at load time — static analysis of data dependencies (§5.10)."""
        from deep_research.agent.workflow.data_flow import build_data_flow_graph
        return build_data_flow_graph(self)
```

`required_inputs` and `output_keys` form the **contract** for subworkflows.
A parent can check at load time that it provides all required inputs and that
the subworkflow will produce expected outputs.

`pools` declares named accumulation points for multi-producer data. Any number
of agents at any position in the tree can contribute to a pool. This is distinct
from `state.data` which is single-producer per output_key. See §5.7 for full
pool mechanics.

---

## 4. Type-Specific Configs

### 4.1 AgentNodeConfig

```python
class PoolWriteConfig(BaseModel):
    """Declares that this agent's output should also be written to a pool."""
    pool: str                              # Pool name
    content_key: str | None = None         # Extract field from JSON output; None = full output


class AgentNodeConfig(BaseModel):
    """Configuration for an LLM agent node."""

    # Identity
    role: str                               # "Researcher", "Reflector", etc.
    subtype: str | None = None              # Standard agent subtype (see §9)
    system_prompt: str | None = None        # Inline system prompt
    system_prompt_template: str | None = None  # Template ref (by name)

    # Model selection
    model_tier: str = "analytical"          # simple | analytical | complex
    model_endpoint: str | None = None       # Direct endpoint override (takes precedence)
    temperature: float | None = None
    max_tokens: int | None = None

    # State interaction
    input_keys: list[str] = []              # State keys to inject into prompt context
    output_key: str = "output"              # Where to store this agent's output
    output_mode: Literal["replace", "append"] = "replace"  # Read hint: replace→get(), append→get_all()

    # Pool interaction (see §5.7)
    pool_writes: list[PoolWriteConfig] = [] # Write output to these pools
    pool_tools: list[str] = []              # Pools this agent can search via auto-generated tools

    # Conversation compaction (see §5.12)
    context_budget_tokens: int = 0          # Max conversation tokens (0 = model's context window)
    compact_keep_recent_turns: int = 5      # Turns to keep in full during compaction

    # Output parsing
    output_format: Literal["text", "json"] = "text"
    output_schema: dict[str, Any] | None = None
    # Agent output contract (JSON Schema). Triple use:
    # 1. LLM hint: included in response_format when output_format="json"
    # 2. Runtime validation: executor validates parsed output against schema
    # 3. Static analysis: DataFlowGraph validates dot-path references
    # Built-in subtypes provide defaults. User-provided values override.
    # For text-format agents, informational only (e.g. {"type": "string"}).

    # Tools (agent decides when to call)
    tools: list[ToolRef] = []

    # ReAct loop limits (only used when tools are present)
    max_tool_calls: int = 20

    # User prompt construction
    user_prompt_template: str | None = None  # Safe template: "{{query}}\n{{findings}}" (see §7.5)
    # If None, auto-constructed from input_keys (see §7); pool access is via pool_tools

    # Output parse failure handling (see §7.4)
    parse_failure: "ParseFailureConfig | None" = None

    # Verification & grounding (see §11)
    verification: "VerificationConfig | None" = None
```

### 4.2 ToolNodeConfig

```python
class ToolNodeConfig(BaseModel):
    """Configuration for a deterministic tool/function call."""

    ref: ToolRef                            # Which tool to call
    input_mapping: dict[str, str] = {}      # state_key -> function_parameter
    output_key: str = "tool_result"         # Where to store the result
    output_schema: dict[str, Any] | None = None
    # Tool output contract (JSON Schema). If None and ref is a builtin,
    # falls back to BUILTIN_TOOL_SCHEMAS[ref.name].
```

No LLM involved. The tool always runs at this point in the workflow with the
mapped inputs. Use for: data fetching, transformations, vote counting.

### 4.3 LoopNodeConfig

```python
class LoopNodeConfig(BaseModel):
    """Configuration for a loop node."""

    until: Condition                        # Exit condition (checked AFTER each iteration)
    max_iterations: int = 10               # Safety limit
    min_iterations: int = 0                # Minimum before exit condition is checked
```

Semantics: **do-while with min/max bounds.**
1. Execute `children[0]`
2. Increment iteration count
3. If `iteration_count < min_iterations` -> goto 1
4. If `evaluate(until, state)` is True -> exit loop
5. If `iteration_count >= max_iterations` -> exit loop
6. Goto 1

Why do-while (not while-do)? Because in research workflows, you always want at
least one iteration before checking. The `min_iterations` config handles cases
where you want even more guaranteed iterations (e.g., "research at least 3
steps before considering completion").

### 4.4 ConditionalNodeConfig

```python
class ConditionalNodeConfig(BaseModel):
    """Configuration for a conditional branch node."""

    conditions: list[Condition]             # conditions[i] -> children[i]
    # If len(children) > len(conditions), last child is the default branch.
    # If len(children) == len(conditions), there is no default branch.
    # If no condition matches AND no default, execution continues (no-op / gate pattern).
    #
    # Gate pattern: 1 condition + 1 child = "execute child if condition matches,
    # otherwise skip." No need for placeholder no-op agents.
```

Evaluation: conditions are checked in order. First match wins. This is an
**if / elif / else** chain, not a switch.

### 4.5 SubworkflowNodeConfig

```python
class SubworkflowNodeConfig(BaseModel):
    """Configuration for delegating to another workflow."""

    ref: str                                # Workflow ID, name, OR "builtin:pattern_name"
    params: dict[str, Any] = {}             # Template parameters (only for builtin: refs)
    input_mapping: dict[str, str] = {}      # parent_key -> child_key
    output_mapping: dict[str, str] = {}     # child_key -> parent_key
    # If output_mapping is empty, child's "output" key maps to parent's output_key
    output_key: str = "subworkflow_result"  # Convenience: where to put child's main output
    pool_writes: list[PoolWriteConfig] = [] # Write output to pools after completion
    pool_mode: Literal["shared", "isolated"] = "shared"  # Pool sharing strategy
```

The `pool_writes` field on subworkflows enables the **caller** to direct the
subworkflow's output into a pool. This is essential when the parent workflow
needs the subworkflow result in a named pool for downstream agents (e.g., a
`builtin:self_critique` subworkflow producing company intel that a synthesizer
later searches via `pool_tools`). Pool writes are processed after output mapping
completes — the value written is `state.get(output_key)`.

When `ref` starts with `"builtin:"`, the executor resolves it as a
**parameterized workflow template** (see §12). The `params` dict is validated
against the template's params schema, and a generator function produces a
`WorkflowDefinition` which is executed as a normal subworkflow.

### 4.6 PlanAndExecuteNodeConfig

```python
class PlanAndExecuteNodeConfig(BaseModel):
    """Configuration for a plan-execute-evaluate cycle.

    Domain-neutral pattern: a planner produces a list of items, a body
    processes each item, and an optional evaluator decides whether to
    continue, replan, or complete. Used by Deep Research for the
    planner → researcher → reflector cycle, but equally applicable to
    code review (changes), content generation (sections), etc.
    """

    planner: AgentNodeConfig              # Produces structured output with items list
    items_path: str = "steps"             # Dot-path to items in planner output
    item_state_key: str = "current_item"  # State key for current item
    body: WorkflowNode                    # Processes each item
    evaluator: AgentNodeConfig | None = None  # Optional: continue/replan/complete
    max_iterations: int = 10              # Total body executions across all cycles
    min_iterations: int = 1               # Before "complete" is honored
    max_replan_cycles: int = 3            # Max planner re-invocations
    complete_on_exhaustion: bool = True   # Exit when all items done (vs. re-plan)
```

**Execution semantics:**

1. Run planner (creates output with items at `items_path`)
2. Extract items from planner output via `items_path` (e.g., `"steps"`, `"changes"`, `"sections"`)
3. For each item: write to `item_state_key` in state, execute body
4. If evaluator is configured: run evaluator, check `EvaluationOutput.decision`:
   - `continue` → proceed to next item
   - `replan` → break item loop, increment replan counter, re-run planner
     (if replan_count < max_replan_cycles). Planner reads completed items
     via `state.get_all(item_state_key)`.
   - `complete` → break item loop, exit plan_and_execute
5. If evaluator is None: process all items sequentially (simple plan-execute-all)
6. After all items exhausted: if `complete_on_exhaustion` is True, exit;
   otherwise re-plan
7. `min_iterations` prevents premature `complete` (evaluator decision is
   overridden to `continue` if items_processed < min_iterations)
8. Emits `ItemsExtractedEvent`, `ItemStartedEvent`/`ItemCompletedEvent`
   for each item, `EvaluationDecisionEvent` after each evaluation,
   `ReplanTriggeredEvent` on replan, `PlanAndExecuteExitEvent` on exit

**Key design decisions:**
- **`evaluator` is optional**: `None` = simple plan-execute-all. Present = adaptive loop.
- **`items_path` is configurable**: Research uses `"steps"`, code review uses
  `"changes"`, content uses `"sections"`. Framework doesn't care about domain naming.
- **`item_state_key`**: Before each body execution, executor writes current item
  to this state key. Body reads it via `input_keys`. Standard state mechanism.
- **Evaluator decisions**: `continue` / `replan` / `complete` (not `adjust` —
  `replan` is more descriptive).

**Why a dedicated node type?** The plan → execute → evaluate cycle has
semantics that cannot be cleanly expressed with generic `loop` + `conditional`:
- The replan counter is orthogonal to the item iteration counter
- Replan must re-run the planner with preserved context (completed items)
- The item loop iterates over planner-generated items, not a fixed count
- min_iterations must be checked against total items across all replan cycles

> **YAML safety**: All config models loaded from YAML MUST use
> `model_config = ConfigDict(extra='forbid')` to reject unknown fields.
> This catches typos and schema mismatches at load time. Applies to:
> `AgentNodeConfig`, `ToolNodeConfig`, `LoopNodeConfig`, `ConditionalNodeConfig`,
> `PlanAndExecuteNodeConfig`, `WorkflowDefinition`, `WorkflowNode`, `PoolConfig`.

---

## 5. State Management

This is the deepest section. Getting state right is what makes the whole
system coherent.

### 5.1 The State Object

State is **append-only**. No shared mutations, no overwrites. Every write is
recorded as a `StateEntry` with timestamp and producer, giving a full audit
trail. This is inherently safe for parallel execution.

```python
@dataclass(frozen=True)
class StateEntry:
    """Single state write. Immutable after creation."""
    node_id: str
    key: str
    value: Any
    timestamp: datetime


@dataclass
class WorkflowState:
    """Append-only state flowing through the workflow tree."""

    # -- Immutable inputs --
    query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # -- Append-only state log --
    # All inter-node communication goes through log entries.
    # Agents and tools write here via state.append(node_id, key, value).
    log: list[StateEntry] = field(default_factory=list)

    # -- Shared research pools --
    # Multi-producer accumulation points accessed via tools (see §5.7).
    pools: dict[str, "PoolState"] = field(default_factory=dict)

    # -- Execution context (read-only for nodes) --
    enterprise_tools: list["ResearchTool"] = field(default_factory=list)
    user_token: str | None = None
    model_overrides: dict[str, str] | None = None
    domain_filter: Any | None = None

    # -- Executor bookkeeping --
    iteration_counts: dict[str, int] = field(default_factory=dict)  # node_id -> count
    is_cancelled: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # -- Checkpointing (§13.7) --
    checkpoint: "CheckpointState | None" = None

    # -- Token budget tracking (§13.8) --
    token_budget: "TokenBudget | None" = None

    # -- Tool call deduplication cache (§5.11) --
    tool_call_cache: "ToolCallCache | None" = None

    # --- Read interface ---

    # -- O(1) index for get() --
    _latest_index: dict[str, int] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        """Latest value for key (O(1) via _latest_index)."""
        idx = self._latest_index.get(key)
        return self.log[idx].value if idx is not None else None

    def get_all(self, key: str) -> list[Any]:
        """All values ever written for key (append-mode accumulation)."""
        return [e.value for e in self.log if e.key == key]

    def get_history(self, key: str) -> list[StateEntry]:
        """Full audit trail for a key."""
        return [e for e in self.log if e.key == key]

    # --- Write interface ---

    def append(self, node_id: str, key: str, value: Any) -> None:
        """Append a state entry. The ONLY way to write state."""
        self.log.append(StateEntry(
            node_id=node_id, key=key, value=value,
            timestamp=datetime.now(UTC),
        ))
        self._latest_index[key] = len(self.log) - 1
```

**Key properties:**
- `get()` returns the latest value for a key (last writer wins = replace semantics)
- `get_all()` returns all values ever written for a key (accumulated list = append semantics)
- `get_history()` returns full audit trail with timestamps and producer node IDs
- `append()` is the ONLY way to write state — no direct mutation
- The old `output_mode: replace` vs `append` distinction is now a **read pattern**: `get()` vs `get_all()`
- LangGraph creates new state versions per update for auditability; the append-only log achieves the same thing more efficiently

### 5.2 Two kinds of state

This is a critical distinction that the first draft missed:

**A) Explicit outputs (`state.log`)** — Agents and tools append their results
here via `state.append(node_id, output_key, value)`. This is the structured
data flow between nodes. Reading is via `state.get(key)` (latest value) or
`state.get_all(key)` (all accumulated values). The tree structure + key
declarations make the data flow visible and debuggable.

Example: researcher appends `state.append("researcher_1", "findings", "...")`,
reflector appends `state.append("reflector_1", "reflection", {...})`,
synthesizer reads via `state.get("reflection")`.

**B) Shared research pools (`state.pools`)** — Multi-producer accumulation
points for data that flows across many agents. Any number of agents at any
position in the tree can contribute. Pools have typed items, dedup, max
capacity, and async locks. Agents access pools via auto-generated search tools
(not prompt injection). See §5.7 for full mechanics.

Why the distinction matters:
- Explicit outputs are **append-only log entries**. `state.get(key)` returns
  the latest value (replace semantics). `state.get_all(key)` returns all
  values (accumulation semantics). Both are safe for parallel execution.
- Pools are **multi-producer** and accumulate across the entire workflow.
  Any agent can contribute from any position. Pool writes use async locks
  for dedup safety.

### 5.3 How nodes read state

Agents read state through two mechanisms:

**A) `input_keys` — small scalar values injected into the prompt.** The
executor reads from the state log via `state.get(key)` (latest value) or
`resolve_key(state, key)` for dot-path access (§5.6). These are for small,
always-needed values like `query`, `plan`, `reflection.decision`.

**B) `pool_tools` — large accumulated data accessed via search tools.** For
each pool name in `pool_tools`, the executor auto-generates search/retrieval
tools that the agent can call during its ReAct loop. These are for large
collections like observations, sources, and evidence. See §5.7 for the full
pool tools specification.

**Auto-inference**: When `input_keys` is empty, the executor auto-infers
them from `user_prompt_template` variable references and subtype defaults
(see §5.10 DataFlowGraph). Users only need to declare `input_keys`
explicitly when overriding auto-inference.

```python
# Executor builds the prompt context for an agent:
def _build_prompt_context(
    self, config: AgentNodeConfig, state: WorkflowState,
) -> dict[str, Any]:
    context = {}
    # Explicit state reads (small scalar values)
    for key in config.input_keys:
        value = resolve_key(state, key)  # Supports dot-paths (see §5.6)
        if value is not None:
            context[key] = value
    return context
    # NOTE: pool data is NOT injected here.
    # Pools are accessed via auto-generated tools (see §5.7).
```

If a key doesn't exist in state yet (e.g., first iteration of a loop before
any findings), it's silently omitted. The agent's prompt should be written to
handle missing context gracefully ("If findings are provided, build on them").

**Data access patterns:**

| Data type | Access method | Example | Why |
|-----------|--------------|---------|-----|
| Small scalar values | `input_keys` (injection) | query, plan, reflection decision | Small, always needed |
| Large accumulated data | `pool_tools` (tool search) | observations, sources, evidence | Large, need relevance filtering |
| Small accumulated data | `pool_inject` (conditional) | reflector reading observations | Below threshold, always-needed |
| External knowledge | `tools` (external APIs) | web_search, enterprise tools | New data acquisition |

### 5.4 How nodes write state

Agents declare `output_key` and optionally `pool_writes`. All state writes
go through `state.append()` — the ONLY mutation interface:

```python
def write_output(state: WorkflowState, node_id: str, key: str, value: Any) -> None:
    """Append agent output to state log. No modes — always appends."""
    state.append(node_id, key, value)


async def write_pools(
    state: WorkflowState,
    pool_writes: list[PoolWriteConfig],
    output: Any,
) -> None:
    """Write agent output to declared pools (Path A — explicit, configured)."""
    for pw in pool_writes:
        pool = state.pools.get(pw.pool)
        if pool is None:
            continue
        content = output
        if pw.content_key and isinstance(output, dict):
            content = output.get(pw.content_key, output)
        await pool.append_async(content)
```

Rules:
- Each agent writes to exactly ONE key (its `output_key`).
- All writes are **append-only** to the state log. The old `replace`/`append`
  mode distinction is now a **read pattern**: `state.get(key)` returns the
  latest value (replace semantics), `state.get_all(key)` returns all values
  (accumulation semantics).
- An agent CANNOT write to arbitrary state keys. Only its declared `output_key`.
  This constraint makes the data flow auditable — every write is recorded with
  `node_id`, `key`, `value`, and `timestamp`.
- After writing to `output_key`, the executor iterates `pool_writes` and
  appends to each declared pool.

### 5.5 Parallel safety

Since we use `asyncio` (cooperative multitasking on one thread), there's no
true parallelism. `asyncio.gather` interleaves coroutines at `await` points.

**Append-only state is inherently safe** — parallel children all call
`state.append(node_id, key, value)`. Since appends never overwrite, there are
no mutation conflicts even if children write to the same key. The `node_id`
on each entry tracks which child produced which value.

For consumers that want the latest value per child, `state.get_history(key)`
provides the full audit trail filtered by producer.

**Pool writes use async locks** — multiple parallel children may write to the
same pool. Each pool has its own `asyncio.Lock` for dedup safety:

```python
async def pool_append(pool: PoolState, item: Any) -> bool:
    async with pool._lock:
        # ... dedup check, capacity enforcement, append
```

This matches the existing pattern in `ResearchState._sources_lock`.

### 5.5.1 Agent Isolation

Agents never see `WorkflowState` directly. The executor constructs an
**immutable input snapshot** and receives a structured output:

```python
@dataclass(frozen=True)
class AgentInput:
    """Immutable. Constructed by executor, consumed by agent."""
    query: str
    context: dict[str, Any]       # Selected state values (executor picks from input_keys)
    instruction: str              # From system prompt / role config
    tools: list[ResearchTool]     # External tools + auto-generated pool tools
    # Agent cannot mutate this. It returns AgentOutput.


@dataclass
class AgentOutput:
    """What the agent produces. Executor decides where it goes."""
    content: str | dict           # Main output (text or parsed JSON)
    # Pool contributions from tool calls are captured automatically (Path B)
```

> **Why `AgentOutput` has no domain-specific fields (e.g. `citations`)**: The
> framework is generic. Domain data (citations, claims, scores, verdicts) flows
> as fields in structured `content: dict` when `output_format: json`. The
> `output_schema` declares what fields `content` will have. Downstream nodes
> access sub-fields via dot-path resolution (§5.6): e.g.
> `verified_section.grounded_text`, `verified_section.verification_summary`.
> This avoids special-casing any one domain concept.

The executor:
1. Constructs `AgentInput` from the state log (using `state.get()` for latest values)
2. Resolves pool tools from `pool_tools` config
3. Runs the agent with the input
4. Appends `AgentOutput.content` to the state log via `state.append()`

This isolation ensures:
- Agents cannot mutate state directly
- All state changes go through the executor (auditable)
- Agent logic is testable in isolation (inject `AgentInput`, assert `AgentOutput`)

### 5.6 Dot-path resolution

State keys support dot-paths for nested access:

```python
def resolve_key(state: WorkflowState, key: str) -> Any:
    """Resolve a dot-separated key path from state.

    "findings"           -> state.get("findings")
    "reflection.decision" -> state.get("reflection")["decision"]
    "query"              -> state.query (special: top-level fields)
    "pool:observations"  -> state.pools["observations"].items (pool access)
    """
    # Pool access (for conditions)
    if key.startswith("pool:"):
        pool_name = key[5:]
        pool = state.pools.get(pool_name)
        return pool.items if pool else None

    # Special top-level fields
    if key == "query":
        return state.query

    parts = key.split(".")
    current = state.get(parts[0])
    for part in parts[1:]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
```

This enables conditions to check nested values:
```yaml
until:
  key: "reflection.decision"    # state.data["reflection"]["decision"]
  operator: eq
  value: "complete"
```

And agents with `output_format: json` + `output_schema` to write structured
dicts that downstream conditions can inspect field-by-field.

### 5.7 Shared Research Pools

Pools solve a fundamental need: **multi-producer accumulation**. In a research
workflow, multiple researchers at various positions in the tree need to
contribute findings to a shared collection that a synthesizer later reads.
The state log is single-producer per `output_key` (though parallel children can
safely append to the same key — see §5.5). Pools remove the need for
coordination — any number of agents write to the same pool with dedup.

#### Why pools are distinct from the state log

| Aspect | `state.log` (via `get`/`append`) | `state.pools["name"]` |
|--------|-------------------|-----------------------|
| Producers | Typically single per key | Multi (any agent, any position) |
| Access | Direct injection via `input_keys` | Via auto-generated search tools (`pool_tools`) |
| Cross-subworkflow | Isolated (only mapped keys cross) | Configurable: **shared** (default) or **isolated** (§10.2) |
| Features | Append-only log, audit trail | Typed items, dedup, max capacity, search index |
| Use case | "What did this agent just produce?" | "What has been found so far across ALL agents?" |

#### Pool declaration (WorkflowDefinition)

Pools are declared at the workflow definition level:

```yaml
pools:
  - name: observations
    item_type: text
  - name: sources
    item_type: source
    dedup_key: url
  - name: evidence
    item_type: evidence
  - name: claims
    item_type: claim
    max_items: 100
```

#### Pool runtime (WorkflowState)

```python
@dataclass
class PoolState:
    config: PoolConfig
    items: list[Any] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _text_hashes: set[str] = field(default_factory=set)  # SHA-256 hashes for text dedup (§5.11)

    async def append_async(self, item: Any) -> bool:
        """Append item to pool. Returns False if deduped."""
        async with self._lock:
            # Dict dedup via dedup_key
            if self.config.dedup_key and isinstance(item, dict):
                key_val = item.get(self.config.dedup_key)
                if key_val and any(
                    existing.get(self.config.dedup_key) == key_val
                    for existing in self.items
                    if isinstance(existing, dict)
                ):
                    return False
            # Text dedup via content hash (§5.11)
            if isinstance(item, str):
                import hashlib
                normalized = " ".join(item.lower().strip().split())
                text_hash = hashlib.sha256(normalized.encode()).hexdigest()
                if text_hash in self._text_hashes:
                    return False
                self._text_hashes.add(text_hash)
            self.items.append(item)
            if self.config.max_items and len(self.items) > self.config.max_items:
                self.items.pop(0)  # Evict oldest
            return True

    async def extend_async(self, items: list[Any]) -> int:
        """Append multiple items. Returns count actually added."""
        count = 0
        for item in items:
            if await self.append_async(item):
                count += 1
        return count

    def read_all(self) -> list[Any]:
        """Read all items. No lock needed (asyncio single-thread reads)."""
        return list(self.items)

    def search(self, query: str, top_k: int = 10) -> list[Any]:
        """Search pool items by relevance. See Pool Registry (below)."""
        # Delegated to PoolRegistry for hybrid BM25+vector search.
        raise NotImplementedError("Use PoolRegistry.search() instead")

    def get_recent(self, n: int = 10) -> list[Any]:
        """Return N most recent items."""
        return self.items[-n:]

    def count(self) -> int:
        """Return total item count."""
        return len(self.items)
```

**Note**: Pools are NEVER compacted. They are external stores accessed via
tools (see Pool Registry below). The old pool compaction mechanism is
removed — conversation compaction (§5.12) handles memory pressure instead.

#### Two paths for pool writes

**Path A: Agent output -> pool (explicit, configured via pool_writes)**

Agent produces LLM output. Executor appends to state log via
`state.append(node_id, output_key, value)`. THEN executor iterates
`config.pool_writes` and appends
to each declared pool. The content written can be the full output or a specific
field (via `content_key`).

```yaml
# Researcher writes findings to state.data AND to observations pool:
- type: agent
  config:
    subtype: researcher
    output_key: findings
    output_mode: append
    pool_writes:
      - pool: observations
```

**Path B: Tool side effects -> pool (implicit, automatic)**

When web_search/web_crawl/enterprise tools execute, they return
`ToolResult.sources`. The executor automatically appends these to the "sources"
pool. This is NOT configured per-agent — it's built-in tool behavior.

```python
# In executor, after tool execution:
if result.sources and "sources" in state.pools:
    await state.pools["sources"].extend_async(result.sources)
```

#### pool_writes vs output_key — NOT redundant

- `output_key` -> `state.log` -> read by the NEXT node via `input_keys` / `state.get(key)`
  -> "what did this agent just produce?"
- `pool_writes` -> `state.pools` -> searched by downstream agents via `pool_tools`
  -> "what has been found so far across ALL agents?"

A reflector reads `state.get("findings")` to evaluate the LATEST step.
A synthesizer searches the "observations" pool to find ALL relevant findings.

#### Pool access — via auto-generated tools (default) or conditional injection

Agents declare `pool_tools: list[str]`. For each pool name, the executor
auto-generates search/retrieval tools that the agent can call during its
ReAct loop. **By default, pool contents are accessed via tools, not prompt
injection.**

**Pool injection for small pools**: When an agent declares `pool_inject`, the
executor checks `pool.count()` against the configured threshold. If the pool
has ≤ threshold items, the formatted contents are injected directly into the
prompt context. Otherwise, only tool access is available. This hybrid approach
matches production behavior — the reflector and planner need full context of
a small number of observations, while the synthesizer uses tool search for
large pools.

```python
class PoolInjectConfig(BaseModel):
    pool: str               # Pool name
    threshold: int = 10     # Inject if pool.count() <= this
    format: str = "numbered"  # "numbered", "bullet", "json"
```

This is the same pattern already used by `EvidenceRegistry` + `HybridSearchIndex`
in the existing codebase (`agent/tools/evidence_registry.py`).

**Why tool access is strictly better than prompt injection:**

| Aspect | Prompt injection (old) | Tool access (new) |
|--------|----------------------|-------------------|
| Pool size limit | Bounded by context window | Unlimited |
| Relevance | Agent gets ALL items | Agent searches for RELEVANT items |
| Compaction needed | Yes (old §5.12) | No — pools are never compacted |
| Citation preservation | Compaction destroys citations | Sources always complete |
| Agent autonomy | Passive (gets what executor gives) | Active (searches for what it needs) |
| Small pools (< 10) | Works fine | Agent calls `get_all()` — same result |
| Large pools (100+) | Context overflow or lossy compaction | Agent calls `search()` — precise retrieval |
| Cost | Prompt tokens for all items | One tool call + tokens for relevant items only |

#### Pool Registry — generalized from existing EvidenceRegistry

The codebase already has production-tested hybrid search in
`agent/tools/evidence_registry.py`:
- BM25 keyword search (via `bm25s` library — already a dependency)
- Vector similarity (GTE embeddings, cosine, 0-1 normalized)
- Hybrid fusion: `0.6 * vector + 0.4 * bm25` (configurable alpha)
- Index-based access (LLM sees indices, not URLs — security)
- Access audit trail (tracks which items were read)

**Implementation plan:** Extract `HybridSearchIndex` from `evidence_registry.py`
into a shared module (e.g., `services/search/hybrid_index.py`). Make
`PoolRegistry` a generalization of `EvidenceRegistry`:

```python
class PoolRegistry:
    """Generalization of EvidenceRegistry for any pool type.

    Supports hybrid BM25 + vector search with 4-tier graceful degradation:
    1. Full hybrid (BM25 + vector): bm25s installed AND embedding model configured
    2. BM25 only: bm25s installed, no embedding model
    3. Keyword fallback: Neither available, simple word overlap
    4. Chronological: get_recent() always works
    """

    def __init__(self, pool: PoolState, *, llm_client: FrameworkLLMClient | None = None, alpha: float = 0.6):
        self._pool = pool
        self._llm_client = llm_client
        self._alpha = alpha
        # BM25 index created lazily on first search (optional dep)
        # Vector embeddings computed via llm_client.embed() if available

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Returns items with indices (not raw content — security)."""
        return self._index.search(query, top_k=top_k)

    def get_by_index(self, index: int) -> Any:
        """Index-based access (follows UrlRegistry/EvidenceRegistry pattern)."""
        return self._pool.items[index]

    def get_topics(self) -> dict:
        """Lightweight topic clustering for coverage analysis.
        No LLM needed — purely algorithmic (keyword frequency + grouping).
        Returns: {"total": N, "topics": [{"topic": "...", "count": N}, ...]}
        """

    def get_recent(self, n: int = 10) -> list[Any]:
        return self._pool.get_recent(n)

    def count(self) -> int:
        return self._pool.count()
```

#### Auto-generated pool tools

For each pool declared in `pool_tools`, the executor auto-generates tools
following the `ResearchTool` protocol:

```python
def make_pool_tools(pool: PoolState, registry: PoolRegistry) -> list[ResearchTool]:
    """Generate search/retrieval tools for a pool."""
    pool_name = pool.config.name
    return [
        PoolSearchTool(registry, pool_name),       # search_{pool}(query, top_k) → results
        PoolGetRecentTool(registry, pool_name),     # get_recent_{pool}(n) → results
        PoolCountTool(registry, pool_name),         # count_{pool}() → int
        PoolTopicsTool(registry, pool_name),        # topics_{pool}() → structured overview
        PoolGetByIndexTool(registry, pool_name),    # get_{pool}_by_index(index) → single item
    ]
```

The `topics_{pool}()` tool is critical for **coverage analysis** (reflector pattern):

```python
class PoolTopicsTool:
    """Returns topic clusters and coverage summary."""
    async def execute(self, arguments: dict) -> ToolResult:
        # Cluster items by keyword overlap / simple topic extraction
        # Returns: {"total": 47, "topics": [
        #   {"topic": "mRNA mechanism", "count": 12, "recent_index": 45},
        #   {"topic": "lipid nanoparticles", "count": 8, "recent_index": 41},
        #   ...
        # ], "earliest": "2025-03-07T10:00", "latest": "2025-03-07T10:45"}
```

This enables the reflector to assess coverage gaps WITHOUT dumping all items
into the prompt. The reflector can call `topics_observations()` to see what
topics have been researched, then `search_observations("topic")` to drill into
specific areas, and reason about what's missing.

#### Evidence pre-selection as tool node

The current evidence pre-selection flow (Stage 1 of `CitationVerificationPipeline`)
maps to a `tool` node in the tree:

```yaml
- type: tool
  label: "Evidence Pre-Selection"
  config:
    ref: {type: builtin, name: preselect_evidence}
    input_mapping: {query: "query"}
    output_key: evidence_ready
    # Reads from sources pool, writes to evidence pool
    # Curates and ranks evidence for synthesis
```

The synthesizer's `pool_tools: [evidence, sources]` then searches the curated
evidence pool (same as today's `EvidenceRegistry` pattern).

#### Complete pool lifecycle walkthrough

```
INITIALIZATION (from WorkflowDefinition.pools):
  state.pools = {
    "observations": PoolState(items=[], lock=Lock()),
    "sources": PoolState(items=[], lock=Lock(), dedup_key="url"),
  }
  Pool registries created, auto-tools generated for each pool.

-- Researcher iteration 1 executes (pool_tools: [observations]) --

  LLM calls web_search -> ToolResult with 3 SourceInfo objects
  -> Path B (automatic): sources pool = [Source1, Source2, Source3]

  LLM calls search_observations("mRNA") -> [] (empty, first iteration)
  LLM produces text: "mRNA carries genetic instructions..."
  -> output_key write: state.append("researcher", "findings", "mRNA carries...")
  -> Path A (pool_writes): observations pool = ["mRNA carries..."]

-- Researcher iteration 2 executes --

  LLM calls web_search -> 2 new SourceInfo + 1 duplicate (deduped)
  -> sources pool = [Source1, Source2, Source3, Source4, Source5]

  LLM calls search_observations("mRNA delivery") -> [item 0: "mRNA carries..."]
  LLM produces: "Lipid nanoparticles deliver mRNA..."
  -> state.append("researcher", "findings", "Lipid nanoparticles...")
  -> observations pool = ["mRNA carries...", "Lipid nanoparticles..."]

-- Reflector executes (pool_tools: [observations, sources]) --

  LLM calls topics_observations() -> {"total": 2, "topics": [
    {"topic": "mRNA mechanism", "count": 1},
    {"topic": "lipid nanoparticles", "count": 1}
  ]}
  LLM calls count_sources() -> 5
  LLM reasons: "2 observations across 2 topics with 5 sources. Need more
    research on immune response and safety profile."
  -> Outputs {"decision": "continue", "reasoning": "..."}

-- After more iterations, Reflector outputs {"decision": "complete"} --

-- Synthesizer executes (pool_tools: [observations, sources]) --

  LLM calls count_observations() -> 47
  LLM calls search_observations("mRNA mechanism") -> [12 relevant findings]
  LLM calls search_observations("safety profile") -> [8 findings]
  LLM calls search_sources("clinical trial efficacy") -> [7 sources]
  -> Generates comprehensive report with citations from retrieved items
```

#### Parallel pool safety

When parallel researchers both write to "observations":
1. Each pool has its own `asyncio.Lock`
2. Appends are serialized at `await` points (asyncio is single-threaded)
3. Order is non-deterministic but accumulation is correct
4. No data races possible

#### Pools in conditions

```yaml
# Exit loop when observations pool has 5+ items:
until:
  key: "pool:observations"
  operator: length_gt
  value: 4
```

### 5.8 Legacy field removal

The legacy accumulated side-effect fields (`sources`, `evidence_pool`, `claims`,
`all_observations`) on `WorkflowState` are **removed**. All accumulation goes
through pools exclusively. The backward compatibility bridge that shared list
objects between legacy fields and pools introduced data race risks with parallel
nodes and is no longer needed — pools are the canonical API for multi-producer
accumulation.

### 5.9 State lifecycle — complete example

This example uses a `plan_and_execute` node (§4.6) which encapsulates the
planner → researcher → evaluator cycle:

```
INITIAL STATE:
  query: "How does mRNA vaccination work?"
  log: []
  pools: {
    observations: PoolState(items=[]),
    sources: PoolState(items=[], dedup_key="url"),
  }

--- sequence starts ---

--- plan_and_execute starts ---

PLANNER (inline, output_key: "plan"):
  log: [
    StateEntry(node_id="planner", key="plan", value={title: "mRNA Research", steps: [...]})
  ]
  state.get("plan") -> {title: "mRNA Research", steps: [...]}
  -> emits PlanCreatedEvent(title="mRNA Research", steps=[...], iteration=1)

--- plan_and_execute item 1 ---
  -> emits ItemStartedEvent(item_index=0, item_summary="Mechanism of action", total_items=2)

BODY — agent "Researcher" (output_key: "findings", pool_writes: [{pool: observations}]):
  log: [
    ...,
    StateEntry(node_id="researcher", key="findings", value="mRNA is a molecule...")
  ]
  pools.sources.items: [Source1, Source2, Source3]               <- Path B auto
  pools.observations.items: ["mRNA is a molecule..."]           <- Path A explicit
  state.get("findings") -> "mRNA is a molecule..."
  -> emits ItemCompletedEvent(item_index=0, items_processed=1)

EVALUATOR (inline, output_key: "evaluation",
           pool_inject: [{pool: observations, threshold: 10}]):
  Observations injected (1 item <= threshold 10)
  log: [
    ...,
    StateEntry(node_id="evaluator", key="evaluation", value={"decision": "continue", ...})
  ]
  -> emits EvaluationDecisionEvent(decision="continue", reasoning="...", items_processed=1)

--- plan_and_execute: decision=continue -> next item ---

--- plan_and_execute item 2 ---
  -> emits ItemStartedEvent(item_index=1, item_summary="Delivery mechanisms", total_items=2)

BODY — agent "Researcher" (output_key: "findings"):
  log: [
    ...,
    StateEntry(node_id="researcher", key="findings", value="Lipid nanoparticles deliver...")
  ]
  pools.sources.items: [Source1, Source2, Source3, Source4, Source5]  <- accumulated
  pools.observations.items: ["mRNA is a molecule...", "Lipid nanoparticles..."]
  state.get("findings") -> "Lipid nanoparticles deliver..."           <- latest
  state.get_all("findings") -> ["mRNA is a molecule...", "Lipid..."]  <- all values

EVALUATOR (inline, output_key: "evaluation"):
  log: [..., StateEntry(node_id="evaluator", key="evaluation", value={"decision": "complete"})]
  -> emits EvaluationDecisionEvent(decision="complete", reasoning="...", items_processed=2)

--- plan_and_execute: decision=complete -> exit ---

AFTER agent "Synthesizer" (pool_tools: [observations, sources],
                           output_key: "report"):
  Synthesizer called: search_observations("mRNA mechanism") -> [relevant findings]
  Synthesizer called: search_sources("clinical trial") -> [relevant sources]
  -> emits SynthesisStartedEvent(total_observations=2, total_sources=5)
  log: [
    ...,
    StateEntry(node_id="synthesizer", key="report", value="# mRNA Vaccination: ...")
  ]

--- sequence ends ---
```

### 5.10 Data Flow Graph — Static Analysis

While `state.data` remains the runtime storage, a **DataFlowGraph** is built
at workflow load time to validate all data dependencies before execution. This
catches errors early (missing producers, key collisions) without requiring
users to manually declare every `input_key`.

```python
@dataclass(frozen=True)
class DataSlot:
    key: str                      # "findings", "plan", "reflection"
    producer_node_id: str         # Which node writes this
    schema: dict | None           # JSON Schema (from output_schema)
    mode: Literal["replace", "append"]

@dataclass(frozen=True)
class DataDependency:
    consumer_node_id: str
    slot_key: str
    producer_node_id: str         # Resolved ancestor that produces this
    required: bool                # False if might not exist (loop iteration 0)

@dataclass
class DataFlowGraph:
    slots: dict[str, DataSlot]
    dependencies: list[DataDependency]
    pool_producers: dict[str, list[str]]
    pool_consumers: dict[str, list[str]]
    warnings: list[str]
    errors: list[str]
    schema_warnings: list[str]    # Dot-path / schema mismatch warnings (step 4)
```

#### Resolution algorithm

1. **Collect producers**: Walk tree, record each node's `output_key` → `DataSlot`
2. **Validate parallel safety**: No two parallel children produce same `output_key`
3. **Resolve dependencies**: For each node, find which ancestor/preceding-sibling
   produces each needed key:
   - Sequence: node C reads keys from siblings A, B (they run before C)
   - Loop body: can read own previous iteration's keys (marked `required=False`
     for iteration 0)
   - Parent scope: inherit from ancestor's preceding siblings
   - Subworkflow boundary: ONLY `input_mapping`/`output_mapping` keys cross
   - `query` + `WorkflowDefinition.required_inputs`: always available
4. **Validate dot-paths against schemas**: For each dependency where the
   consumer uses a dot-path (e.g., `reflection.decision`), check the
   producer's `DataSlot.schema`. If the schema exists and does not contain
   the referenced sub-field, emit a schema warning. If the schema type is
   not `object`, warn that dot-path access will fail. Warnings are collected
   in `DataFlowGraph.schema_warnings`.

   | Check | Severity | Message |
   |-------|----------|---------|
   | Dot-path sub-field not in producer schema | Warning | `field 'Y' not in schema of 'X'. Available: [a, b, c]` |
   | Dot-path into non-object type | Warning | `'X' has type 'string', cannot access sub-field '.Y'` |
   | Condition operator incompatible with schema type | Warning | `operator 'gt' on field with type 'string'` |

5. **Report errors/warnings**:
   - **Error**: node reads key no ancestor produces and not in `required_inputs`
   - **Warning**: node produces key nothing downstream reads

#### Input key auto-inference

When `input_keys` is empty, the graph builder infers them automatically:

```python
def infer_input_keys(config: AgentNodeConfig) -> list[str]:
    keys = set(config.input_keys)  # User-declared (may be empty)
    if config.user_prompt_template:
        keys |= SafeTemplateRenderer().extract_variables(config.user_prompt_template)
    if config.subtype and not config.input_keys:
        defaults = AGENT_SUBTYPE_DEFAULTS.get(config.subtype)
        if defaults:
            keys |= set(defaults.input_keys)
    keys -= set(config.pool_tools)  # Pools are accessed via tools, not input_keys
    keys.discard("query")  # Always available
    return sorted(keys)
```

This eliminates manual `input_keys` declaration for most cases — the graph
builder extracts variable references from templates and subtype defaults.

**Implementation file**: `src/deep_research/agent/workflow/data_flow.py`

### 5.11 Tool Call Deduplication

To prevent duplicate work from tool retries and repeated loop iterations, each
workflow execution maintains a `ToolCallCache`:

```python
@dataclass
class ToolCallCache:
    _cache: dict[str, "ToolResult"] = field(default_factory=dict)

    def _make_key(self, tool_name: str, arguments: dict) -> str:
        import hashlib, json
        content = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, tool_name: str, arguments: dict) -> "ToolResult | None":
        return self._cache.get(self._make_key(tool_name, arguments))

    def put(self, tool_name: str, arguments: dict, result: "ToolResult") -> None:
        self._cache[self._make_key(tool_name, arguments)] = result
```

- Added to `WorkflowState.tool_call_cache`
- Checked in `_execute_tool()` before execution; on hit, returns cached result
- Cache NOT cleared between node retries → automatic idempotency
- New event: `ToolCacheHitEvent(tool_name, arguments_hash)`

### 5.12 Memory Management — Conversation Compaction

With pools as tool-accessible stores (not prompt context), the memory pressure
shifts from pools to the **agent's conversation history**. ReAct loops
accumulate tool results that grow the conversation beyond context limits.

**Key insight**: Pools are never compacted — they're external stores accessed
via search tools. Only the agent's conversation (the messages list) needs
management  .

#### SOTA: Observation Masking > Summarization

JetBrains Research (Dec 2025) compared three approaches:
1. **Raw** (baseline): unbounded context growth
2. **Observation masking**: replace old tool results with `[MASKED]`
3. **LLM summarization**: summarize older messages

**Finding: Observation masking won.** 2.6% higher solve rates, 52% cheaper
than summarization. LLM summarization added >7% cost and caused agents to
run 13-15% longer (trajectory elongation).

Manus uses a similar dual strategy: first compaction (replace old tool results
with references), then summarization only when compaction yields diminishing
returns.

#### Two-phase conversation compaction

```python
async def compact_conversation(
    messages: list[Message],
    max_context_tokens: int,
    keep_recent_turns: int = 5,
) -> list[Message]:
    """Two-phase conversation compaction."""

    current_tokens = estimate_tokens(messages)
    if current_tokens < max_context_tokens * 0.7:
        return messages  # No compaction needed

    # Phase 1: Observation masking (lossless)
    # Replace old tool results with references — data still accessible via pool tools
    masked = []
    for i, msg in enumerate(messages):
        is_recent = i >= len(messages) - keep_recent_turns * 2
        if not is_recent and msg.role == "tool":
            masked.append(Message(
                role="tool",
                content="[Result available via tool. Re-query if needed.]",
                tool_call_id=msg.tool_call_id,
            ))
        else:
            masked.append(msg)

    if estimate_tokens(masked) < max_context_tokens * 0.7:
        return masked

    # Phase 2: Summarization (only if masking insufficient)
    older = masked[:-keep_recent_turns * 2]
    recent = masked[-keep_recent_turns * 2:]
    summary = await summarize_trajectory(older)
    return [
        Message(role="system", content=f"[Prior research summary]\n{summary}"),
        *recent,
    ]
```

#### Configuration

Configured per-agent via `AgentNodeConfig` (§4.1):
- `context_budget_tokens: int = 0` — Max conversation tokens (0 = model's context window)
- `compact_keep_recent_turns: int = 5` — Turns to keep in full during compaction

#### Trigger

Compaction is checked BEFORE each LLM call in a ReAct loop when the agent
has `context_budget_tokens > 0`. Each agent compacts independently — blast
radius is limited to the current agent's conversation.

#### Why this is better than pool compaction

| Aspect | Pool compaction (removed) | Conversation compaction |
|--------|--------------------------|------------------------|
| What's compacted | Pool items (shared across agents) | One agent's conversation (isolated) |
| Information loss | Permanent — original items replaced | Recoverable — agent can re-query pool |
| Blast radius | All downstream agents see compacted data | Only the current agent affected |
| Citation safety | Citations destroyed | Citations preserved in pool |
| Cost | LLM call per compaction trigger | Phase 1 is free; Phase 2 only if needed |
| Research backing | None | JetBrains 2025, Manus, ACON paper |

New event: `ConversationCompactedEvent(node_id, phase, tokens_before, tokens_after)`

**Implementation file**: `src/deep_research/agent/workflow/conversation_compactor.py`

---

## 6. Condition System

Conditions are used in `loop.until` and `conditional.conditions`. They are
**declarative** — no code, no expression language. Three types that compose:

### 6.1 StateCondition — check a value

```python
class StateCondition(BaseModel):
    type: Literal["state"] = "state"
    key: str                # Dot-path into state.data, or "pool:name" for pools
    operator: Literal[
        "eq", "neq",           # equality
        "gt", "lt", "gte", "lte",  # numeric
        "contains",            # substring or list membership
        "not_contains",
        "empty", "not_empty",  # None or empty collection
        "length_gt", "length_lt",  # len() comparison
    ]
    value: Any = None       # Right-hand side (not needed for empty/not_empty)
```

Evaluation:
```python
def eval_state_condition(c: StateCondition, state: WorkflowState) -> bool:
    actual = resolve_key(state, c.key)  # Supports "pool:observations" syntax
    match c.operator:
        case "eq":          return actual == c.value
        case "neq":         return actual != c.value
        case "gt":          return actual is not None and actual > c.value
        case "lt":          return actual is not None and actual < c.value
        case "gte":         return actual is not None and actual >= c.value
        case "lte":         return actual is not None and actual <= c.value
        case "contains":    return c.value in actual if actual else False
        case "not_contains": return c.value not in actual if actual else True
        case "empty":       return actual is None or actual == [] or actual == "" or actual == {}
        case "not_empty":   return actual is not None and actual != [] and actual != "" and actual != {}
        case "length_gt":   return hasattr(actual, '__len__') and len(actual) > c.value
        case "length_lt":   return hasattr(actual, '__len__') and len(actual) < c.value
```

The `"pool:"` prefix in `key` enables conditions on pool contents:
```yaml
# Exit loop when observations pool has 5+ items:
until:
  key: "pool:observations"
  operator: length_gt
  value: 4

# Exit loop when sources pool is non-empty:
until:
  key: "pool:sources"
  operator: not_empty
```

### 6.2 LLMCondition — ask a model

```python
class LLMCondition(BaseModel):
    type: Literal["llm"] = "llm"
    prompt: str             # Safe template (§7.5): "Is research on {{query}} complete?"
    model_tier: str = "simple"
    true_value: str = "yes" # If LLM response contains this string -> True
    input_keys: list[str] = []  # State keys to inject into prompt
```

The executor renders the prompt template with state values, calls the LLM, and
checks if the response contains `true_value` (case-insensitive).

Use sparingly — each evaluation costs an LLM call. Best for complex judgments
that can't be reduced to a state check.

### 6.3 CompositeCondition — boolean logic

```python
class CompositeCondition(BaseModel):
    type: Literal["all", "any", "not"]
    conditions: list[Condition]  # Condition = StateCondition | LLMCondition | CompositeCondition
    # "all" = AND: true if ALL children are true
    # "any" = OR:  true if ANY child is true
    # "not" = NOT: true if first child is false (only uses conditions[0])
```

Example — exit loop when reflector says "complete" AND at least 3 findings:
```yaml
until:
  type: all
  conditions:
    - {type: state, key: reflection_decision, operator: eq, value: complete}
    - {type: state, key: findings, operator: length_gt, value: 2}
```

Example — exit loop when observations pool has enough items:
```yaml
until:
  type: all
  conditions:
    - {type: state, key: reflection_decision, operator: eq, value: complete}
    - {type: state, key: "pool:observations", operator: length_gt, value: 4}
```

### 6.4 When NOT to use LLMCondition

A common pitfall: using `LLMCondition` where a `StateCondition` suffices.

Bad:
```yaml
until:
  type: llm
  prompt: "Is the reflection decision 'complete'?"
```

Good:
```yaml
until:
  type: state
  key: reflection_decision
  operator: eq
  value: complete
```

`LLMCondition` is for cases where the judgment is GENUINELY complex — e.g.,
"given all the findings, is the evidence sufficient to answer the query?" There,
no single state key captures the answer.

### 6.5 How the reflector bridges agent output -> condition

A key design question: the reflector is an `agent` node that writes text. How
does the `loop.until` condition reliably parse that text?

**Answer: use `output_format: json` with `output_schema`.**

```yaml
- id: reflector
  type: agent
  config:
    role: "Reflector"
    model_tier: complex
    input_keys: [query, plan, findings]
    pool_tools: [observations, sources]
    output_key: reflection
    output_format: json
    output_schema:
      type: object
      properties:
        decision:
          type: string
          enum: [continue, adjust, complete]
        reasoning:
          type: string
      required: [decision, reasoning]
```

Now `state.get("reflection")` is `{"decision": "complete", "reasoning": "..."}`,
and the loop condition can reliably check:

```yaml
until:
  key: "reflection.decision"
  operator: eq
  value: "complete"
```

The `enum` constraint in the JSON schema ensures the LLM outputs exactly one of
the valid values. No regex parsing, no fragile string matching.

---

## 7. Agent Prompt Construction

When the executor runs an `agent` node, it needs to build LLM messages. This
section specifies exactly how.

### 7.1 System message

Priority order (first non-None wins):
1. `config.system_prompt` (inline)
2. `config.system_prompt_template` (resolved from template library)
3. Default system prompt for the role/subtype (built-in)

The system prompt is static — it doesn't reference state.

### 7.2 User message

**If `user_prompt_template` is provided**, render it using the
`SafeTemplateRenderer` (§7.5) with state values:

```python
from deep_research.services.template_renderer import SafeTemplateRenderer

_renderer = SafeTemplateRenderer()

def render_user_prompt(
    template: str,
    state: WorkflowState,
    input_keys: list[str],
) -> str:
    context = {key: resolve_key(state, key) for key in input_keys}
    context["query"] = state.query
    # NOTE: Pool contents are NOT injected into prompt context.
    # Pools are accessed via auto-generated tools (pool_tools config).
    return _renderer.render(template, context)
```

Example:
```yaml
user_prompt_template: |
  Research question: {{query}}

  Plan: {{plan}}

  Findings so far: {{findings}}

  Use search_observations and search_sources tools to retrieve relevant
  accumulated findings and sources. Execute the next research step.
```

**If `user_prompt_template` is NOT provided**, auto-construct from `input_keys`:

```python
def auto_construct_user_prompt(
    state: WorkflowState,
    input_keys: list[str],
    pool_tools: list[str],
) -> str:
    parts = [f"Query: {state.query}"]

    # Explicit state reads
    for key in input_keys:
        value = resolve_key(state, key)
        if value is not None:
            if isinstance(value, list):
                formatted = "\n".join(f"- {item}" for item in value)
                parts.append(f"\n{key.replace('_', ' ').title()}:\n{formatted}")
            elif isinstance(value, dict):
                parts.append(f"\n{key.replace('_', ' ').title()}:\n{json.dumps(value, indent=2)}")
            else:
                parts.append(f"\n{key.replace('_', ' ').title()}: {value}")

    # Pool tools guidance (instead of injecting pool contents)
    if pool_tools:
        tool_names = ", ".join(f"search_{p}" for p in pool_tools)
        parts.append(
            f"\nUse the following tools to retrieve relevant accumulated data: {tool_names}."
            f" Also available: get_recent_*, count_*, topics_* variants."
        )

    return "\n".join(parts)
```

Auto-construction covers 80% of cases. Templates give full control for the 20%.

### 7.3 Tool definitions

If `config.tools` is non-empty, the executor resolves each `ToolRef` into a
`ToolDefinition` (JSON schema) and includes them in the LLM request. The LLM
then operates in ReAct mode — reasoning + tool calls — until it responds
without calling any tools (signal to stop) or hits `max_tool_calls`.

### 7.4 Output handling

After the LLM responds:

- **`output_format: "text"`** (default): Store the text response as-is in
  the state log via `state.append(node_id, output_key, value)`.
- **`output_format: "json"`**: Parse the response as JSON. Append the parsed
  dict to the state log. This enables dot-path access in conditions.

When `output_schema` is provided and `output_format: "json"`, the executor
validates the parsed dict against the schema using JSON Schema validation.
On failure, `ParseFailureConfig` logic applies (retry with corrective prompt
including schema violation details, stop, or continue).

**JSON parse failure handling** is controlled by `ParseFailureConfig`:

```python
class ParseFailureConfig(BaseModel):
    on_parse_failure: Literal["stop", "retry", "continue"] = "stop"
    max_parse_retries: int = 2
    retry_prompt: str = "Your previous response was not valid JSON. Please respond with valid JSON only."
```

Parse failure behavior:
1. If JSON parsing fails and `on_parse_failure="stop"` (default) → retry up to
   `max_parse_retries` times, appending `retry_prompt` to the conversation.
2. If retries exhausted → raise `ParseFailureStopError`. In a loop context,
   this exits the loop cleanly with `LoopExitEvent(reason="parse_failure")`.
3. If `on_parse_failure="continue"` → store raw text (opt-in only, for backward
   compatibility with workflows that tolerate malformed output).
4. If `on_parse_failure="retry"` → same as "stop" but does NOT raise after
   exhausting retries; falls through to storing raw text.

**Critical safety invariant**: If the LLM fails to generate a response that
says "continue" (in a loop control context), the loop MUST stop. The default
`on_parse_failure="stop"` enforces this.

After writing to `output_key`, the executor processes `pool_writes` (§5.4).
If `verification` is configured, run the verification pipeline on the output
before writing to state (§11).

### 7.5 Template Security

**CRITICAL**: User-provided templates MUST NOT use Jinja2. The codebase uses a
`SafeTemplateRenderer` — a restricted template language that prevents
Server-Side Template Injection (SSTI → RCE).

#### Why not Jinja2

Jinja2's `Template(user_string).render(**context)` allows arbitrary Python
execution via attribute traversal (`{{''.__class__.__mro__[1].__subclasses__()}}`).
Even with `SandboxedEnvironment`, escapes are regularly discovered. The attack
surface is too large for user-provided templates.

#### Supported syntax

| Syntax | Purpose | Example |
|--------|---------|---------|
| `{{variable}}` | Variable substitution | `{{query}}` |
| `{{variable\|length}}` | Length filter (only filter) | `{{sources\|length}}` |
| `{{#if variable}}...{{/if}}` | Conditional block | `{{#if findings}}...{{/if}}` |
| `{{#for item in variable}}...{{/for}}` | Iteration | `{{#for obs in observations}}...{{/for}}` |
| `{{loop.index}}` | Loop counter (1-based) | Inside `{{#for}}` blocks |
| `{{item.key}}` | Dict key access in loops | `{{item.url}}` → `item["url"]` |

#### Security invariants

- Variable names MUST match `[a-zA-Z_][a-zA-Z0-9_]*` — no dots outside loops, no brackets
- Dict access ONLY inside for-loops, ONLY single-level (`item["key"]`), NOT attribute traversal
- No expression evaluation, no method calls, no imports, no filters beyond `|length`
- Context values stringified via `str()` — no object introspection
- Max nesting depth: 3. Max loop iterations: 1000.
- Templates validated on save (API rejects forbidden patterns before persisting)

#### Forbidden patterns (rejected on validation)

- `__dunder__` access (e.g., `__class__`, `__mro__`)
- Method calls (e.g., `.format(`, `.join(`)
- `import`, `eval`, `exec`, `getattr`, `setattr`, `globals`, `locals`
- Bracket string access (e.g., `["key"]`)
- Jinja2 block syntax (`{% %}`)

**Implementation file**: `src/deep_research/services/template_renderer.py`

### 7.6 Typed Output Models

Each builtin subtype defines a **typed Pydantic output model** as the contract
for `AgentOutput.content` when `output_format: json`. These models serve as
both runtime validation contracts and documentation of what each subtype
produces.

| Subtype | Output Model | Key Fields |
|---------|-------------|------------|
| planner | `PlanOutput` | title, thought, steps, has_enough_context, iteration |
| reflector | `ReflectionOutput` | decision (continue/adjust/complete), reasoning, suggested_changes |
| coordinator | `CoordinatorOutput` | complexity, is_simple, recommended_depth, direct_response |
| researcher | `ResearcherOutput` | findings, sources_found |
| synthesizer | `SynthesizerOutput` | report, structured_output |
| background | `BackgroundOutput` | data_landscape, summary, query_decomposition |

The agent harness emits a generic `AgentOutputEvent` for all agents. Builtin
subtypes **additionally** emit their domain-specific events (see §9.5) — the
harness calls the subtype's `emit_domain_events()` method after output parsing.
This makes the event mapper trivial: forward domain events directly instead of
parsing raw dict output.

See `contracts/events.py` for the full model and event definitions.

---

## 8. Tool System

### 8.1 ToolRef — referencing a tool

```python
class ToolRef(BaseModel):
    """Reference to a tool resolved at runtime."""

    type: Literal["builtin", "uc_function", "uc_tool", "enterprise"]
    name: str
    description: str | None = None     # Override default description
    config: dict[str, Any] | None = None  # Extra configuration
```

| `type`         | `name` format                        | Example                                  |
|----------------|--------------------------------------|------------------------------------------|
| `builtin`      | Predefined name                      | `web_search`, `web_crawl`, `file_search`, `majority_vote`, `verify_and_ground` |
| `uc_function`  | `catalog.schema.function`            | `main.analytics.search_docs`             |
| `uc_tool`      | `catalog.schema.tool`                | `main.tools.code_interpreter`            |
| `enterprise`   | Data source name                     | `sales_genie`, `product_docs_vs`         |

### 8.2 Tool resolution

The executor resolves `ToolRef` objects into `ResearchTool` instances at
workflow startup (not per-call), caching them for the workflow's lifetime:

```python
async def resolve_tool_ref(ref: ToolRef, context: ExecutionContext) -> ResearchTool:
    match ref.type:
        case "builtin":
            return BUILTIN_TOOLS[ref.name]
        case "uc_function":
            return UCFunctionTool(
                function_name=ref.name,
                client=context.workspace_client,
                description=ref.description,
            )
        case "uc_tool":
            return UCToolTool(
                tool_name=ref.name,
                client=context.workspace_client,
            )
        case "enterprise":
            tool = find_by_name(context.enterprise_tools, ref.name)
            if tool is None:
                raise ToolResolutionError(f"Enterprise source '{ref.name}' not found")
            return tool
```

### 8.3 UC Function integration

A UC function becomes a `ResearchTool` by fetching its metadata from Unity
Catalog and translating to JSON Schema for the LLM:

```python
class UCFunctionTool:
    """Wraps a Unity Catalog function as a ResearchTool."""

    def __init__(self, function_name: str, client: WorkspaceClient,
                 description: str | None = None):
        self._name = function_name
        self._client = client
        func_info = client.functions.get(function_name)
        self._definition = ToolDefinition(
            name=function_name.replace(".", "_"),
            description=description or func_info.comment or f"Call {function_name}",
            parameters=uc_params_to_json_schema(func_info.input_params),
            source_type="uc_function",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any],
                      context: ResearchContext) -> ToolResult:
        try:
            result = self._client.functions.execute(
                self._name, input_params=arguments
            )
            return ToolResult(content=str(result.value), success=True)
        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        return []  # UC validates on execution
```

UC tools follow the same pattern but may have different execution semantics
(e.g., multi-turn conversations).

### 8.4 Tools on agent nodes vs standalone tool nodes

**Agent tool** — LLM decides when to call:
```yaml
- type: agent
  config:
    role: "Analyst"
    tools:
      - {type: uc_function, name: "main.finance.get_revenue"}
      - {type: builtin, name: web_search}
```

**Standalone tool node** — always executes at this point:
```yaml
- type: tool
  config:
    ref: {type: uc_function, name: "main.finance.get_revenue"}
    input_mapping: {company: "company_name"}  # state key -> param
    output_key: revenue_data
```

Use standalone tool nodes for deterministic steps: "always fetch this data
before analysis." Use agent tools when the LLM should decide whether and how
to use the tool.

### 8.5 Built-in functions

The `builtin` type includes external tools, transform functions, and the
verification pipeline:

| Name                | Kind       | Purpose                                |
|---------------------|------------|----------------------------------------|
| `web_search`        | External   | Brave API search                       |
| `web_crawl`         | External   | Fetch and extract web page content     |
| `file_search`       | External   | Search user-uploaded files             |
| `majority_vote`     | Transform  | Count most common value from inputs    |
| `concatenate`       | Transform  | Join multiple inputs into one string   |
| `pick_longest`      | Transform  | Select the longest input               |
| `json_merge`        | Transform  | Deep-merge JSON objects                |
| `verify_and_ground` | Pipeline   | Run citation verification pipeline (see §11) |
| `merge_sections`    | Transform  | Merge verified sections into unified report |
| `preselect_evidence`| Pipeline   | Stage 1 evidence curation (raw sources → curated evidence pool) |
| `search_{pool}`     | Pool       | BM25+vector hybrid search over pool items (auto-generated) |
| `get_recent_{pool}` | Pool       | Return N most recent pool items (auto-generated) |
| `count_{pool}`      | Pool       | Return pool item count (auto-generated) |
| `topics_{pool}`     | Pool       | Structured topic overview of pool contents (auto-generated) |
| `get_{pool}_by_index` | Pool     | Get single item by index (auto-generated) |

Pool tools are auto-generated for each pool name declared in an agent's
`pool_tools` config. Implementation reuses `HybridSearchIndex` from
`agent/tools/evidence_registry.py` (no new search infrastructure needed).

Transform-type builtins are used as standalone `tool` nodes (not as agent tools)
for aggregating parallel outputs:

```yaml
- type: tool
  config:
    ref: {type: builtin, name: majority_vote}
    input_mapping:
      candidate_a: "candidate_a"
      candidate_b: "candidate_b"
      candidate_c: "candidate_c"
    output_key: voted_result
```

The `verify_and_ground` builtin runs the existing 7-stage citation pipeline
as a standalone tool node (see §11 for details).

#### Builtin tool output schemas

Each builtin tool registers its output schema in `BUILTIN_TOOL_SCHEMAS`. This
enables DataFlowGraph to validate dot-path references into tool outputs without
requiring users to redeclare schemas.

```python
BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "verify_and_ground": {
        "type": "object",
        "properties": {
            "grounded_text": {"type": "string"},
            "claims": {"type": "array", "items": {"type": "object"}},
            "verification_summary": {"type": "object"},
        },
        "required": ["grounded_text", "verification_summary"],
    },
    "merge_sections": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "claims": {"type": "array", "items": {"type": "object"}},
            "verification_summary": {"type": "object"},
        },
        "required": ["text", "verification_summary"],
    },
    "preselect_evidence": {
        "type": "object",
        "properties": {
            "evidence_count": {"type": "integer"},
            "ready": {"type": "boolean"},
        },
        "required": ["evidence_count", "ready"],
    },
}
```

When a `ToolNodeConfig` has `output_schema: None` and `ref` is a builtin, the
executor falls back to `BUILTIN_TOOL_SCHEMAS[ref.name]` for DataFlowGraph
analysis and runtime validation.

### 8.6 URL Registry

The `UrlRegistry` is a first-class field on `ToolContext` that maps integer
indices to URLs. The LLM never sees raw URLs — only integer indices. This is
a security pattern ported from the existing app's `tools/url_registry.py`.

**Lifecycle**: Created once per workflow execution, shared across all tool
calls within that execution. The executor creates the `UrlRegistry` instance
and passes it via `ToolContext` to every tool call.

**Usage pattern**:
1. `web_search` tool registers discovered URLs → returns indices
2. LLM sees search results with `[1]`, `[2]` indices instead of full URLs
3. `web_crawl` tool receives index as argument → resolves to URL via registry
4. Final synthesis references sources by index

**Why not raw URLs?** Raw URLs in LLM context enable prompt injection via
crafted URLs, leak information about internal search infrastructure, and
consume unnecessary context window tokens.

See `contracts/tool_protocol.py` for the `UrlRegistry` class definition.

### 8.7 Constructor Dependency Injection

Tools receive their dependencies at construction time, not per-call via
`ToolContext`. Only per-call values (current query, URL registry) belong
in `ToolContext`.

```python
# Framework builtin — deps injected at construction
class WebSearchTool:
    def __init__(self, search_client: BraveSearchClient, *, domain_filter: str | None = None) -> None:
        self._client = search_client
        self._domain_filter = domain_filter

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Uses self._client and self._domain_filter (construction time)
        # Uses context.query and context.url_registry (per-call)
        ...

# App enterprise tool — all context at construction
class GenieTool:
    def __init__(self, space_id: str, ws_client: WorkspaceClient, user_token: str | None = None) -> None:
        self._space_id = space_id
        self._ws = ws_client
        self._user_token = user_token
```

**App tool factory** creates tools once per workflow execution:
```python
async def create_framework_tools(config: OrchestrationConfig, ...) -> list[ResearchTool]:
    tools = []
    if brave_client:
        tools.append(WebSearchTool(search_client=brave_client, domain_filter=config.domain_filter))
    if crawler:
        tools.append(WebCrawlTool(crawler=crawler))
    for app_tool in enterprise_tools:
        tools.append(EnterpriseToolAdapter(app_tool=app_tool, user_token=config.user_token))
    return tools
```

**Rationale**: Constructor DI keeps `ToolContext` minimal (2 fields instead of 5),
eliminates the `extra: dict[str, Any]` escape hatch that defeats typed protocols,
and makes tool dependencies explicit and testable.

---

## 9. Standard Agent Subtypes

A `subtype` field on `AgentNodeConfig` resolves to sensible defaults for common
agent roles. This eliminates boilerplate — a researcher, synthesizer, or
verifier can be configured in one line. User-provided values ALWAYS override
subtype defaults.

### 9.1 The 6 built-in subtypes

| Subtype | role | model_tier | output_key | tools | pool_writes | pool_tools | output_format | output_model |
|---------|------|------------|-----------|-------|-------------|------------|---------------|--------------|
| `coordinator` | Coordinator | simple | coordination | — | — | — | json | `CoordinatorOutput` |
| `researcher` | Researcher | analytical | findings | web_search, web_crawl | [{pool: observations}] | [observations] | text | `ResearcherOutput` |
| `planner` | Planner | analytical | plan | — | [observations] | — | json | `PlanOutput` |
| `reflector` | Reflector | simple | reflection | — | — | [observations] | json | `ReflectionOutput` |
| `synthesizer` | Synthesizer | complex | report | — | [claims] | [observations, sources] | text | `SynthesizerOutput` |
| `background` | Background | simple | background | [web_search] | [sources] | — | json | `BackgroundOutput` |

### 9.2 Merging logic

```python
AGENT_SUBTYPE_DEFAULTS: dict[str, AgentNodeConfig] = {
    "coordinator": AgentNodeConfig(
        role="Coordinator",
        model_tier="simple",
        output_key="coordination",
        output_format="json",
        output_model=CoordinatorOutput,
    ),
    "researcher": AgentNodeConfig(
        role="Researcher",
        model_tier="analytical",
        output_key="findings",
        tools=[
            ToolRef(type="builtin", name="web_search"),
            ToolRef(type="builtin", name="web_crawl"),
        ],
        pool_writes=[PoolWriteConfig(pool="observations")],
        pool_tools=["observations"],
        output_format="text",
        output_model=ResearcherOutput,
    ),
    "planner": AgentNodeConfig(
        role="Planner",
        model_tier="analytical",
        output_key="plan",
        pool_writes=[PoolWriteConfig(pool="observations")],
        output_format="json",
        output_model=PlanOutput,
    ),
    # Reflector uses simple tier — the decision is straightforward
    # (CONTINUE/ADJUST/COMPLETE) and doesn't require complex reasoning.
    # Pool injection used for small observation pools (see §5.3).
    "reflector": AgentNodeConfig(
        role="Reflector",
        model_tier="simple",
        output_key="reflection",
        pool_tools=["observations"],
        output_format="json",
        output_model=ReflectionOutput,
    ),
    "synthesizer": AgentNodeConfig(
        role="Synthesizer",
        model_tier="complex",
        output_key="report",
        pool_writes=[PoolWriteConfig(pool="claims")],
        pool_tools=["observations", "sources"],
        output_format="text",
        output_model=SynthesizerOutput,
    ),
    "background": AgentNodeConfig(
        role="Background",
        model_tier="simple",
        output_key="background",
        tools=[
            ToolRef(type="builtin", name="web_search"),
        ],
        pool_writes=[PoolWriteConfig(pool="sources")],
        output_format="json",
        output_model=BackgroundOutput,
    ),
}


def resolve_agent_config(raw_config: dict) -> AgentNodeConfig:
    subtype = raw_config.get("subtype")
    if not subtype:
        return AgentNodeConfig(**raw_config)
    defaults = AGENT_SUBTYPE_DEFAULTS[subtype].model_dump(exclude_none=True)
    for key, value in raw_config.items():
        if key != "subtype":
            defaults[key] = value  # User overrides
    return AgentNodeConfig(**defaults)
```

Merging logic for `output_schema` is unchanged from other fields —
user-provided `output_schema` overrides the subtype default. This means a user
can narrow (or widen) the schema for a subtype without losing other defaults:

```yaml
- type: agent
  config:
    subtype: reflector
    output_schema:
      type: object
      properties:
        decision: {type: string, enum: [CONTINUE, COMPLETE]}
        reasoning: {type: string}
      required: [decision, reasoning]
```

### 9.3 Implementation delegation

When a `subtype` has a registered implementation (e.g., `researcher` ->
`run_react_researcher`), the executor delegates to it. This preserves existing
sophistication (token budgets, quality signals, source tracking).

```python
SUBTYPE_IMPLEMENTATIONS: dict[str, Callable] = {
    "researcher": run_react_researcher,   # or run_researcher based on config
    "synthesizer": run_citation_synthesizer,
    "reflector": run_reflector,
    "planner": run_planner,
}
```

When the executor encounters an agent node with a recognized `subtype` AND a
registered implementation, it delegates to that function instead of the generic
LLM call path. For user-defined roles or subtypes without a registered
implementation, the executor uses the generic agent execution path: build
prompt from config, resolve pool tools, call LLM with ReAct loop, parse output.

### 9.4 YAML examples

**Minimal** — one line to get a fully-configured researcher:
```yaml
- type: agent
  config: {subtype: researcher}
```

**Override specific fields** — keep defaults, change what you need:
```yaml
- type: agent
  config:
    subtype: researcher
    model_tier: complex
    tools: [{type: enterprise, name: sales_genie}]
    pool_writes: [{pool: enterprise_findings}]
```

**Synthesizer with verification** — activate citation pipeline on output:
```yaml
- type: agent
  config:
    subtype: synthesizer
    verification:
      enabled: true
      stages: {verification_retrieval: true}
```

**Complete research pipeline using subtypes:**
```yaml
root:
  type: sequence
  children:
    - type: agent
      config: {subtype: planner}
    - type: loop
      config:
        until: {key: "reflection.decision", operator: eq, value: complete}
        max_iterations: 5
      children:
        - type: sequence
          children:
            - type: agent
              config: {subtype: researcher}
            - type: agent
              config: {subtype: reflector}
    - type: agent
      config:
        subtype: synthesizer
        verification: {enabled: true}
```

### 9.4.6 Background Investigator

The `background` subtype is a specialized agent for quick initial context
gathering before planning. It differs fundamentally from a researcher:

**Specialized behavior**:
- **Query decomposition**: Generates 2-3 focused sub-queries from the user's
  original query for parallel search
- **Data landscape assessment**: Probes enterprise sources in parallel with
  a hard 5-second timeout, producing a structured data landscape map
- **Quick context gathering**: Uses SIMPLE model tier (not ANALYTICAL) for speed
- **Small search budget**: Minimal web searches, focused on discovering what
  sources exist rather than deep research

**Output**: `BackgroundOutput` with `data_landscape` (dict mapping source types
to availability), `summary` (brief context overview), and `query_decomposition`
(list of sub-queries). This output feeds source-aware planning — the planner
reads `background.data_landscape` to decide which sources to prioritize.

**Domain event**: Emits `BackgroundCompletedEvent(sources_discovered, data_landscape_summary)`.

### 9.5 Domain Events from Builtin Subtypes

Each builtin subtype emits **domain-specific events** in addition to the
generic `AgentOutputEvent`. These events make the event mapper trivial — it
forwards domain events directly instead of parsing raw dict output.

| Subtype | Domain Events | When Emitted |
|---------|--------------|--------------|
| coordinator | `CoordinatorClassifiedEvent` | After query classification |
| planner | `PlanCreatedEvent` | After plan generation or replan |
| researcher | *(none — item events emitted by plan_and_execute node)* | — |
| reflector | `ReflectionDecisionEvent` | After CONTINUE/ADJUST/COMPLETE decision |
| synthesizer | `SynthesisStartedEvent` | At synthesis begin |
| background | `BackgroundCompletedEvent` | After background investigation |

The `plan_and_execute` node additionally emits `ItemsExtractedEvent`,
`ItemStartedEvent`, `ItemCompletedEvent`, `EvaluationDecisionEvent`,
`ReplanTriggeredEvent`, and `PlanAndExecuteExitEvent` during its cycle.

**Event flow**: The agent harness calls `builtin.emit_domain_events(output)`
after parsing the agent's output. Domain events are yielded to the executor's
stream alongside the generic `AgentOutputEvent`. The app's domain context
tracker (§15.3) receives these events and translates them directly to SSE
events — no dict parsing required.

---

## 10. Subworkflows

### 10.1 What a subworkflow is

A `subworkflow` node delegates execution to a separately defined
`WorkflowDefinition`. The child workflow runs with **isolated `state.data`** —
it cannot directly read or write the parent's data keys. Communication happens
only through explicit `input_mapping` and `output_mapping`.

This is how **sub-agents stay opaque**. The parent sees a single node with
defined inputs/outputs. It doesn't know (or care) about the internal structure.

### 10.2 State isolation — data is ISOLATED, pools are CONFIGURABLE

```
Parent state:  { log: [..., (planner, query, "..."), (planner, company, "ACME")] }
                         | input_mapping: {query: "query", company: "target"}
Child state:   { log: [(init, query, "..."), (init, target, "ACME")] }
                                    | (child executes its tree)
Child state:   { log: [..., (analyst, analysis, "...")] }
                         | output_mapping: {analysis: "financial_analysis"}
Parent state:  { log: [..., (subworkflow, financial_analysis, "...")] }
```

**What IS always shared:**
- `enterprise_tools` — child inherits the parent's resolved tools.
- `is_cancelled` — cancellation propagates down.

**What is NEVER shared:**
- **State log** — fully isolated. Only mapped keys cross the boundary.
- `iteration_counts` — scoped to each workflow.

**Pools — configurable via `pool_mode`:**

| `pool_mode` | Behavior | Use case |
|-------------|----------|----------|
| `"shared"` (default) | Child sees and writes to parent pools by reference. Sub-agent findings accumulate in the parent's pools automatically. | Inner research loops where a parent synthesizer needs raw data. |
| `"isolated"` | Child creates its OWN pools from its `WorkflowDefinition.pools`. Parent pools are NOT accessible to child. Only `output_mapping` keys cross the boundary. | Self-contained research units, BestOfN candidates, parallel section research. |

**`pool_mode: "isolated"` enables the Anthropic pattern**: Each subworkflow
is a self-contained research unit that researches, synthesizes, and verifies
internally, returning only verified output. Internal pools are discarded after
completion.

**Example: Section-based parallel research with isolated subworkflows:**

```yaml
- type: parallel
  children:
    - type: subworkflow
      label: "Research: mRNA Mechanism"
      config:
        ref: "section_research_and_verify"
        pool_mode: isolated                          # OWN pools
        input_mapping: {query: "query", section_topic: "plan.sections[0]"}
        output_mapping: {verified_section: "section_1"}

    - type: subworkflow
      label: "Research: Clinical Efficacy"
      config:
        ref: "section_research_and_verify"
        pool_mode: isolated                          # OWN pools
        input_mapping: {query: "query", section_topic: "plan.sections[1]"}
        output_mapping: {verified_section: "section_2"}
```

Each parallel subworkflow has its own observations, sources, and evidence
pools. They cannot see each other's work. Only the `verified_section` output
crosses back to the parent.

```python
async def _exec_subworkflow(self, node, state):
    config = SubworkflowNodeConfig(**node.config)

    # Resolve workflow (may be builtin template, see §12)
    child_workflow = await self._resolve_workflow(config, state)

    # Create isolated child state
    child_state = WorkflowState(query=state.query)
    for parent_key, child_key in config.input_mapping.items():
        child_state.append("_init", child_key, resolve_key(state, parent_key))

    # Pool isolation based on pool_mode
    if config.pool_mode == "shared":
        child_state.pools = state.pools       # Shared by reference (default)
    elif config.pool_mode == "isolated":
        child_state.pools = {}                # Fresh pools from child definition
        # _init_pools() will create them from child_workflow.pools

    # Share context (always)
    child_state.enterprise_tools = state.enterprise_tools
    child_state.user_token = state.user_token
    child_state.model_overrides = state.model_overrides
    child_state.is_cancelled = state.is_cancelled

    # Execute
    async for event in self.execute(child_workflow, child_state):
        yield event

    # Map outputs back (only these cross the boundary)
    if config.output_mapping:
        for child_key, parent_key in config.output_mapping.items():
            state.append(node.id, parent_key, child_state.get(child_key))
    else:
        state.append(node.id, config.output_key, child_state.get("output"))

    # Write subworkflow output to parent pools (if configured)
    if config.pool_writes:
        output_value = state.get(config.output_key)
        await write_pools(state, config.pool_writes, output_value)

    # If isolated: child pools are now unreferenced → garbage collected
```

### 10.3 Defining reusable workflows

Users save workflows in the database:

```yaml
# "critique_cycle" — reusable pattern
name: critique_cycle
required_inputs: [query, draft]
output_keys: [refined_draft]

root:
  type: loop
  config:
    until: {key: "verdict.approved", operator: eq, value: true}
    max_iterations: 3
  children:
    - type: sequence
      children:
        - type: agent
          label: "Critic"
          config:
            role: "Critic"
            model_tier: complex
            input_keys: [query, draft]
            output_key: verdict
            output_format: json
            output_schema:
              type: object
              properties:
                approved: {type: boolean}
                feedback: {type: string}
              required: [approved, feedback]
        - type: agent
          label: "Refiner"
          config:
            role: "Refiner"
            model_tier: analytical
            input_keys: [query, draft, verdict]
            output_key: draft
            output_mode: replace
```

Then reference it in a parent workflow:
```yaml
- type: subworkflow
  label: "Quality Review"
  config:
    ref: "critique_cycle"
    input_mapping: {query: "query", draft: "report"}
    output_mapping: {draft: "refined_report"}
```

### 10.4 Dynamic workflows

An agent generates a workflow tree at runtime:

```yaml
name: "Dynamic Planner"

root:
  type: sequence
  children:
    - id: planner
      type: agent
      label: "Adaptive Planner"
      config:
        role: "Workflow Planner"
        model_tier: complex
        input_keys: [query]
        output_key: generated_plan
        output_format: json
        output_schema:
          type: object
          description: "A workflow tree definition"
        system_prompt: |
          You are a research workflow planner. Given a query, output a workflow
          tree as JSON. Available node types: agent, tool, sequence, parallel,
          loop. Keep it to max 8 nodes, max depth 2.
    - id: executor
      type: subworkflow
      label: "Execute Generated Plan"
      config:
        ref: "__dynamic__"        # Special: executor interprets generated_plan as a workflow
        constraints:
          max_nodes: 12
          max_depth: 3
          allowed_node_types: [agent, sequence, parallel, loop]
          max_loop_iterations: 5
          max_parallel_children: 5
        tool_allowlist:
          - {type: builtin, name: web_search}
          - {type: builtin, name: web_crawl}
        subtypes_only: true
        max_model_tier: analytical
        execution_budget:
          total_llm_calls: 20
          total_tool_calls: 30
          timeout_seconds: 300
        input_mapping: {query: "query"}
        output_key: report
```

The `__dynamic__` ref is a special marker. The executor loads `generated_plan`
from state and validates it through a **4-layer defense** before executing
(see §10.5).

### 10.5 Dynamic Workflow Security

Dynamic workflows (`__dynamic__` ref) let an LLM generate executable workflow
trees. This is powerful but dangerous — structural constraints alone are
insufficient. Four layers of defense validate the generated tree.

#### Layer 1: Structural constraints

```python
class DynamicWorkflowConstraints(BaseModel):
    max_nodes: int = 12
    max_depth: int = 3
    allowed_node_types: frozenset[str] = frozenset({"agent", "sequence", "parallel", "loop"})
    # NO subworkflow (prevents recursive dynamic), NO conditional with LLMCondition
    max_loop_iterations: int = 5
    max_parallel_children: int = 5
```

#### Layer 2: Tool allowlist (most critical)

```python
class DynamicSubworkflowConfig(SubworkflowNodeConfig):
    constraints: DynamicWorkflowConstraints = DynamicWorkflowConstraints()
    tool_allowlist: list[ToolRef] = []       # ONLY these tools permitted
    subtypes_only: bool = True               # All agents MUST use recognized subtypes
    max_model_tier: str = "analytical"       # Cost ceiling
```

When `subtypes_only=True` (default), the LLM can only compose known agent
subtypes (researcher, synthesizer, reflector, planner, verifier) — no freeform
roles with arbitrary tool access. Any agent node in the generated tree that
uses tools not in `tool_allowlist` is rejected.

#### Layer 3: Argument policy

- `web_search`: query ≤ 500 chars, no URL patterns in query text
- `web_crawl`: URL must pass existing SSRF checks + domain allowlist
- `uc_function`: REJECTED unless explicitly in `tool_allowlist`

#### Layer 4: Runtime budgets

```python
class DynamicExecutionBudget(BaseModel):
    total_llm_calls: int = 20
    total_tool_calls: int = 30
    timeout_seconds: int = 300
```

These are hard limits enforced during execution. When any budget is exhausted,
the dynamic workflow terminates with a `BudgetExceededEvent`.

#### Validation flow

```
LLM generates JSON → Parse as WorkflowNode tree
  → Layer 1: check node count, depth, types → reject if violated
  → Layer 2: check all tool refs against allowlist → reject if unauthorized
  → Layer 2: check all agents use recognized subtypes → reject if freeform
  → Layer 2: check model tiers ≤ max_model_tier → reject if exceeded
  → Layer 3: validate tool arguments against policies → reject if dangerous
  → DataFlowGraph (§5.10): validate data dependencies → reject if broken
  → Execute with Layer 4 budgets
```

**Implementation files**:
- `src/deep_research/agent/workflow/dynamic_validator.py`
- `src/deep_research/agent/workflow/tool_policy.py`

---

## 11. Verification & Grounding

The existing codebase has a sophisticated 7-stage citation verification pipeline
(`services/citation/pipeline.py`). This section specifies how to activate it
within the workflow architecture — either per-agent, as a standalone tool node,
or as a workflow-level pattern.

### 11.1 VerificationConfig

```python
class VerificationStages(BaseModel):
    """Toggle individual stages of the 7-stage citation pipeline."""
    evidence_preselection: bool = True       # Stage 1
    interleaved_generation: bool = False     # Stage 2
    confidence_classification: bool = True   # Stage 3
    isolated_verification: bool = True       # Stage 4 (CoVe)
    citation_correction: bool = True         # Stage 5 (CiteFix)
    numeric_qa_verification: bool = True     # Stage 6
    verification_retrieval: bool = False     # Stage 7 (ARE) — expensive


class VerificationConfig(BaseModel):
    """Configuration for verification/grounding on agent output."""
    enabled: bool = True
    stages: VerificationStages = VerificationStages()
    generation_mode: Literal["classical", "natural", "strict"] = "natural"
    synthesis_mode: Literal["interleaved", "react"] = "interleaved"
    use_shared_sources: bool = True          # Read sources from pools
    write_claims_to_state: bool = True       # Write verified claims to "claims" pool
```

### 11.2 Three activation levels

**(a) Per-agent** — `verification` field on agent config. Runs the pipeline on
the agent's output BEFORE writing to state.data:

```yaml
- type: agent
  config:
    subtype: synthesizer
    verification:
      enabled: true
      stages: {verification_retrieval: true}  # Full pipeline
```

When `verification.enabled` is true, the executor:
1. Runs the agent normally, gets output text
2. Calls `_run_verification(output, state)` which bridges to the existing
   `CitationVerificationPipeline`
3. Stores the verified/grounded output in the state log via `state.append()`
4. If `write_claims_to_state`, appends verified claims to the "claims" pool

**(b) Standalone tool node** — `verify_and_ground` builtin tool:

```yaml
- type: tool
  config:
    ref: {type: builtin, name: verify_and_ground}
    input_mapping: {text: "report", query: "query"}
    output_key: verified_report
```

This runs the pipeline on an existing state value. Useful when verification
should happen as a separate step (e.g., after a subworkflow produces a draft).

**(c) Workflow-level pattern** — no new mechanism needed. Simply place
verification on the final synthesizer or as a tool node after synthesis:

```yaml
root:
  type: sequence
  children:
    - type: agent
      config: {subtype: planner}
    - type: loop
      config: {until: ..., max_iterations: 5}
      children:
        - type: sequence
          children:
            - type: agent
              config: {subtype: researcher}
            - type: agent
              config: {subtype: reflector}
    - type: agent
      id: synthesizer
      config:
        subtype: synthesizer
        verification:          # <-- activate on final output
          enabled: true
          stages:
            isolated_verification: true
            citation_correction: true
```

### 11.3 Executor integration

```python
async def _run_verification(
    self, output: str, state: WorkflowState, config: VerificationConfig,
) -> dict[str, Any]:
    """Bridge WorkflowState -> CitationVerificationPipeline.

    Returns a structured dict (not a string) so that downstream nodes can
    access sub-fields via dot-path: e.g. verified_report.grounded_text,
    verified_report.claims, verified_report.verification_summary.
    """
    # Gather sources from pools (verification reads ALL sources — no search needed)
    sources = []
    if config.use_shared_sources and "sources" in state.pools:
        sources = list(state.pools["sources"].items)  # Full list for verification

    # Build temporary ResearchState for the pipeline
    temp_state = ResearchState(
        query=state.query,
        sources=sources,
        final_report=output,
    )

    # Run the existing pipeline
    pipeline = CitationVerificationPipeline(
        llm=self.llm,
        stages=config.stages,
        generation_mode=config.generation_mode,
    )
    verified = await pipeline.run(temp_state)

    # Write claims back to pool
    if config.write_claims_to_state and "claims" in state.pools:
        await state.pools["claims"].extend_async(verified.claims)

    return {
        "grounded_text": verified.grounded_report,
        "claims": [c.to_dict() for c in verified.claims],
        "verification_summary": summary.to_dict(),
    }
```

### 11.4 Backward compatibility

| OrchestrationConfig flag | Maps to |
|---|---|
| `verify_sources` | `verification.enabled` on synthesizer node |
| `enable_post_verification` | `verification.stages.isolated_verification` (et al.) |
| `enable_citation_verification` | Presence of `verification` block on synthesizer |

### 11.5 Citation Flow in Multi-Section Workflows

The `merge_sections` builtin combines outputs from multiple parallel
verified sections into a single unified report. It:

1. **Concatenates `grounded_text`** from each section (in order), separated
   by `\n\n---\n\n`.
2. **Merges `claims`** arrays with re-indexed character offsets. Each
   section's claim offsets are shifted by the cumulative length of preceding
   sections' grounded text (+ separator length).
3. **Aggregates `verification_summary`** counts: total claims, verified
   claims, unverified claims, confidence scores averaged.

This is ordinary structured data flow — no special framework support.
Each section subworkflow returns a dict matching the `verify_and_ground`
output schema, and `merge_sections` consumes those dicts via `input_mapping`.

#### Full YAML example — parallel section research with citation flow

```yaml
name: "Multi-Section Research with Verification"
pools:
  - name: sources
    item_type: source
    dedup_key: url
  - name: observations
    item_type: text

root:
  type: sequence
  children:
    # 1. Plan the research
    - type: agent
      id: planner
      config:
        subtype: planner
        output_key: plan

    # 2. Research sections in parallel (each is an isolated subworkflow)
    - type: parallel
      children:
        - type: subworkflow
          id: section_1_workflow
          config:
            ref: section_research
            input_mapping: {query: "query", section: "plan.steps[0]"}
            output_mapping:
              verified_output: "section_1"
            output_key: section_1
            pool_mode: shared       # Sources shared across sections

        - type: subworkflow
          id: section_2_workflow
          config:
            ref: section_research
            input_mapping: {query: "query", section: "plan.steps[1]"}
            output_mapping:
              verified_output: "section_2"
            output_key: section_2
            pool_mode: shared

    # 3. Merge verified sections into unified report
    - type: tool
      id: merge
      config:
        ref: {type: builtin, name: merge_sections}
        input_mapping:
          sections: ["section_1", "section_2"]
        output_key: merged_report

    # 4. Lead synthesizer writes final report using merged grounded text
    - type: agent
      id: lead_synthesizer
      config:
        subtype: synthesizer
        input_keys: ["query", "merged_report.text", "merged_report.verification_summary"]
        output_key: report
```

The `section_research` subworkflow (saved separately):

```yaml
name: "Section Research"
required_inputs: [query, section]
output_keys: [verified_output]

root:
  type: sequence
  children:
    - type: agent
      config:
        subtype: researcher
        input_keys: [query, section]
        output_key: section_findings

    - type: agent
      config:
        subtype: synthesizer
        input_keys: [query, section, section_findings]
        output_key: section_draft

    - type: tool
      config:
        ref: {type: builtin, name: verify_and_ground}
        input_mapping: {text: "section_draft", query: "query"}
        output_key: verified_output
```

The lead synthesizer accesses `merged_report.text` and
`merged_report.verification_summary` via dot-path resolution (§5.6).
DataFlowGraph validates these dot-paths against the `merge_sections` output
schema at load time — no runtime surprises.

---

## 12. Parameterized Workflow Templates

Convenience patterns like BestOfN, SelfCritique, Debate, and MajorityVote are
**parameterized subworkflow templates** — builtin generator functions that take
parameters and return a `WorkflowDefinition` composed of the 7 primitive types.

### 12.1 Why NOT sugar types

The spec argues against type proliferation (§2). Sugar types (new NodeType
values like `BestOfNNode`, `DebateNode`) would:
1. Add 4+ new NodeType values, each with its own config model
2. Contradict the architecture's own design philosophy (§2, "Why not more types?")
3. Require 4+ new code paths in the executor
4. Mix abstraction levels — some types are primitives, others are patterns

Parameterized templates achieve the same ergonomics (+2 lines of YAML) using
the existing subworkflow mechanism, with better extensibility (new patterns
don't change the type system) and better progressive disclosure (users can
inspect the template, copy it, customize it, save as their own workflow).

### 12.2 Mechanism

Templates use the existing `subworkflow` node type with a `"builtin:"` ref
prefix and a `params` dict (see §4.5):

```yaml
- type: subworkflow
  config:
    ref: "builtin:best_of_n"
    params: {n: 3, diverse_tiers: true}
    input_mapping: {query: "query"}
    output_key: best_answer
```

When the executor encounters `ref: "builtin:..."`:
1. Look up the generator function in `BUILTIN_TEMPLATE_REGISTRY`
2. Validate `params` against the template's params schema
3. Call the generator to produce a `WorkflowDefinition`
4. Execute it as a normal subworkflow (isolated state.data, shared pools)

### 12.3 Template registry

```python
BUILTIN_TEMPLATE_REGISTRY: dict[str, tuple[type[BaseModel], Callable]] = {
    "builtin:best_of_n": (BestOfNParams, generate_best_of_n),
    "builtin:self_critique": (SelfCritiqueParams, generate_self_critique),
    "builtin:debate": (DebateParams, generate_debate),
    "builtin:majority_vote": (MajorityVoteParams, generate_majority_vote),
}
```

Each generator function takes params and returns a `WorkflowDefinition`
(a normal tree of primitives).

### 12.4 The 4 builtin templates

#### best_of_n — `sequence(parallel(N candidates), agent(judge))`

```python
class BestOfNParams(BaseModel):
    n: int = 3                              # Number of candidates (2-10)
    candidate_role: str = "Candidate"
    candidate_model_tier: str = "analytical"
    candidate_tools: list[ToolRef] = []
    diverse_tiers: bool = False             # Cycle through simple/analytical/complex
    candidate_pool_tools: list[str] = []    # Pools injected into candidate prompts
    candidate_input_keys: list[str] = []    # State keys injected into candidate prompts
    judge_role: str = "Judge"
    judge_model_tier: str = "complex"
    judge_system_prompt: str | None = None
    judge_pool_tools: list[str] = []        # Pools injected into judge prompt
    judge_input_keys: list[str] = []        # State keys injected into judge prompt
```

Generated tree:
```
sequence
├── parallel
│   ├── agent: "Candidate 1" (output_key: candidate_1)
│   ├── agent: "Candidate 2" (output_key: candidate_2)
│   └── agent: "Candidate 3" (output_key: candidate_3)
└── agent: "Judge"
    input_keys: [query, candidate_1, candidate_2, candidate_3]
    output_key: output
```

#### self_critique — `loop(sequence(generator, critic), until=approved)`

```python
class SelfCritiqueParams(BaseModel):
    generator_role: str = "Generator"
    generator_model_tier: str = "analytical"
    generator_tools: list[ToolRef] = []
    generator_pool_tools: list[str] = []          # Pools injected into generator prompts
    generator_pool_writes: list[PoolWriteConfig] = []  # Pools the generator writes to
    generator_input_keys: list[str] = []          # State keys injected into generator prompts
    critic_role: str = "Critic"
    critic_model_tier: str = "complex"
    max_iterations: int = 3
    min_iterations: int = 1
```

Generated tree:
```
loop (until: critique.approved == true, max: 3)
└── sequence
    ├── agent: "Generator"
    │   input_keys: [query, critique]
    │   output_key: draft
    └── agent: "Critic"
        input_keys: [query, draft]
        output_key: critique
        output_format: json {approved: bool, feedback: str}
```

#### debate — `sequence(loop(sequence(pro, con), rounds), agent(judge))`

```python
class DebateParams(BaseModel):
    advocate_a_role: str = "Advocate (Pro)"
    advocate_b_role: str = "Advocate (Con)"
    judge_role: str = "Judge"
    rounds: int = 3
    min_rounds: int = 2
    advocate_pool_tools: list[str] = []     # Pools injected into advocate prompts
    judge_pool_tools: list[str] = []        # Pools injected into judge prompt
```

Generated tree:
```
sequence
├── loop (max: 3, min: 2)
│   └── sequence
│       ├── agent: "Advocate (Pro)"
│       │   input_keys: [query, con_argument]
│       │   output_key: pro_argument
│       └── agent: "Advocate (Con)"
│           input_keys: [query, pro_argument]
│           output_key: con_argument
└── agent: "Judge"
    input_keys: [query, pro_argument, con_argument]
    output_key: output
```

#### majority_vote — `sequence(parallel(N voters), tool(majority_vote))`

```python
class MajorityVoteParams(BaseModel):
    n: int = 3                              # Number of voters (odd preferred)
    voter_role: str = "Voter"
    voter_model_tier: str = "analytical"
    diverse_tiers: bool = False
    voter_pool_tools: list[str] = []        # Pools injected into voter prompts
    voter_input_keys: list[str] = []        # State keys injected into voter prompts
```

Generated tree:
```
sequence
├── parallel
│   ├── agent: "Voter 1" (output_key: vote_1)
│   ├── agent: "Voter 2" (output_key: vote_2)
│   └── agent: "Voter 3" (output_key: vote_3)
└── tool: majority_vote
    input_mapping: {vote_1: vote_1, vote_2: vote_2, vote_3: vote_3}
    output_key: output
```

### 12.5 YAML usage examples

```yaml
# Best of 3 with diverse models:
- type: subworkflow
  id: select_best
  label: "Select Best Answer"
  config:
    ref: "builtin:best_of_n"
    params:
      n: 3
      candidate_role: "Research Analyst"
      diverse_tiers: true
      judge_model_tier: complex
    input_mapping: {query: "query", context: "findings"}
    output_key: best_answer

# Self-critique loop:
- type: subworkflow
  id: refine_report
  label: "Refine Report"
  config:
    ref: "builtin:self_critique"
    params:
      generator_role: "Report Writer"
      critic_role: "Quality Reviewer"
      max_iterations: 3
    input_mapping: {query: "query", draft: "report"}
    output_key: polished_report

# Debate:
- type: subworkflow
  id: policy_debate
  label: "Policy Debate"
  config:
    ref: "builtin:debate"
    params:
      advocate_a_role: "Benefits Advocate"
      advocate_b_role: "Risks Advocate"
      judge_role: "Policy Analyst"
      rounds: 2
    input_mapping: {query: "query"}
    output_key: recommendation

# Majority vote:
- type: subworkflow
  id: consensus
  label: "Consensus Vote"
  config:
    ref: "builtin:majority_vote"
    params:
      n: 5
      voter_role: "Analyst"
      diverse_tiers: true
    input_mapping: {query: "query", context: "findings"}
    output_key: consensus_answer
```

### 12.6 Extensibility

New templates can be added via three paths:

1. **System-defined**: Add to `BUILTIN_TEMPLATE_REGISTRY` — new `builtin:` ref
2. **User-defined**: Save a `WorkflowDefinition` in the DB, reference via
   regular `subworkflow.ref` (no params, just a pre-built tree)
3. **Plugin-defined**: Plugins register templates during initialization via the
   plugin API

---

## 13. Execution Engine

### 13.1 Core executor

```python
class WorkflowExecutor:
    """Walks a workflow tree, executing each node and yielding stream events."""

    def __init__(self, llm: LLMClient, brave: BraveSearchClient,
                 crawler: WebCrawler, context: ExecutionContext):
        self.llm = llm
        self.brave = brave
        self.crawler = crawler
        self.context = context
        self._resolved_tools: dict[str, ResearchTool] = {}  # Cache

    async def execute(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
    ) -> AsyncGenerator[StreamEvent, None]:
        # Initialize pools from workflow definition
        self._init_pools(workflow, state)

        # Initialize token budget if configured (§13.8)
        if workflow.token_budget > 0 and state.token_budget is None:
            state.token_budget = TokenBudget(max_total_tokens=workflow.token_budget)

        # Initialize tool call cache (§5.11)
        if state.tool_call_cache is None:
            state.tool_call_cache = ToolCallCache()

        # Resolve all tool refs once
        await self._resolve_all_tools(workflow.root)

        # Walk the tree
        async for event in self._exec(workflow.root, state):
            yield event

    async def resume_from_checkpoint(
        self,
        workflow: WorkflowDefinition,
        state: WorkflowState,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Resume execution from the last checkpoint (§13.7).

        Loads checkpoint from state, skips completed nodes, resumes at
        first non-completed node.
        """
        if not state.checkpoint or not state.checkpoint.completed_nodes:
            # No checkpoint — execute from scratch
            async for event in self.execute(workflow, state):
                yield event
            return

        completed = set(state.checkpoint.completed_nodes)
        yield CheckpointResumedEvent(
            completed_count=len(completed),
            resumed_at=state.checkpoint.last_checkpoint_at,
        )
        # _exec skips nodes whose IDs are in completed set
        async for event in self._exec(workflow.root, state, skip_completed=completed):
            yield event

    def _init_pools(self, workflow: WorkflowDefinition, state: WorkflowState) -> None:
        """Initialize pools from WorkflowDefinition.pools declarations."""
        for pool_config in workflow.pools:
            if pool_config.name not in state.pools:
                state.pools[pool_config.name] = PoolState(config=pool_config)

    async def _exec(
        self, node: WorkflowNode, state: WorkflowState,
    ) -> AsyncGenerator[StreamEvent, None]:
        if state.is_cancelled:
            return

        yield NodeStartedEvent(node_id=node.id, node_type=node.type, label=node.label)
        start = time.perf_counter()

        try:
            match node.type:
                case "agent":
                    async for e in self._exec_agent(node, state): yield e
                case "tool":
                    async for e in self._exec_tool(node, state): yield e
                case "sequence":
                    async for e in self._exec_sequence(node, state): yield e
                case "parallel":
                    async for e in self._exec_parallel(node, state): yield e
                case "loop":
                    async for e in self._exec_loop(node, state): yield e
                case "conditional":
                    async for e in self._exec_conditional(node, state): yield e
                case "subworkflow":
                    async for e in self._exec_subworkflow(node, state): yield e

            duration_ms = (time.perf_counter() - start) * 1000
            yield NodeCompletedEvent(node_id=node.id, duration_ms=duration_ms)

        except Exception as exc:
            yield NodeErrorEvent(node_id=node.id, error=str(exc))
            raise  # Propagate — see §13.4 for error handling
```

### 13.2 Node executors (pseudo-code)

**Sequence:**
```python
async def _exec_sequence(self, node, state):
    for child in node.children:
        async for event in self._exec(child, state):
            yield event
```

**Parallel:**
```python
async def _exec_parallel(self, node, state):
    async def run_child(child):
        events = []
        async for event in self._exec(child, state):
            events.append(event)
        return events

    tasks = [run_child(child) for child in node.children]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            raise result
        for event in result:
            yield event
```

Note: parallel children share `state` directly. This is safe because:
- Each child writes to a DIFFERENT `output_key` (validated at load time)
- Pool writes use async locks
- asyncio is single-threaded — no memory races

**Loop:**
```python
async def _exec_loop(self, node, state):
    config = LoopNodeConfig(**node.config)
    body = node.children[0]

    for i in range(config.max_iterations):
        state.iteration_counts[node.id] = i + 1
        yield LoopIterationEvent(node_id=node.id, iteration=i + 1)

        try:
            async for event in self._exec(body, state):
                yield event
        except ParseFailureStopError as e:
            # JSON parse failure in a loop control node → exit loop safely (§7.4)
            yield LoopExitEvent(node_id=node.id, iteration=i + 1, reason="parse_failure",
                                detail=str(e))
            break

        if state.is_cancelled:
            break
        if i + 1 >= config.min_iterations:
            if evaluate_condition(config.until, state):
                yield LoopExitEvent(node_id=node.id, iteration=i + 1, reason="condition_met")
                break
    else:
        yield LoopExitEvent(node_id=node.id, iteration=config.max_iterations,
                            reason="max_iterations")
```

**Conditional:**
```python
async def _exec_conditional(self, node, state):
    config = ConditionalNodeConfig(**node.config)

    for i, condition in enumerate(config.conditions):
        if evaluate_condition(condition, state):
            yield BranchSelectedEvent(node_id=node.id, branch_index=i)
            async for event in self._exec(node.children[i], state):
                yield event
            return

    # Default branch (last child if more children than conditions)
    if len(node.children) > len(config.conditions):
        default_idx = len(node.children) - 1
        yield BranchSelectedEvent(node_id=node.id, branch_index=default_idx, is_default=True)
        async for event in self._exec(node.children[default_idx], state):
            yield event
```

**Agent:**
```python
async def _exec_agent(self, node, state):
    config = resolve_agent_config(node.config)  # Apply subtype defaults (§9)

    # Build AgentInput (§5.5.1 — agent never sees WorkflowState)
    agent_input = AgentInput(
        query=state.query,
        context=self._build_prompt_context(config, state),
        instruction=self._resolve_system_prompt(config),
        tools=self._resolve_agent_tools(config, state),  # external + pool tools
    )

    # Check for registered implementation (subtype delegation)
    if config.subtype and config.subtype in SUBTYPE_IMPLEMENTATIONS:
        result = await SUBTYPE_IMPLEMENTATIONS[config.subtype](config, state, self)
    else:
        system_msg = agent_input.instruction
        user_msg = self._build_user_prompt(config, state)
        # Resolve external tools + auto-generated pool tools
        tools = list(agent_input.tools)

        if tools:
            result = await self._react_loop(system_msg, user_msg, tools, config, state)
        else:
            result = await self.llm.generate(
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                model_tier=config.model_tier,
                model_overrides=state.model_overrides,
                response_format="json" if config.output_format == "json" else None,
            )

    # Parse output
    output = self._parse_output(result, config)

    # Run verification if configured (§11)
    if config.verification and config.verification.enabled:
        output = await self._run_verification(output, state, config.verification)

    # Write to state log (append-only)
    write_output(state, node.id, config.output_key, output)

    # Write to pools (Path A)
    await write_pools(state, config.pool_writes, output)


def _resolve_agent_tools(
    self, config: AgentNodeConfig, state: WorkflowState,
) -> list[ResearchTool]:
    """Resolve external tools + auto-generated pool tools."""
    tools = [self._resolved_tools[ref.name] for ref in config.tools]
    # Auto-generate pool tools for each pool in pool_tools
    for pool_name in config.pool_tools:
        pool = state.pools.get(pool_name)
        if pool:
            registry = PoolRegistry(pool)
            tools.extend(make_pool_tools(pool, registry))
    return tools
```

**Tool:**
```python
async def _exec_tool(self, node, state):
    config = ToolNodeConfig(**node.config)
    tool = self._resolved_tools[config.ref.name]

    # Map state keys to function parameters
    arguments = {}
    for param_name, state_key in config.input_mapping.items():
        arguments[param_name] = resolve_key(state, state_key)

    result = await tool.execute(arguments, self.context.research_context)

    # Path B: auto-append sources to pool
    if result.sources and "sources" in state.pools:
        await state.pools["sources"].extend_async(result.sources)

    state.append(node.id, config.output_key, result.content)
```

**Subworkflow:**
```python
async def _exec_subworkflow(self, node, state):
    config = SubworkflowNodeConfig(**node.config)

    # Resolve workflow: builtin template, database, or dynamic
    child_workflow = await self._resolve_workflow(config, state)

    # Create isolated child state
    child_state = WorkflowState(query=state.query)
    for parent_key, child_key in config.input_mapping.items():
        child_state.append("_init", child_key, resolve_key(state, parent_key))

    # Pool isolation based on pool_mode (§10.2)
    if config.pool_mode == "shared":
        child_state.pools = state.pools       # Shared by reference (default)
    elif config.pool_mode == "isolated":
        child_state.pools = {}                # Fresh pools from child definition

    # Share context
    child_state.enterprise_tools = state.enterprise_tools
    child_state.user_token = state.user_token
    child_state.model_overrides = state.model_overrides
    child_state.is_cancelled = state.is_cancelled

    # Execute
    async for event in self.execute(child_workflow, child_state):
        yield event

    # Map outputs back (only these cross the boundary)
    if config.output_mapping:
        for child_key, parent_key in config.output_mapping.items():
            state.append(node.id, parent_key, child_state.get(child_key))
    else:
        state.append(node.id, config.output_key, child_state.get("output"))

    # Write subworkflow output to pools (if configured)
    if config.pool_writes:
        output_value = state.get(config.output_key)
        await write_pools(state, config.pool_writes, output_value)


async def _resolve_workflow(
    self, config: SubworkflowNodeConfig, state: WorkflowState,
) -> WorkflowDefinition:
    """Resolve a subworkflow ref to a WorkflowDefinition."""
    if config.ref.startswith("builtin:"):
        # Parameterized template (§12)
        template_name = config.ref
        if template_name not in BUILTIN_TEMPLATE_REGISTRY:
            raise WorkflowError(f"Unknown builtin template: {template_name}")
        params_cls, generator_fn = BUILTIN_TEMPLATE_REGISTRY[template_name]
        validated_params = params_cls(**config.params)
        return generator_fn(validated_params)
    elif config.ref == "__dynamic__":
        # Dynamic workflow from state
        plan = state.data.get("generated_plan")
        return self._validate_and_build_dynamic(plan, config)
    else:
        # Database lookup
        return await self._load_workflow_from_db(config.ref)
```

**Plan and Execute:**
```python
async def _exec_plan_and_execute(self, node, state):
    config = PlanAndExecuteNodeConfig(**node.config)
    replan_cycles = 0
    total_items_processed = 0

    while True:
        # Run planner (creates/updates plan with items)
        async for event in self._exec_agent_inline(config.planner, node.id + "_planner", state):
            yield event

        plan = state.get(config.planner.output_key or "plan")
        items = plan.get(config.items_path, []) if isinstance(plan, dict) else []
        yield ItemsExtractedEvent(node_id=node.id, total_items=len(items),
                                  items_path=config.items_path, cycle=replan_cycles + 1)

        for item_index, item in enumerate(items):
            total_items_processed += 1
            state.append(node.id, config.item_state_key, item)
            yield ItemStartedEvent(node_id=node.id, item_index=item_index,
                                   item_summary=str(item.get("title", ""))[:200] if isinstance(item, dict) else str(item)[:200],
                                   total_items=len(items))

            # Execute body
            async for event in self._exec(config.body, state):
                yield event

            yield ItemCompletedEvent(node_id=node.id, item_index=item_index,
                                     items_processed=total_items_processed)

            # Execute evaluator (if present)
            if config.evaluator is not None:
                async for event in self._exec_agent_inline(config.evaluator, node.id + "_evaluator", state):
                    yield event

                evaluation = state.get(config.evaluator.output_key or "evaluation") or {}
                decision = evaluation.get("decision", "continue").lower()

                if decision == "complete" and total_items_processed >= config.min_iterations:
                    yield EvaluationDecisionEvent(node_id=node.id, decision="complete",
                                                  reasoning=evaluation.get("reasoning", ""),
                                                  items_processed=total_items_processed)
                    yield PlanAndExecuteExitEvent(node_id=node.id, reason="complete",
                                                  total_items_processed=total_items_processed,
                                                  replan_cycles=replan_cycles)
                    return
                elif decision == "replan":
                    replan_cycles += 1
                    yield ReplanTriggeredEvent(node_id=node.id, cycle=replan_cycles,
                                               reason=evaluation.get("reasoning", ""),
                                               items_remaining=len(items) - item_index - 1)
                    if replan_cycles >= config.max_replan_cycles:
                        yield PlanAndExecuteExitEvent(node_id=node.id, reason="max_replan_cycles",
                                                      total_items_processed=total_items_processed,
                                                      replan_cycles=replan_cycles)
                        return
                    break  # Break item loop, re-run planner
                else:
                    yield EvaluationDecisionEvent(node_id=node.id, decision="continue",
                                                  reasoning=evaluation.get("reasoning", ""),
                                                  items_processed=total_items_processed)

            if total_items_processed >= config.max_iterations or state.is_cancelled:
                yield PlanAndExecuteExitEvent(node_id=node.id, reason="max_iterations" if not state.is_cancelled else "cancelled",
                                              total_items_processed=total_items_processed,
                                              replan_cycles=replan_cycles)
                return
        else:
            # All items completed without replan
            if config.complete_on_exhaustion:
                yield PlanAndExecuteExitEvent(node_id=node.id, reason="exhausted",
                                              total_items_processed=total_items_processed,
                                              replan_cycles=replan_cycles)
                return
            else:
                replan_cycles += 1
                if replan_cycles >= config.max_replan_cycles:
                    yield PlanAndExecuteExitEvent(node_id=node.id, reason="max_replan_cycles",
                                                  total_items_processed=total_items_processed,
                                                  replan_cycles=replan_cycles)
                    return
                # Continue to re-plan
```

### 13.3 Streaming events

Every node emits `NodeStartedEvent` and `NodeCompletedEvent`. The node `id` +
`label` enable the UI to show a progress tree. Additional events per type:

| Event                      | Emitted by       | Purpose                              |
|----------------------------|------------------|--------------------------------------|
| `NodeStartedEvent`         | All              | Node began executing                 |
| `NodeCompletedEvent`       | All              | Node finished                        |
| `NodeErrorEvent`           | All              | Node failed                          |
| `LoopIterationEvent`       | `loop`           | Starting iteration N                 |
| `LoopExitEvent`            | `loop`           | Loop terminated (reason: condition_met, max_iterations, parse_failure) |
| `BranchSelectedEvent`      | `conditional`    | Which branch was chosen              |
| `ToolCallEvent`            | `agent`          | LLM decided to call a tool           |
| `ToolResultEvent`          | `agent`/`tool`   | Tool returned a result               |
| `ToolCacheHitEvent`        | `agent`/`tool`   | Tool call skipped (cached result, §5.11) |
| `AgentOutputEvent`         | `agent`          | Agent produced final output          |
| `PlanCreatedEvent`         | planner builtin  | Plan created/updated (§9.5)          |
| `ReflectionDecisionEvent`  | reflector builtin| CONTINUE/ADJUST/COMPLETE (§9.5)      |
| `ItemsExtractedEvent`      | `plan_and_execute` | Items extracted from planner output (§4.6) |
| `ItemStartedEvent`         | `plan_and_execute` | Item processing began (§4.6)         |
| `ItemCompletedEvent`       | `plan_and_execute` | Item processing finished (§4.6)      |
| `EvaluationDecisionEvent`  | `plan_and_execute` | Evaluator decision (§4.6)            |
| `ReplanTriggeredEvent`     | `plan_and_execute` | Replan triggered (§4.6)              |
| `PlanAndExecuteExitEvent`  | `plan_and_execute` | Cycle completed (§4.6)               |
| `CoordinatorClassifiedEvent`| coordinator builtin | Query classified (§9.5)           |
| `BackgroundCompletedEvent` | background builtin| Background investigation done (§9.5) |
| `SynthesisStartedEvent`    | synthesizer builtin| Synthesis began (§9.5)              |
| `CheckpointSavedEvent`     | executor         | State checkpointed to DB (§13.7)     |
| `CheckpointResumedEvent`   | executor         | Execution resumed from checkpoint (§13.7) |
| `TokenUsageEvent`          | executor         | Periodic token usage report (§13.8)  |
| `TokenBudgetExceededEvent` | executor         | Token budget exhausted (§13.8)       |
| `GateWaitingEvent`         | executor         | Workflow paused for human input (§13.9) |
| `GateResumedEvent`         | executor         | Human responded to gate (§13.9)      |
| `GateTimeoutEvent`         | executor         | Gate timed out (§13.9)               |
| `ConversationCompactedEvent`| executor        | Agent conversation compacted (§5.12)  |

These compose with existing `StreamEvent` types. The UI receives a flat stream
of events tagged with `node_id`, allowing it to reconstruct the tree progress.

### 13.4 Error handling

Per-node error configuration (optional, with defaults):

```python
class ErrorConfig(BaseModel):
    on_error: Literal["fail", "skip", "retry"] = "fail"
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
```

- **`fail`** (default): Error propagates up the tree. Parent nodes abort.
  The workflow terminates with an error event.
- **`skip`**: Node is marked as skipped. Execution continues to the next
  sibling (in a sequence) or ignores this child's output (in parallel).
  The node's `output_key` is not written to state.
- **`retry`**: Re-execute the node up to `max_retries` times with exponential
  backoff. If all retries fail, fall through to `fail`.

For parallel nodes with `skip`, partial results are still available — the
children that succeeded have their outputs in state.

### 13.4.1 Error Propagation in Nested Structures

Error propagation follows these rules in nested contexts:

| Context | `on_error: fail` | `on_error: skip` | `on_error: retry` |
|---------|-------------------|--------------------|--------------------|
| Parallel child | Cancel siblings, propagate up | Continue with other siblings, emit `NodeSkippedEvent` | Retry up to `max_retries`, then apply fallback |
| Loop body | Exit loop, propagate up | Skip iteration, continue loop | Retry iteration, then apply fallback |
| plan_and_execute body | Exit cycle, propagate up | Skip item, continue to next | Retry item, then apply fallback |
| Sequence child | Abort remaining children, propagate up | Skip child, continue to next | Retry child, then apply fallback |

For parallel nodes: when one child fails with `on_error: fail`, all running
siblings receive cancellation. Results from already-completed siblings are
preserved in state.

### 13.5 Cancellation

`state.is_cancelled` is checked at these points:
1. Before each node starts executing
2. Before each loop iteration
3. Before each parallel child starts
4. After each tool call in ReAct loop
5. After each iteration in loops and plan_and_execute
6. In parallel nodes: cancel siblings on failure (configurable via ErrorConfig)

Check: `if state.is_cancelled: raise WorkflowCancelledError()`

When cancelled, the executor stops starting new work but does NOT forcefully
terminate running LLM calls or tool executions. Those complete naturally, and
their results are discarded.

### 13.6 Timeouts

Configurable at two levels:
- **Workflow-level**: `WorkflowDefinition.timeout_seconds` (default: 1800s)
- **Node-level**: `WorkflowNode.config.timeout_seconds` (optional, per-node)

The executor tracks elapsed time and sets `state.is_cancelled = True` when the
workflow timeout is exceeded.

### 13.7 Checkpointing & Recovery

Crash at minute 28 of a 30-minute workflow loses everything. Checkpointing
persists `WorkflowState` after each leaf node completes, enabling resume.

**Key fact**: `research_sessions.execution_state` JSONB column already exists.
`ResearchState.to_dict()` already exists. Two-phase persistence pattern already
exists in `persistence.py`. No migration needed.

#### Data structures

```python
@dataclass
class NodeCheckpoint:
    node_id: str
    status: Literal["completed", "failed", "skipped"]
    timestamp: str  # ISO

@dataclass
class CheckpointState:
    completed_nodes: list[str]      # IDs of completed nodes
    node_statuses: dict[str, str]   # node_id → status
    state_snapshot: dict[str, Any]  # Full to_dict() at last checkpoint
    last_checkpoint_at: str         # ISO timestamp
```

#### Checkpoint flow

1. After each leaf node's `NodeCompletedEvent`, executor calls `_checkpoint_state()`
2. `_checkpoint_state()` serializes `WorkflowState.to_dict()` → writes to
   `execution_state` JSONB via independent session (pattern from `persistence.py`)
3. Emits `CheckpointSavedEvent`
4. On resume: `WorkflowState.from_dict()` loads from JSONB,
   `resume_from_checkpoint()` (§13.1) skips completed nodes

#### Checkpoint configuration

Configured on `WorkflowDefinition`:
```python
class CheckpointConfig(BaseModel):
    enabled: bool = True
    granularity: Literal["every_node", "every_n"] = "every_node"
    checkpoint_interval: int = 1    # Checkpoint every N leaf nodes
```

### 13.8 Token Budget Enforcement

Parallel + loop workflows can run up unlimited LLM costs. Token budgets
provide a hard ceiling.

**Key fact**: `LLMClient` already returns `usage: dict[str, int]` with
`prompt_tokens`, `completion_tokens`, `total_tokens` in every response.

#### TokenBudget tracker

```python
@dataclass
class NodeTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class TokenBudget:
    max_total_tokens: int           # 0 = unlimited
    total_tokens: int = 0           # Cumulative
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    node_usage: dict[str, NodeTokenUsage] = field(default_factory=dict)

    def record_usage(self, node_id: str, usage: dict[str, int]) -> None:
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        nu = self.node_usage.setdefault(node_id, NodeTokenUsage())
        nu.total_tokens += usage.get("total_tokens", 0)
        nu.prompt_tokens += usage.get("prompt_tokens", 0)
        nu.completion_tokens += usage.get("completion_tokens", 0)

    def is_exceeded(self) -> bool:
        return self.max_total_tokens > 0 and self.total_tokens >= self.max_total_tokens

    def remaining_tokens(self) -> int:
        if self.max_total_tokens <= 0:
            return float("inf")
        return max(0, self.max_total_tokens - self.total_tokens)
```

#### BudgetAwareLLMClient wrapper

```python
class BudgetAwareLLMClient:
    """Wrapper: checks budget before each call, records usage after."""
    def __init__(self, inner: LLMClient, budget: TokenBudget):
        self._inner = inner
        self._budget = budget

    async def complete(self, **kwargs) -> LLMResponse:
        if self._budget.is_exceeded():
            raise TokenBudgetExceededError(
                used=self._budget.total_tokens,
                limit=self._budget.max_total_tokens,
            )
        response = await self._inner.complete(**kwargs)
        self._budget.record_usage(kwargs.get("node_id", "unknown"), response.usage)
        return response
```

#### Integration

- `WorkflowDefinition.token_budget: int = 0` (per-workflow limit, 0 = unlimited)
- `WorkflowState.token_budget: TokenBudget | None`
- Executor checks `is_exceeded()` before each node in `_exec()`
- Budget included in `ResearchCompletedEvent` and checkpoints
- Resolves Open Question #5

**Implementation file**: `src/deep_research/services/llm/token_budget.py`

### 13.9 Human-in-the-Loop Gates

Workflows may need human approval before proceeding past certain nodes (e.g.,
plan review, budget approval, sensitive data access).

**Key fact**: `PlanReviewEvent` + `asyncio.Queue` response pattern already
exists in `orchestrator.py:3303`. `enable_plan_review` and
`require_plan_approval` already exist on `OrchestrationConfig`. This
generalizes that pattern to any node.

#### GateConfig

```python
class GateConfig(BaseModel):
    type: Literal["approval", "review", "input"] = "approval"
    message: str = "Review required before continuing"
    timeout_seconds: int = 300
    auto_proceed_on_timeout: bool = True
    include_state_keys: list[str] = []   # State keys to expose for review
    include_pools: list[str] = []        # Pools to expose for review
    trigger: Literal["before", "after"] = "before"
```

#### WorkflowNode integration

Any node can have an optional `gate`:
```python
class WorkflowNode(BaseModel):
    # ... existing fields
    gate: GateConfig | None = None      # See §3.1
```

#### Execution flow

1. Before/after node (based on `trigger`), executor checks `node.gate`
2. Checkpoint state (§13.7) to ensure recoverability
3. Emit `GateWaitingEvent` with snapshot of requested state keys/pools
4. `await context.response_queue.get()` (with timeout)
5. On approve → continue. On reject → cancel workflow. On edit → apply
   modifications to `state.data`
6. On timeout → auto-proceed or cancel (based on `auto_proceed_on_timeout`)

#### New events

- `GateWaitingEvent(node_id, gate_type, message, state_snapshot)` — workflow paused
- `GateResumedEvent(node_id, action, modifications)` — human responded
- `GateTimeoutEvent(node_id, auto_proceeded)` — timeout reached

#### Resume API (framework only, UI later)

- `POST /jobs/{session_id}/gate/{gate_id}/respond` — accepts `{action, modifications}`
- Job manager routes response to the workflow's `response_queue`

#### Backward compatibility

`enable_plan_review=True` maps to `gate: {type: review, trigger: after}` on the
planner node. Existing plan review behavior is preserved.

Resolves Open Question #7.

---

## 14. Backward Compatibility

### 14.1 Three levels of abstraction

Users interact at the level that matches their sophistication:

**Level 1 — Preset Steps (current UX, zero learning curve):**
User provides a list of steps with titles and descriptions. The system
auto-generates the workflow tree:

```python
def preset_steps_to_tree(steps: list[AgentPresetStep]) -> WorkflowNode:
    """Convert legacy preset steps into a workflow tree."""
    research_nodes = []
    for step in sorted(steps, key=lambda s: s.order):
        agent_config = {
            "subtype": "researcher",
            "system_prompt": f"Research the following aspect: {step.title}\n{step.description or ''}",
            "tools": _tools_from_hints(step.source_hints),
            "input_keys": ["query", "findings"],
            "output_key": "findings",
            "output_mode": "append",
            "pool_writes": [{"pool": "observations"}],
        }
        research_nodes.append(WorkflowNode(
            id=f"step_{step.order}",
            type=NodeType.AGENT,
            label=step.title,
            config=agent_config,
        ))

    return WorkflowNode(
        id="root",
        type=NodeType.SEQUENCE,
        label="Research Pipeline",
        children=[
            *research_nodes,
            WorkflowNode(
                id="synthesizer",
                type=NodeType.AGENT,
                label="Synthesizer",
                config={
                    "subtype": "synthesizer",
                    "verification": {"enabled": True},
                },
            ),
        ],
    )
```

**Level 2 — Template Selection:**
User picks a parameterized template (e.g., "Best-of-N Research",
"Self-Critique Report") and customizes parameters (which models, how many
iterations, which tools). The template generates a tree via `builtin:` refs.

**Level 3 — Full Tree Editor (advanced):**
User defines the complete workflow tree in YAML/JSON or a visual editor.

### 14.2 Database migration

Add a nullable JSONB column to `CustomAgent`:

```sql
ALTER TABLE custom_agents ADD COLUMN workflow_tree JSONB;
```

When `workflow_tree` is NULL -> legacy code path (current orchestrator).
When `workflow_tree` is set -> `WorkflowExecutor` runs the tree.

This allows gradual migration. Existing custom agents keep working unchanged.

### 14.3 Mapping current agent modes to trees

| Current `default_mode` | Generated tree                                                  |
|------------------------|-----------------------------------------------------------------|
| `planner`              | `sequence(agent(planner), loop(seq(researcher, reflector)), agent(synthesizer))` |
| `manual`               | `sequence(*[agent(step) for step in preset_steps], agent(synthesizer))` |
| `hybrid`               | `sequence(agent(planner_with_seeds), loop(...), agent(synthesizer))` |

### 14.4 Pool migration (legacy fields removed)

See §5.8. Legacy accumulated side-effect fields (`sources`, `evidence_pool`,
`claims`, `all_observations`) are removed from `WorkflowState`. All
accumulation goes through pools exclusively. Existing code that referenced
these fields must be migrated to use `state.pools["sources"]`,
`state.pools["observations"]`, etc.

### 14.5 Verification flag mapping

| OrchestrationConfig flag | Maps to |
|---|---|
| `verify_sources` | `verification.enabled` on synthesizer node |
| `enable_post_verification` | `verification.stages.isolated_verification` + `.citation_correction` |
| `enable_citation_verification` | Presence of `verification` block on synthesizer |

---

## 15. Integration with Existing Code

### 15.1 Existing agent functions

The current codebase has specialized functions: `run_coordinator`,
`run_planner`, `run_researcher`, `run_react_researcher`, `run_reflector`,
`run_synthesizer`. These contain sophisticated logic: prompt templates,
token budgeting, error recovery, source tracking, quality signals.

**Integration strategy:**

For the **default workflow template** (the "Default Deep Research" tree),
agent nodes map to existing functions via a role registry AND a subtype
registry:

```python
ROLE_IMPLEMENTATIONS: dict[str, Callable] = {
    "Coordinator": run_coordinator,
    "Planner": run_planner,
    "Researcher": run_react_researcher,  # or run_researcher based on config
    "Reflector": run_reflector,
    "Synthesizer": run_citation_synthesizer,
    "Background Investigator": run_background_investigator,
}

SUBTYPE_IMPLEMENTATIONS: dict[str, Callable] = {
    "researcher": run_react_researcher,
    "synthesizer": run_citation_synthesizer,
    "reflector": run_reflector,
    "planner": run_planner,
}
```

When the executor encounters an agent node whose `subtype` or `role` matches a
registered implementation, it delegates to that function instead of the generic
LLM call. This preserves all existing quality and sophistication.

For **user-defined roles** (no registered implementation), the executor uses
the generic agent execution path: build prompt from config, resolve pool tools,
call LLM with ReAct loop, parse output.

Over time, the useful parts of existing functions (token budgeting, source
tracking, quality signals) can be extracted into reusable middleware that any
agent node can use.

### 15.2 OrchestrationConfig mapping

Most `OrchestrationConfig` fields map to the workflow tree or execution context:

| OrchestrationConfig field       | New location                                |
|--------------------------------|---------------------------------------------|
| `query_mode`                   | Top-level `conditional` node in the tree    |
| `research_depth`               | Loop `max_iterations` + agent `model_tier`  |
| `workflow_mode`                | Tree structure (Level 1/2/3 abstraction)    |
| `manual_steps`                 | Agent nodes in the tree                     |
| `enable_background_investigation` | Presence/absence of background agent node  |
| `enable_clarification`         | Presence/absence of clarification loop      |
| `model_overrides`              | `WorkflowState.model_overrides` (context)   |
| `domain_filter`                | `WorkflowState.domain_filter` (context)     |
| `source_scope`                 | Per-agent `tools` list                      |
| `verify_sources`               | `verification` config on synthesizer        |
| `output_format`                | Synthesizer agent's `output_format`         |
| `output_schema`                | Synthesizer agent's `output_schema`         |
| `timeout_seconds`              | `WorkflowDefinition.timeout_seconds`        |
| `user_token`                   | `WorkflowState.user_token` (context)        |

Fields that are truly global (auth, timeouts, user identity) live in
`ExecutionContext`, not in the tree. The tree only contains structural and
behavioral config.

### 15.3 Domain Context Tracker

The `DomainContextTracker` is a thin event-forwarding adapter (~200-250 LOC).
It is NOT a state reconstructor — enriched domain events are self-contained
with all metadata the app needs.

**Three responsibilities:**
1. **Pattern match** `event_type` → produce `list[AppSSEEvent]` (trivial forwarding)
2. **Accumulate** `PersistenceDelta` (sources, observations, step updates)
3. **Signal** when persistence is needed via `should_persist()`

```python
@dataclass
class PersistenceDelta:
    """Incremental changes since last persistence checkpoint."""
    # Session metadata (set once)
    complexity: str | None = None
    recommended_depth: str | None = None
    # Plan (replaced on each PlanCreatedEvent)
    plan: dict[str, Any] | None = None
    plan_steps: list[dict] | None = None
    # Accumulated (append-only between checkpoints)
    new_sources: list[dict] = field(default_factory=list)
    new_observations: list[str] = field(default_factory=list)
    step_updates: dict[str, dict] = field(default_factory=dict)
    # Final
    final_report: str | None = None


class DomainContextTracker:
    """Event-forwarding adapter with persistence delta accumulation.

    Each handler is 5-15 lines of pattern matching. Total: ~200-250 LOC
    (down from the originally planned 600-800 for a state reconstructor).
    """

    def __init__(self, config: OrchestrationConfig) -> None: ...

    def process_event(self, event: StreamEvent) -> list[AppSSEEvent]:
        """Pattern-match event_type → forward to app SSE events.

        Also accumulates PersistenceDelta incrementally.
        Enriched domain events are self-contained — no dict parsing,
        no state reconstruction needed.
        """
        ...

    def get_persistence_delta(self) -> PersistenceDelta:
        """Return accumulated changes since last checkpoint.

        Resets internal accumulators after returning.
        """
        ...

    def should_persist(self) -> bool:
        """Check if enough state has changed to warrant persistence."""
        ...
```

**Why event-forwarding (not state reconstruction)?** Enriched domain events
(§9.5) carry all metadata the app needs — `CoordinatorClassifiedEvent` includes
`direct_response` and `reasoning`, `BackgroundCompletedEvent` includes the full
`data_landscape` and `query_decomposition`, `WorkflowCompletedEvent` includes
`final_report` and `total_sources`. The tracker simply forwards these fields
to the corresponding SSE events. No `get_research_state()` method needed.

**Orchestrator usage**:
```python
tracker = DomainContextTracker(config)
async for event in executor.execute(definition, state):
    sse_events = tracker.process_event(event)
    for sse in sse_events:
        yield sse
    if tracker.should_persist():
        await persist(tracker.get_persistence_delta())
```

---

## 16. Validation

### 16.1 Load-time validation (before execution starts)

| Rule                                              | Severity |
|---------------------------------------------------|----------|
| All node IDs are unique within the tree           | Error    |
| Leaf nodes have empty children                    | Error    |
| Container nodes have required children count      | Error    |
| Parallel children have non-overlapping output_keys| Error    |
| Loop has exactly one child                        | Error    |
| Conditional has >= 1 child (gate pattern allowed)  | Error    |
| Tool refs point to known types                    | Error    |
| Subworkflow refs exist in database or builtin registry | Error |
| Pool names in pool_writes/pool_tools exist in WorkflowDefinition.pools | Warning |
| SubworkflowNodeConfig.pool_writes reference pools declared in WorkflowDefinition.pools | Warning |
| Pool names are unique within WorkflowDefinition.pools | Error |
| Builtin template params validate against schema   | Error    |
| input_keys reference keys that WILL exist (produced by a preceding node or a required_input) | Warning |
| Tree depth does not exceed max (configurable, default 10) | Error |
| Total node count does not exceed max (configurable, default 50) | Error |
| **DataFlowGraph has no errors** (§5.10) — missing producers, overlapping parallel keys | Error |
| **DataFlowGraph warnings** — unused producer keys | Warning |
| **Dot-path references validated against producer `output_schema`** (§5.10 step 4) | Warning |
| **Condition operators type-compatible with schema-declared types** (§5.10 step 4) | Warning |
| **Templates validated against safe syntax** (§7.5) — user_prompt_template, system_prompt | Error |
| **Dynamic subworkflow constraints** (§10.5) — tool allowlist, subtypes_only, model tier | Error |
| **Dynamic subworkflow DataFlowGraph** — generated tree validated same as static trees | Error |

### 16.2 Runtime validation

| Rule                                              | Action   |
|---------------------------------------------------|----------|
| LLM returns invalid JSON when `output_format: json` | Apply ParseFailureConfig (§7.4): retry, then stop or continue |
| Tool execution fails                              | Apply error handling config |
| Condition key not found in state                  | Evaluate as `None` (safe for operators) |
| Subworkflow not found at runtime                  | Error (node fails)  |
| Pool not found for pool_write                     | Error (not warning — pools are load-bearing) |
| Verification pipeline fails                       | Log warning, use unverified output |
| Conversation budget exceeded                      | Trigger conversation compaction (§5.12) |
| Timeout exceeded                                  | Cancel    |
| Token budget exceeded (§13.8)                     | Cancel with `TokenBudgetExceededEvent` |
| Gate timeout (§13.9)                              | Auto-proceed or cancel (per `auto_proceed_on_timeout`) |

---

## 17. Observability

Each node execution creates an **MLflow span**, forming a trace tree that
mirrors the workflow tree:

```
research_orchestration (CHAIN)
├── coordinator (AGENT)
├── research_flow (CHAIN)
│   ├── init (CHAIN)
│   │   ├── background (AGENT)
│   │   └── planner (AGENT)
│   ├── plan_and_execute (CHAIN)
│   │   ├── iteration_1 (CHAIN)
│   │   │   ├── researcher (AGENT)
│   │   │   │   ├── web_search (TOOL)
│   │   │   │   └── web_crawl (TOOL)
│   │   │   └── reflector (AGENT)
│   │   └── iteration_2 (CHAIN)
│   │       ├── researcher (AGENT)
│   │       └── reflector (AGENT)
│   └── synthesizer (AGENT)
│       └── verification (CHAIN)    <- if verification enabled
│           ├── evidence_preselection (TOOL)
│           ├── confidence_classification (AGENT)
│           └── citation_correction (AGENT)
```

Span attributes include: node type, label, duration, model tier, tool calls
count, output key, output size, pool write count. This makes debugging
straightforward.

---

## 18. Open Questions

1. **Conversation history in loops** — Should the researcher agent see its own
   prior outputs? Currently the ReAct researcher maintains its own message
   history. In the new model, the loop body executes fresh each time. Should we
   carry forward agent message history across iterations?

2. **Parallel event ordering** — When parallel children emit events, should they
   be interleaved (more responsive) or buffered per-child (easier to display)?
   Buffering is simpler; interleaving requires UI work.

3. **Dynamic plan validation** — ~~For `DynamicPlanNode`, should the generated
   tree be validated by a separate "safety" LLM before execution? Or are
   structural constraints (max nodes, max depth, allowed types) sufficient?~~
   **RESOLVED**: 4-layer defense (structural + tool allowlist + argument policy +
   runtime budgets). See §10.5.

4. **Template versioning** — When a builtin template is updated, do existing
   workflows that used it get the update? Since templates are generators (not
   stored snapshots), they always produce the latest version. Should we support
   pinned versions for reproducibility?

5. ~~**Cost budgets** — Should nodes have token/cost budgets? A `parallel` node
   with 10 `complex`-tier agents is expensive. Budget awareness could prevent
   runaway costs.~~
   **RESOLVED**: `TokenBudget` on `WorkflowState` + `BudgetAwareLLMClient`
   wrapper. See §13.8.

6. **Streaming synthesis** — The current synthesizer streams tokens via SSE.
   How does this integrate with the node-based event model? Probably
   `SynthesisProgressEvent` carries token chunks, same as today.

7. ~~**Plan review HITL** — Current system supports `PlanReviewEvent` where the
   user can approve/modify the plan. In the tree model, this maps to a `gate`
   mechanism on certain nodes. Should `gate` be a node type (#8) or a config
   on any node?~~
   **RESOLVED**: `GateConfig` on any `WorkflowNode` (not a new node type).
   See §13.9.

8. ~~**Pool scoping for nested subworkflows** — Currently pools are shared by
   reference across all subworkflow levels. Should there be an option to create
   pool-isolated subworkflows (e.g., for BestOfN where candidates shouldn't
   see each other's findings)?~~
   **RESOLVED**: `pool_mode: Literal["shared", "isolated"]` on
   `SubworkflowNodeConfig` (§4.5, §10.2). `"shared"` (default) shares pools
   by reference. `"isolated"` gives the child its own pools from its
   `WorkflowDefinition.pools` — parent pools not accessible, only
   `output_mapping` keys cross the boundary. BestOfN and Debate templates
   default to `pool_mode: "isolated"` for candidates/advocates.

9. **Subtype extensibility** — Should users be able to register custom subtypes
   with their own default configs, or is the 5-subtype set sufficient? Plugin-
   defined subtypes could enable domain-specific agent patterns.

---

## 19. Implementation Phases

| Phase | Scope | What it enables |
|-------|-------|-----------------|
| **P0: Foundation + Pools + Subtypes** | `WorkflowNode`, `WorkflowState`, executor for `agent`, `sequence`, `loop`, shared research pools (`PoolState`, `PoolConfig`), standard agent subtypes (researcher, synthesizer, verifier, planner, reflector) with implementation delegation | Replace hardcoded pipeline for custom agents with basic trees. Researchers accumulate to shared pools. Synthesizers read accumulated findings. Subtypes give one-line agent config. |
| **P1: Branching, Parallelism & Verification** | `conditional`, `parallel`, `tool` node execution, `VerificationConfig` integration with existing citation pipeline, `verify_and_ground` builtin tool | Multi-path workflows, concurrent research, UC tool integration. Verification/grounding activatable per-agent or as standalone step. |
| **P2: Composition & Templates** | `subworkflow` node, workflow definitions DB table, input/output contracts, pool sharing across subworkflows, parameterized builtin templates (best_of_n, self_critique, debate, majority_vote) | Reusable workflow fragments, sub-agents. Convenience patterns via `builtin:` refs with params. Users get BestOfN/Debate/etc. without building trees. |
| **P3: Dynamic & Advanced** | `__dynamic__` subworkflow, LLM-generated trees with validation, plugin-defined templates, user-defined workflows in DB | Self-organizing agent teams, adaptive planning. Full extensibility. |

---

## Appendix A — Real-World Use Cases

These two use cases validate that the 7-primitive architecture (§2) can express
complex, production-grade workflows. Each use case was designed from realistic
customer requirements, and gaps discovered during design drove the architecture
improvements in §4.4 (gate pattern), §4.5 (subworkflow pool_writes), and §12.4
(template pool params).

---

### A.1 Multi-Source Deep Research with Quality Assurance

**Scenario**: A custom agent combines web search with enterprise sources
(Knowledge Assistants, Vector Search). Uses a research loop with parallel
multi-source steps, BestOfN synthesis for quality, and standalone verification
for grounding.

#### Architecture tree

```
sequence
├── agent: "Planner" (subtype: planner)
├── loop: "Research Cycle" (max: 7, min: 2)
│   └── sequence
│       ├── parallel: "Multi-Source Research"
│       │   ├── agent: "Web Researcher" (web_search, web_crawl → web_findings pool)
│       │   ├── agent: "KA Researcher" (enterprise:product_docs_ka → enterprise_findings pool)
│       │   └── agent: "VS Researcher" (enterprise:technical_docs_vs → enterprise_findings pool)
│       └── agent: "Reflector" (reads all pools, decides continue/adjust/complete)
├── subworkflow: "Best-of-3 Synthesis" (builtin:best_of_n)
│   params: n=3, diverse_tiers=true
│   candidate_pool_tools: [observations, sources, web_findings, enterprise_findings]
│   output_key: best_report
└── tool: "verify_and_ground" (verification on best_report → report)
```

#### Features demonstrated

| Feature | Where used | Section |
|---------|-----------|---------|
| Parallel multi-source research | 3 agents in parallel node | §2, §13.2 |
| Topic-specific pools | web_findings, enterprise_findings | §5.7 |
| BestOfN synthesis with pool access | builtin:best_of_n with candidate_pool_tools | §12.4 |
| Standalone verification tool | verify_and_ground as tool node | §11.2(b) |
| Planner with subtype defaults | One-line `{subtype: planner}` | §9 |
| Reflector with pool tools | pool_tools for coverage assessment via search | §5.3, §5.7 |
| Enterprise tools | KA + VS via ToolRef type=enterprise | §8.1 |

#### Full WorkflowDefinition

```yaml
id: "multi-source-deep-research-qa"
name: "Multi-Source Deep Research with Quality Assurance"
description: >
  Parallel web + enterprise research with BestOfN synthesis and verification.
version: 1
required_inputs: [query]
output_keys: [report]

pools:
  - name: observations
    item_type: text
  - name: sources
    item_type: source
    dedup_key: url
  - name: web_findings
    item_type: text
  - name: enterprise_findings
    item_type: text
  - name: claims
    item_type: claim
    max_items: 100

root:
  id: root
  type: sequence
  label: "Multi-Source Deep Research Pipeline"
  children:

    # ── Phase 1: Planning ──
    - id: planner
      type: agent
      label: "Research Planner"
      config:
        subtype: planner
        input_keys: [query]
        system_prompt: |
          You are a research planner. Given a query, produce a structured plan
          that identifies which aspects should be researched via web search and
          which require enterprise knowledge bases (product docs, technical docs).
          Output JSON with: title, steps[], where each step has a title, description,
          and preferred_sources (web, enterprise, both).

    # ── Phase 2: Research Loop ──
    - id: research_loop
      type: loop
      label: "Research Cycle"
      config:
        until:
          key: "reflection.decision"
          operator: eq
          value: "complete"
        max_iterations: 7
        min_iterations: 2
      children:
        - id: loop_body
          type: sequence
          label: "Research Iteration"
          children:

            # Parallel multi-source research
            - id: multi_source
              type: parallel
              label: "Multi-Source Research"
              children:

                - id: web_researcher
                  type: agent
                  label: "Web Researcher"
                  config:
                    subtype: researcher
                    output_key: web_step_findings
                    tools:
                      - {type: builtin, name: web_search}
                      - {type: builtin, name: web_crawl}
                    pool_writes:
                      - {pool: observations}
                      - {pool: web_findings}
                    pool_tools: [observations]
                    input_keys: [query, plan]
                    system_prompt: |
                      You are a web researcher. Search the open web for information
                      relevant to the current research plan step. Focus on authoritative
                      sources: academic papers, official documentation, reputable news.

                - id: ka_researcher
                  type: agent
                  label: "KA Researcher"
                  config:
                    role: "Knowledge Assistant Researcher"
                    model_tier: analytical
                    output_key: ka_step_findings
                    output_mode: append
                    tools:
                      - {type: enterprise, name: product_docs_ka}
                    pool_writes:
                      - {pool: observations}
                      - {pool: enterprise_findings}
                    pool_tools: [observations]
                    input_keys: [query, plan]
                    system_prompt: |
                      You are an enterprise knowledge researcher. Query the product
                      documentation Knowledge Assistant to find internal information
                      relevant to the research plan. Cite specific documents and sections.

                - id: vs_researcher
                  type: agent
                  label: "VS Researcher"
                  config:
                    role: "Vector Search Researcher"
                    model_tier: analytical
                    output_key: vs_step_findings
                    output_mode: append
                    tools:
                      - {type: enterprise, name: technical_docs_vs}
                    pool_writes:
                      - {pool: observations}
                      - {pool: enterprise_findings}
                    pool_tools: [observations]
                    input_keys: [query, plan]
                    system_prompt: |
                      You are a technical documentation researcher. Search the vector
                      index of technical docs for implementation details, architecture
                      decisions, and API specifications relevant to the research plan.

            # Reflect on accumulated findings
            - id: reflector
              type: agent
              label: "Research Reflector"
              config:
                subtype: reflector
                input_keys: [query, plan]
                pool_tools: [observations, sources, web_findings, enterprise_findings]
                system_prompt: |
                  You are a research reflector. Evaluate the accumulated findings
                  across ALL sources (web, Knowledge Assistant, Vector Search).
                  Assess: coverage of the plan, quality of evidence, gaps remaining.
                  If important aspects are uncovered but shallow, output "continue".
                  If the plan needs adjustment based on findings, output "adjust".
                  If coverage is comprehensive with sufficient evidence, output "complete".
                output_format: json
                output_schema:
                  type: object
                  properties:
                    decision:
                      type: string
                      enum: [continue, adjust, complete]
                    reasoning:
                      type: string
                    coverage_pct:
                      type: number
                    gaps:
                      type: array
                      items: {type: string}
                  required: [decision, reasoning]

    # ── Phase 3: BestOfN Synthesis ──
    - id: best_of_3_synthesis
      type: subworkflow
      label: "Best-of-3 Synthesis"
      config:
        ref: "builtin:best_of_n"
        params:
          n: 3
          diverse_tiers: true
          candidate_role: "Research Synthesizer"
          candidate_pool_tools: [observations, sources, web_findings, enterprise_findings]
          candidate_input_keys: [query, plan]
          judge_role: "Synthesis Quality Judge"
          judge_model_tier: complex
          judge_pool_tools: [sources]
          judge_system_prompt: |
            You are judging 3 synthesis candidates for a multi-source research report.
            Evaluate each on: factual accuracy, source coverage (web + enterprise),
            coherent narrative, proper attribution, and completeness vs the plan.
            Select the best candidate and explain your choice.
        input_mapping: {query: "query", plan: "plan"}
        output_key: best_report

    # ── Phase 4: Verification ──
    - id: verify
      type: tool
      label: "Verify and Ground Report"
      config:
        ref: {type: builtin, name: verify_and_ground}
        input_mapping: {text: "best_report", query: "query"}
        output_key: report
```

#### How it works

1. **Planner** creates a structured plan identifying web vs enterprise aspects.
2. **Research Loop** (2–7 iterations) runs 3 parallel researchers per iteration:
   web, Knowledge Assistant, and Vector Search. Each writes to `observations`
   (shared) plus topic-specific pools (`web_findings`, `enterprise_findings`).
   Sources auto-accumulate via Path B (§5.7).
3. **Reflector** reads ALL pools to assess coverage. Outputs JSON with
   `decision`, `reasoning`, `coverage_pct`, and `gaps`. Loop exits on
   `"complete"`.
4. **Best-of-3 Synthesis** generates 3 diverse reports (cycling through
   simple/analytical/complex tiers). Each candidate reads from all 4 content
   pools via `candidate_pool_tools`. A complex-tier judge evaluates and picks
   the best report.
5. **Verification** runs the citation pipeline (§11) on the winning report as a
   standalone tool node, producing the final grounded `report`.

#### Where convenience patterns add value

Without `builtin:best_of_n`, the user would need to manually build:
```yaml
sequence:
  parallel:
    agent: Candidate 1 (pool_tools: [observations, sources, ...])
    agent: Candidate 2 (pool_tools: [observations, sources, ...])
    agent: Candidate 3 (pool_tools: [observations, sources, ...])
  agent: Judge (input_keys: [candidate_1, candidate_2, candidate_3])
```

That's ~60 lines of YAML vs ~8 lines with the template. The `candidate_pool_tools`
param (§12.4) propagates pool access to all generated candidate agents
automatically.

---

### A.2 Meeting Prep Bot

**Scenario**: A sales meeting prep bot that prepares meeting
briefings. 6 research phases, conditional execution, parallel web research,
enterprise CRM integration, debate for competitive positioning, and structured
MEDDPICC output with citation verification.

#### Phase dependency diagram

```
                    ┌─────────────────┐
                    │ Query Extractor │
                    │ (company, names,│
                    │  competitor)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  CRM Researcher  │
                    │  (sfdc_genie)    │
                    └────────┬────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
      ┌────────▼───────┐ ┌──▼──────────┐ ┌▼────────────────┐
      │ Company Intel  │ │  Industry   │ │ Attendee        │
      │ (self_critique)│ │  Trends     │ │ Research        │
      └────────┬───────┘ └──┬──────────┘ │ (if attendees)  │
               │            │            └──┬──────────────┘
               └────────────┼───────────────┘
                            │
                   ┌────────▼────────┐
                   │   Customer      │
                   │   Competitive   │
                   │   Research      │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   Competitive   │
                   │   Positioning   │
                   │   Debate        │
                   │ (if competitor) │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   Meeting Prep  │
                   │   Synthesizer   │
                   │   (MEDDPICC +   │
                   │    verification)│
                   └─────────────────┘
```

#### Architecture tree

```
sequence
├── agent: "Query Extractor" (JSON: company, attendees[], competitor, deal_stage)
├── agent: "CRM Researcher" (enterprise:sfdc_genie → crm_context pool)
├── parallel: "Web Research Block"
│   ├── subworkflow: "Company Intel" (builtin:self_critique → company_intel pool)
│   ├── agent: "Industry Trends Researcher" (web → industry_trends pool)
│   └── conditional: has_attendees?
│       └── agent: "Attendee Researcher" (web → attendee_findings pool)
├── agent: "Customer Competitive Researcher" (web → competitive_intel pool)
├── conditional: has_competitor?
│   └── subworkflow: "Competitive Positioning Debate" (builtin:debate → vendor_positioning pool)
├── agent: "Meeting Prep Synthesizer"
│       pool_tools: [all 8 pools]
│       output_format: json (MeetingPrepOutput schema)
│       verification: enabled
```

#### Features demonstrated

| Feature | Where used | Section |
|---------|-----------|---------|
| Query extraction → conditional branching | Extractor JSON → dot-path conditions | §6.1, §6.5 |
| Enterprise-only CRM phase | sfdc_genie tool | §8.1 |
| Parallel web research (3-way) | Company Intel + Industry + conditional Attendee | §2, §13.2 |
| SelfCritique for thorough research | builtin:self_critique for company intel | §12.4 |
| Conditional gate pattern (1 child) | has_attendees?, has_competitor? | §4.4 |
| Debate for competitive positioning | builtin:debate with domain-specific advocates | §12.4 |
| SubworkflowNodeConfig.pool_writes | self_critique → company_intel pool | §4.5 |
| 8 topic-specific pools | Separate pools per research phase | §5.7 |
| Complex structured JSON output | MeetingPrepOutput with MEDDPICC | §7.4 |
| Per-pool safe template | Conditional sections for optional pools | §7.2, §7.5 |
| Citation verification | 5-stage pipeline on final output | §11 |

#### Full WorkflowDefinition

```yaml
id: "sa-presales-meeting-prep"
name: "SA Pre-Sales Meeting Prep Bot"
description: >
  Prepares comprehensive meeting briefings for Solutions Architects.
  Extracts context from the query, pulls CRM data, researches company/industry/
  attendees in parallel, optionally runs competitive positioning debate, and
  synthesizes a structured MEDDPICC-aligned meeting prep document.
version: 1
required_inputs: [query]
output_keys: [meeting_prep]

pools:
  - name: observations
    item_type: text
  - name: sources
    item_type: source
    dedup_key: url
  - name: crm_context
    item_type: text
  - name: company_intel
    item_type: text
  - name: industry_trends
    item_type: text
  - name: attendee_findings
    item_type: text
  - name: competitive_intel
    item_type: text
  - name: vendor_positioning
    item_type: text

root:
  id: root
  type: sequence
  label: "SA Meeting Prep Pipeline"
  children:

    # ── Phase 1: Query Extraction ──
    - id: query_extractor
      type: agent
      label: "Query Extractor"
      config:
        role: "Meeting Context Extractor"
        model_tier: simple
        input_keys: [query]
        output_key: extraction
        output_format: json
        output_schema:
          type: object
          properties:
            company:
              type: string
              description: "Target company name"
            attendees:
              type: array
              items: {type: string}
              description: "List of attendee names/titles (empty if not mentioned)"
            competitor_mentioned:
              type: string
              description: "Competitor name if mentioned, null otherwise"
            deal_stage:
              type: string
              enum: [discovery, qualification, demo, proposal, negotiation, unknown]
            meeting_type:
              type: string
              enum: [initial, follow_up, technical_deep_dive, executive_briefing, unknown]
            key_topics:
              type: array
              items: {type: string}
          required: [company, attendees, deal_stage]
        system_prompt: |
          Extract structured meeting context from the user's query. Identify the
          target company, any named attendees or titles, any competitor mentioned,
          the deal stage, meeting type, and key topics of interest.
          If attendees are not mentioned, return an empty array.
          If no competitor is mentioned, return null for competitor_mentioned.

    # ── Phase 2: CRM Lookup ──
    - id: crm_researcher
      type: agent
      label: "CRM Researcher"
      config:
        role: "CRM Intelligence Researcher"
        model_tier: analytical
        input_keys: [query, extraction]
        output_key: crm_data
        tools:
          - {type: enterprise, name: sfdc_genie}
        pool_writes:
          - {pool: crm_context}
          - {pool: observations}
        system_prompt: |
          You are a CRM researcher. Using the Salesforce Genie tool, look up the
          target company "{{extraction.company}}". Retrieve:
          - Account overview (industry, size, revenue)
          - Opportunity history and current pipeline
          - Recent activity/touchpoints
          - Key contacts and their roles
          - Any previous technical evaluations or POCs
          - Deal stage context: {{extraction.deal_stage}}
          Summarize findings concisely with specific data points.

    # ── Phase 3: Parallel Web Research ──
    - id: web_research_block
      type: parallel
      label: "Web Research Block"
      children:

        # 3a: Company Intel via Self-Critique
        - id: company_intel_subworkflow
          type: subworkflow
          label: "Company Intel (Self-Critique)"
          config:
            ref: "builtin:self_critique"
            params:
              generator_role: "Company Intelligence Researcher"
              generator_model_tier: analytical
              generator_tools:
                - {type: builtin, name: web_search}
                - {type: builtin, name: web_crawl}
              generator_pool_tools: [observations]
              generator_pool_writes:
                - {pool: observations}
              generator_input_keys: [query, extraction]
              critic_role: "Company Research Quality Reviewer"
              critic_model_tier: complex
              max_iterations: 2
              min_iterations: 1
            input_mapping: {query: "query", extraction: "extraction"}
            output_key: company_intel_result
            pool_writes:
              - {pool: company_intel}

        # 3b: Industry Trends
        - id: industry_researcher
          type: agent
          label: "Industry Trends Researcher"
          config:
            role: "Industry Analyst"
            model_tier: analytical
            input_keys: [query, extraction]
            output_key: industry_data
            output_mode: replace
            tools:
              - {type: builtin, name: web_search}
              - {type: builtin, name: web_crawl}
            pool_writes:
              - {pool: observations}
              - {pool: industry_trends}
            system_prompt: |
              Research current industry trends relevant to {{extraction.company}}.
              Focus on: market dynamics, regulatory changes, digital transformation
              initiatives, competitive landscape, and technology adoption patterns
              in their industry. Prioritize recent developments (last 6 months).

        # 3c: Attendee Research (conditional — gate pattern)
        - id: attendee_gate
          type: conditional
          label: "Attendee Research Gate"
          config:
            conditions:
              - type: state
                key: "extraction.attendees"
                operator: not_empty
          children:
            - id: attendee_researcher
              type: agent
              label: "Attendee Researcher"
              config:
                role: "Attendee Background Researcher"
                model_tier: analytical
                input_keys: [query, extraction]
                output_key: attendee_data
                tools:
                  - {type: builtin, name: web_search}
                  - {type: builtin, name: web_crawl}
                pool_writes:
                  - {pool: observations}
                  - {pool: attendee_findings}
                system_prompt: |
                  Research the following meeting attendees:
                  {{#for name in attendees}}
                  - {{name}}
                  {{/for}}
                  For each person, find: current role/title, professional background,
                  recent publications or talks, areas of expertise, LinkedIn profile
                  summary. Focus on information relevant to a technical sales meeting.

    # ── Phase 4: Customer Competitive Research ──
    - id: competitive_researcher
      type: agent
      label: "Customer Competitive Researcher"
      config:
        role: "Competitive Intelligence Researcher"
        model_tier: analytical
        input_keys: [query, extraction, crm_data]
        output_key: competitive_data
        tools:
          - {type: builtin, name: web_search}
          - {type: builtin, name: web_crawl}
        pool_writes:
          - {pool: observations}
          - {pool: competitive_intel}
        pool_tools: [crm_context]
        system_prompt: |
          Research the competitive landscape for {{extraction.company}}.
          Using CRM context and web sources, identify:
          - Technologies they currently use (from job postings, tech blogs, case studies)
          - Recent vendor evaluations or RFPs
          - Pain points with current solutions (from reviews, forums, social media)
          - Potential objections or concerns based on their tech stack
          {{#if competitor_mentioned}}
          Pay special attention to {{competitor_mentioned}} — they are
          a known competitor in this deal.
          {{/if}}

    # ── Phase 5: Competitive Positioning Debate (conditional — gate pattern) ──
    - id: competitor_debate_gate
      type: conditional
      label: "Competitive Debate Gate"
      config:
        conditions:
          - type: state
            key: "extraction.competitor_mentioned"
            operator: not_empty
      children:
        - id: competitive_debate
          type: subworkflow
          label: "Competitive Positioning Debate"
          config:
            ref: "builtin:debate"
            params:
              advocate_a_role: "Our Solution Champion"
              advocate_b_role: "Competitor Advocate"
              judge_role: "Objective Positioning Analyst"
              rounds: 2
              min_rounds: 2
              advocate_pool_tools: [competitive_intel, crm_context]
              judge_pool_tools: [competitive_intel, crm_context, company_intel]
            input_mapping: {query: "query", extraction: "extraction"}
            output_key: debate_result
            pool_writes:
              - {pool: vendor_positioning}

    # ── Phase 6: Meeting Prep Synthesis ──
    - id: synthesizer
      type: agent
      label: "Meeting Prep Synthesizer"
      config:
        role: "SA Meeting Prep Synthesizer"
        model_tier: complex
        input_keys: [query, extraction]
        output_key: meeting_prep
        output_format: json
        output_schema:
          type: object
          properties:
            executive_summary:
              type: string
              description: "2-3 paragraph company overview and meeting context"
            company_overview:
              type: object
              properties:
                name: {type: string}
                industry: {type: string}
                size: {type: string}
                key_facts: {type: array, items: {type: string}}
            attendee_briefs:
              type: array
              items:
                type: object
                properties:
                  name: {type: string}
                  title: {type: string}
                  background: {type: string}
                  talking_points: {type: array, items: {type: string}}
            discovery_questions:
              type: object
              description: "MEDDPICC-aligned discovery questions"
              properties:
                metrics: {type: array, items: {type: string}}
                economic_buyer: {type: array, items: {type: string}}
                decision_criteria: {type: array, items: {type: string}}
                decision_process: {type: array, items: {type: string}}
                paper_process: {type: array, items: {type: string}}
                implications_of_pain: {type: array, items: {type: string}}
                champion: {type: array, items: {type: string}}
                competition: {type: array, items: {type: string}}
            competitive_positioning:
              type: object
              properties:
                our_strengths: {type: array, items: {type: string}}
                their_weaknesses: {type: array, items: {type: string}}
                competitive_landmines: {type: array, items: {type: string}}
                objection_handlers:
                  type: array
                  items:
                    type: object
                    properties:
                      objection: {type: string}
                      response: {type: string}
            industry_context:
              type: object
              properties:
                trends: {type: array, items: {type: string}}
                opportunities: {type: array, items: {type: string}}
            recommended_agenda:
              type: array
              items:
                type: object
                properties:
                  topic: {type: string}
                  duration_minutes: {type: integer}
                  talking_points: {type: array, items: {type: string}}
            risk_flags:
              type: array
              items: {type: string}
              description: "Potential risks or red flags to be aware of"
          required: [executive_summary, company_overview, discovery_questions]
        pool_tools:
          - observations
          - sources
          - crm_context
          - company_intel
          - industry_trends
          - attendee_findings
          - competitive_intel
          - vendor_positioning
        user_prompt_template: |
          Prepare a comprehensive SA meeting prep document for:
          Company: {{extraction.company}}
          Deal Stage: {{extraction.deal_stage}}
          Meeting Type: {{extraction.meeting_type}}

          ## CRM Intelligence
          {{#for item in crm_context}}
          {{item}}
          {{/for}}

          ## Company Intelligence
          {{#for item in company_intel}}
          {{item}}
          {{/for}}

          ## Industry Trends
          {{#for item in industry_trends}}
          {{item}}
          {{/for}}

          {{#if attendee_findings}}
          ## Attendee Backgrounds
          {{#for item in attendee_findings}}
          {{item}}
          {{/for}}
          {{/if}}

          ## Competitive Intelligence
          {{#for item in competitive_intel}}
          {{item}}
          {{/for}}

          {{#if vendor_positioning}}
          ## Competitive Positioning Analysis
          {{#for item in vendor_positioning}}
          {{item}}
          {{/for}}
          {{/if}}

          ## Sources ({{sources|length}} total)
          {{#for src in sources}}
          - [{{loop.index}}] {{src.url}} — {{src.title}}
          {{/for}}

          Generate a complete MeetingPrepOutput JSON document. Ensure all discovery
          questions are MEDDPICC-aligned and tailored to the specific company and
          deal stage. If attendee data is available, include personalized talking
          points. If competitive positioning data is available, include competitive
          landmines and objection handlers.
        verification:
          enabled: true
          stages:
            evidence_preselection: true
            confidence_classification: true
            isolated_verification: true
            citation_correction: true
            numeric_qa_verification: true
```

#### How it works

1. **Query Extractor** (simple tier, fast): Parses the user's natural language
   request into structured JSON — company name, attendee list, competitor
   mentioned, deal stage. This JSON drives all downstream conditions.

2. **CRM Researcher**: Uses the `sfdc_genie` enterprise tool to pull Salesforce
   data. Results go to `crm_context` and `observations` pools. No web search —
   enterprise-only phase.

3. **Parallel Web Research** (3-way):
   - **Company Intel** via `builtin:self_critique`: The generator researches the
     company using web search/crawl, writing to `observations`. The critic
     reviews for thoroughness. After completion, the subworkflow's
     `pool_writes` directive writes the final output to `company_intel` pool.
   - **Industry Trends**: Single ReAct agent researching market dynamics.
   - **Attendee Research**: Conditional gate — only executes if
     `extraction.attendees` is non-empty (§4.4 gate pattern, 1 condition +
     1 child).

4. **Customer Competitive Research**: Runs after the parallel block. Reads
   `crm_context` pool for CRM context, searches web for competitive
   intelligence. Writes to `competitive_intel` pool.

5. **Competitive Positioning Debate** (conditional gate): Only runs if
   `extraction.competitor_mentioned` is non-empty. When triggered, uses
   `builtin:debate` with 3 domain-specific roles:
   - **Our Solution Champion**: Argues for our platform's strengths, reading
     from `competitive_intel` and `crm_context` pools.
   - **Competitor Advocate**: Steelmans the competitor's position.
   - **Objective Analyst**: Judges the debate and produces balanced positioning.
   Output goes to `vendor_positioning` pool via `pool_writes`.

6. **Meeting Prep Synthesizer**: Accesses ALL 8 pools via `pool_tools`. Uses a
   safe template with conditional sections (attendee briefs only if attendees
   were researched, competitive positioning only if debate ran). Outputs
   structured JSON matching `MeetingPrepOutput` schema with MEDDPICC discovery
   questions, attendee briefs, competitive landmines, and a recommended agenda.

7. **Verification**: The synthesizer has `verification.enabled: true` with 5
   stages active. The citation pipeline grounds claims against accumulated
   sources, corrects citations, and verifies numeric claims.

#### Design decisions

**Single agent vs subworkflow per phase**: Simple phases (industry trends,
attendees, competitive research) use single ReAct agents — they need 2-3 tool
calls at most. Company Intel uses `builtin:self_critique` because thorough
company research benefits from a quality review pass. Competitive positioning
uses `builtin:debate` because balanced vendor positioning requires adversarial
reasoning.

**Conditional gate pattern (1 condition + 1 child)**: The `attendee_gate` and
`competitor_debate_gate` nodes use the gate pattern (§4.4). When `conditions`
has 1 entry and `children` has 1 entry, the conditional acts as a gate: execute
if condition matches, skip (no-op) otherwise. This avoids needing placeholder
no-op agents for the "else" branch.

**SubworkflowNodeConfig.pool_writes**: Both `company_intel_subworkflow` and
`competitive_debate` use `pool_writes` on the subworkflow node config (§4.5).
This is the CALLER directing the subworkflow's output into a named pool. The
subworkflow itself doesn't know about the pool — it just produces an output
via `output_key`, and the parent's `pool_writes` routes it to the correct pool
after completion.

**8 topic-specific pools**: Rather than a single `findings` pool, each research
phase has its own pool. This enables the synthesizer's safe template to render
phase-specific sections with conditional logic (`{{#if attendee_findings}}`).
The `observations` pool still aggregates everything for cross-phase awareness.

**Per-pool safe template**: The synthesizer uses `user_prompt_template` (§7.2)
with conditional sections (§7.5). Pools that may be empty (attendee_findings,
vendor_positioning) are wrapped in `{{#if}}` blocks. This handles the case
where conditional gates skipped those phases gracefully.

---

## 20. References

- [Agentic AI: Architectures, Taxonomies (arxiv 2601.12560)](https://arxiv.org/html/2601.12560v1)
- [Multi-Agent Collaboration Mechanisms Survey (arxiv 2501.06322)](https://arxiv.org/html/2501.06322v1)
- [Deep Search Agents Survey (arxiv 2508.05668)](https://arxiv.org/html/2508.05668v3)
- [Mixture-of-Agents — ICLR 2025 (arxiv 2406.04692)](https://arxiv.org/abs/2406.04692)
- [Beyond Majority Voting (arxiv 2510.01499)](https://arxiv.org/abs/2510.01499)
- [Debate or Vote (arxiv 2508.17536)](https://arxiv.org/abs/2508.17536)
- [SWE-Search MCTS (OpenReview)](https://openreview.net/forum?id=G7sIFXugTX)
- [LangGraph Multi-Agent Concepts](https://langchain-ai.github.io/langgraphjs/concepts/multi_agent/)
- [LangGraph Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)
- [CrewAI Hierarchical Process](https://docs.crewai.com/en/learn/hierarchical-process)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Google A2A Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Agent-as-Judge (arxiv 2508.02994)](https://arxiv.org/html/2508.02994v1)
- [Agentic Workflow Description Language](https://www.emergentmind.com/topics/agentic-workflow-description-language)
