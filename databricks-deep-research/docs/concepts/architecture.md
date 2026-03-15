# Architecture

> High-level design, execution model, and component layers of the framework.

## Design Principles
1. **Plain async Python** — No framework lock-in (no LangGraph, DSPy, AutoGen). Just asyncio + Pydantic.
2. **YAML-defined DAGs** — Workflows are declarative YAML trees, not Python code graphs.
3. **Step-by-step reflection** — After EACH research step, a reflector agent decides: CONTINUE, ADJUST, or COMPLETE.
4. **Tiered model routing** — simple (fast/cheap), analytical (balanced), complex (reasoning). Automatic fallback.
5. **Append-only state** — WorkflowState is a log. Reads are O(1), writes never mutate history.
6. **Streaming-first** — Every node emits typed events. UIs consume event streams, not final results.
7. **Tool security** — LLMs see URL indices, not raw URLs. UrlRegistry prevents hallucinated references.

## Component Diagram (ASCII)
```
┌─────────────────────────────────────────────────────┐
│                   WorkflowRunner                     │
│  (High-level API: run() / stream())                  │
├─────────────────────────────────────────────────────┤
│                  WorkflowExecutor                    │
│  (Tree walker: sequence, parallel, loop, conditional)│
├───────────┬───────────┬──────────────┬──────────────┤
│  Agent    │   Tool    │   Pool       │   Events     │
│  Harness  │  Resolver │  Registry    │   Stream     │
│  + ReAct  │  + URL    │  + BM25      │   (50+       │
│   Loop    │  Registry │  + Vector    │    types)    │
├───────────┴───────────┴──────────────┴──────────────┤
│               FrameworkLLMClient                     │
│  (Tier routing, health tracking, fallback, budget)   │
├─────────────────────────────────────────────────────┤
│               AsyncOpenAI Client                     │
│  (Databricks Model Serving / OpenAI)                 │
└─────────────────────────────────────────────────────┘
```

## Three-Layer Architecture

### Layer 1: Schema Layer (definition + validation)
- `WorkflowDefinition` — The blueprint (YAML-serializable)
- `WorkflowNode` — Recursive tree of typed nodes
- `AgentNodeConfig` — 25+ fields controlling agent behavior
- `ToolDeclaration` / `SourceDefinition` — Tool and source metadata
- `ErrorConfig` — Per-node error handling (fail/skip/retry)
- Validation: `validate_workflow()` checks references, cycles, required fields

### Layer 2: Execution Layer (state + orchestration)
- `WorkflowExecutor` — Walks the node tree depth-first
- `WorkflowState` — Append-only log with O(1) latest-value lookup
- `ExecutionContext` — Binds LLM client, state, tools, pools together
- `WorkflowRunner` — Convenience wrapper (load + execute + collect)

### Layer 3: Tool Layer (LLM + tools + pools)
- `FrameworkLLMClient` — Model tier routing with health/fallback
- `ResearchTool` protocol — Standard interface for all tools
- `ToolResolver` — Unified tool lookup (builtin + custom + enterprise)
- `PoolRegistry` — Shared memory pools with hybrid search
- `UrlRegistry` — Secure LLM-to-source mapping via integer indices

## Execution Flow

Walk through a typical `simple_research.yaml` execution:

```
1. LOAD        load_workflow("simple_research.yaml")
                → WorkflowDefinition with root sequence node

2. INIT        WorkflowState(query="...")
                ExecutionContext(llm, state, ...)
                WorkflowExecutor(workflow, context)

3. EXECUTE     executor.execute() → AsyncGenerator[FrameworkEvent]
                ├─ yield WorkflowStartedEvent
                └─ walk root node (sequence)

4. SEQUENCE    Execute children in order:
   ├─ 4a. COORDINATOR (agent node)
   │    execute_agent() → LLM call → CoordinatorOutput
   │    state.append("coordination", {...complexity, mode})
   │    yield CoordinatorClassifiedEvent
   │
   ├─ 4b. BACKGROUND (agent node)
   │    execute_agent() with web_search tool
   │    Pool writes: sources[], observations[]
   │    yield BackgroundCompletedEvent
   │
   ├─ 4c. PLAN_AND_EXECUTE (plan_and_execute node)
   │    ├─ PLANNER → PlanOutput with steps[]
   │    │   yield PlanCreatedEvent
   │    │
   │    ├─ For each step (loop):
   │    │   ├─ RESEARCHER → web_search + web_crawl
   │    │   │   yield ToolCallEvent, ToolResultEvent
   │    │   │   Pool writes: sources[], observations[]
   │    │   │
   │    │   └─ REFLECTOR → ReflectionOutput
   │    │       yield ReflectionDecisionEvent
   │    │       (CONTINUE → next step, ADJUST → modify plan, COMPLETE → exit)
   │    │
   │    └─ yield PlanAndExecuteExitEvent
   │
   └─ 4d. SYNTHESIZER (agent node)
        Pool inject: sources + observations → context
        execute_agent() → final markdown report
        state.append("synthesis", report)
        yield AgentOutputEvent

5. COMPLETE    yield WorkflowCompletedEvent(total_tokens=...)
```

## Agent Execution Cycle
```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Build    │────▶│ LLM Call │────▶│ Parse    │────▶│ State    │
│ Prompt   │     │ (tier)   │     │ Output   │     │ Write    │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                                                    │
     │              ┌──────────┐                          │
     └──────────────│ Pool     │◀─────────────────────────┘
                    │ Inject   │
                    └──────────┘
```

For ReAct agents (researcher), the cycle includes tool calls:
```
LLM Call → Tool Calls? → Execute Tools → Append Results → LLM Call → ...
(loop until LLM stops calling tools or budget exceeded)
```

## See Also
- [Workflow Engine](workflow-engine.md) — Deep dive into WorkflowDefinition
- [Agent System](agent-system.md) — Agent harness and ReAct loop
- [State Management](state-management.md) — Append-only log design
- [Events](events.md) — Streaming event system
