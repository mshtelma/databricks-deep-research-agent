# Feature Specification: Multi-Agent Framework Extraction

**Feature Branch**: `011-multi-agent-framework`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "Extract an abstract agentic framework from this project, making it consist of two modules: (1) an abstract multi-agent framework usable outside this project with YAML config support, and (2) a UI project including FastAPI backend and existing frontend."

## Clarifications

### Session 2026-03-08

- Q: Should the old orchestrator be replaced entirely or coexist with the framework during transition? → A: Replace entirely — no parallel coexistence. The old orchestrator is fully replaced by the framework in one step.
- Q: Should domain-specific builtin tools (web_search, web_crawl, verify_and_ground) live in the framework or the app? → A: All builtins ship with the framework. Web search, web crawling, and verification/grounding are general-purpose agent capabilities that most agents need, not Deep Research-specific.
- Q: Should existing agent implementations (run_react_researcher, run_reflector, etc.) live in the framework or the app? → A: Move into the framework as builtin subtype implementations. The framework is fully batteries-included — production-quality agents ship out of the box.
- Q: How should the framework and app be structured? → A: Monorepo with two subdirectory projects: `databricks-deep-research/` (framework) and `databricks-deep-research-app/` (application). Root `pyproject.toml` with `uv` workspace ties them together for local development. The framework MUST be independently publishable to PyPI. The app declares the framework as a normal dependency and provides its own YAML workflow definition for the deep research pipeline.
- Q: What should the framework package be named? → A: `databricks-deep-research` (import: `databricks_deep_research`).

### Session 2026-03-08 (Planning Decisions)

- D1: Full replacement — no feature flag, no gradual migration. Project is not released.
- D2: Full port — all production agents move to framework builtins. No simplified vs production split.
- D3: AsyncOpenAI directly — no LLM Protocol abstraction. Framework wraps `openai.AsyncOpenAI` via `FrameworkLLMClient`. Constitution Principle I deviation acknowledged.
- D4: Coordinator is a framework builtin — 5th agent subtype (coordinator, researcher, planner, reflector, synthesizer). Essential routing pattern.
- D5: YAML is first-class — `from_yaml()` / `to_yaml()` are full framework features, not demoted.
- D6: Drop classic researcher mode — only ReAct mode ported. Classic is dead code.
- D7: Drop simple synthesizer mode — only ReAct + Reclaim modes. Simple is dead code.
- D8: Prompt customization — `system_prompt` + `user_prompt_template` overrides in YAML (main pair only). Internal prompts stay internal.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Framework Developer Defines and Executes a Workflow via YAML (Priority: P1)

A developer building a new agentic application (outside of the Deep Research project) installs the multi-agent framework as a standalone Python package. They define a workflow tree in YAML using the 8 primitive node types (agent, tool, sequence, parallel, loop, conditional, subworkflow, plan_and_execute), configure agent subtypes, shared pools, and conditions, then execute the workflow programmatically. The framework ships with builtin tools for web search, web crawling, and verification/grounding, so common agent patterns work out of the box. The framework handles orchestration, state management, tool resolution, and streaming events without requiring the Deep Research UI, database, or FastAPI layer.

**Why this priority**: This is the core value proposition — the framework must be independently usable. Without this, the extraction has no purpose.

**Independent Test**: Can be fully tested by installing the framework package, writing a YAML workflow definition, and executing it via the Python API to get a streaming result — delivers a reusable orchestration engine.

**Acceptance Scenarios**:

1. **Given** a developer has installed the framework package, **When** they load a YAML workflow definition with a sequence of agent nodes and a loop with a reflector, **Then** the framework validates the tree structure, resolves agent subtypes, and returns a `WorkflowDefinition` object ready for execution.
2. **Given** a valid workflow definition and executor configured with an LLM client, **When** they execute the workflow, **Then** the executor walks the tree, yields streaming events (node started, agent output, loop iteration, node completed, etc.), and populates the workflow state with results.
3. **Given** a YAML workflow referencing a researcher subtype with pool writes to an "observations" pool, **When** the researcher agent executes, **Then** findings are written to both the state log and the declared pool, and downstream agents can search the pool via auto-generated pool tools.
4. **Given** a YAML workflow with an invalid structure (e.g., a loop node with two children, or parallel children sharing the same output key), **When** the definition is loaded, **Then** the framework raises a validation error with a clear message before execution begins.

---

### User Story 2 - Deep Research App Uses the Framework as a Dependency (Priority: P1)

The existing Deep Research application (FastAPI backend + React frontend) is refactored to import and use the multi-agent framework as an internal dependency. The app's orchestrator is fully replaced by the framework's executor — there is no parallel coexistence or feature flag. All existing functionality — streaming SSE, chat persistence, custom agent configs, citation verification, enterprise tools — continues to work identically. Users of the Deep Research app notice no behavioral change.

**Why this priority**: The extraction must not break the existing product. This is equally critical because the framework is only valuable if it replaces the current orchestration without regression.

**Independent Test**: Can be fully tested by running the existing E2E test suite against the refactored codebase — all tests pass with identical behavior.

**Acceptance Scenarios**:

1. **Given** the Deep Research app is refactored to use the framework, **When** a user submits a research query via the chat UI, **Then** the app loads its own YAML workflow definition (describing the deep research pipeline), configures it with the orchestration config, executes via the framework's executor, and streams events to the frontend identically to the current behavior.
2. **Given** a custom agent configuration with preset steps (Level 1 abstraction), **When** the app generates a workflow tree from the steps, **Then** the generated tree matches the existing pipeline behavior (planner, plan-and-execute cycle, synthesizer) and produces equivalent research output.
3. **Given** enterprise tools are configured for a user, **When** the workflow executes, **Then** enterprise tools are resolved and available to agents, with OBO token flow preserved.

---

### User Story 3 - Developer Composes Advanced Patterns via Templates (Priority: P2)

A developer uses parameterized workflow templates (BestOfN, SelfCritique, Debate, MajorityVote) as builtin subworkflow references in their YAML definition. They customize template parameters (number of candidates, model tiers, iteration limits) without manually building the underlying tree structure. The templates generate valid workflow trees composed of the 8 primitive node types.

**Why this priority**: Templates are the primary ergonomic feature that makes the framework accessible to non-expert users. Without them, every advanced pattern requires manual tree construction.

**Independent Test**: Can be fully tested by defining a workflow that references a BestOfN template with custom params and executing it — delivers quality-improving patterns out of the box.

**Acceptance Scenarios**:

1. **Given** a workflow with a BestOfN template reference and params specifying 3 diverse-tier candidates, **When** the subworkflow node is resolved, **Then** the template generator produces a tree with a parallel node containing 3 candidate agents and a judge agent, each with the correct model tier and pool configurations.
2. **Given** a workflow with a SelfCritique template reference and max 3 iterations, **When** executed, **Then** the generated loop runs up to 3 iterations of generator + critic, exiting when the critic approves.
3. **Given** a developer provides invalid template params (e.g., zero candidates for BestOfN), **When** the template is resolved, **Then** a validation error is raised with a clear message about the invalid parameter.

---

### User Story 4 - Developer Creates Reusable Subworkflows with State Isolation (Priority: P2)

A developer defines reusable workflow fragments (e.g., a critique cycle, a section research pipeline) as standalone workflow definitions with declared required inputs and output keys. They reference these subworkflows from parent workflows via input/output mappings, with configurable pool isolation (shared or isolated).

**Why this priority**: Subworkflows enable compositional design — the ability to build complex workflows from tested, reusable pieces. This is essential for production use but can follow the core execution engine.

**Independent Test**: Can be fully tested by defining a parent workflow that delegates to a child subworkflow and verifying that state isolation, input/output mapping, and pool sharing work correctly.

**Acceptance Scenarios**:

1. **Given** a parent workflow referencing a subworkflow with explicit input and output mappings, **When** the subworkflow executes, **Then** only the mapped keys cross the boundary — the child cannot read or write arbitrary parent state.
2. **Given** a subworkflow with isolated pool mode, **When** it executes, **Then** it operates with its own fresh pools and the parent's pools are not accessible. Only output-mapped keys return to the parent.
3. **Given** a subworkflow with shared pool mode (default), **When** it executes, **Then** it reads from and writes to the parent's pools by reference, enabling cross-agent accumulation.

---

### User Story 5 - Developer Extends the Framework with Custom Tools (Priority: P3)

A developer registers custom tools (Unity Catalog functions, enterprise data sources, or custom Python functions) that agents can use during workflow execution. Tools are referenced in YAML configurations and resolved at workflow startup. The framework provides a tool protocol that any custom tool must implement.

**Why this priority**: Tool extensibility is what makes the framework useful for diverse domains beyond web research. However, the core framework works with built-in tools first.

**Independent Test**: Can be fully tested by implementing a custom tool that conforms to the tool protocol, referencing it in a workflow YAML, and verifying the agent can call it during execution.

**Acceptance Scenarios**:

1. **Given** a developer implements a custom tool conforming to the framework's tool protocol, **When** they register it and reference it in a workflow YAML, **Then** agents can discover and call the tool during their execution loops.
2. **Given** a YAML workflow referencing a Unity Catalog function tool, **When** the workflow starts, **Then** the framework resolves the tool via UC metadata, translates parameters to a standard schema, and makes it available to the configured agent.

---

### Edge Cases

- What happens when a loop's exit condition never evaluates to true? The `max_iterations` safety limit terminates the loop and emits an exit event with reason "max_iterations".
- How does the system handle an agent that produces invalid JSON when structured output is configured? A configurable parse failure policy controls behavior: retry with corrective prompt (up to a configured retry limit), then stop or continue based on configuration.
- What happens when parallel children both write to the same pool? Pool writes use async locks — appends are serialized at await points. Order is non-deterministic but accumulation is correct. No data races.
- What happens when a referenced subworkflow does not exist? Load-time validation catches missing refs and raises a clear error before execution begins.
- How does the framework handle cancellation mid-execution? A cancellation flag is checked before each node starts, before each loop iteration, and before each parallel child. Running LLM calls complete naturally; their results are discarded.
- What happens when a YAML workflow definition has circular subworkflow references? Dynamic refs prohibit subworkflow nodes in generated trees. Static subworkflow refs are resolved from a registry — circular refs result in a load-time error.
- What happens when token budget is exceeded mid-workflow? The budget-aware LLM client wrapper raises an error, the executor emits a budget exceeded event, and the workflow terminates gracefully.

## Requirements *(mandatory)*

### Functional Requirements

#### Module Separation

- **FR-001**: The project MUST be split into two subdirectory projects in a monorepo: (a) `databricks-deep-research/` — a standalone multi-agent framework package, and (b) `databricks-deep-research-app/` — the Deep Research application (FastAPI backend + React frontend) that depends on the framework. Each project has its own `pyproject.toml`, dependencies, and test suite. A root `pyproject.toml` with `uv` workspace ties them together for local development.
- **FR-002**: The framework (`databricks-deep-research`, import: `databricks_deep_research`) MUST be independently publishable to PyPI as a standalone Python package with no dependencies on the Deep Research application code, database, or UI. The framework includes its own builtin tools (web search, web crawling, verification/grounding, transform functions) as general-purpose agent capabilities. The framework depends on `openai` (AsyncOpenAI) directly for LLM access — Databricks standardizes on OpenAI-compatible endpoints.
- **FR-003**: The Deep Research application MUST declare the framework as a normal dependency and delegate workflow orchestration to it. The app MUST define its own YAML workflow definition(s) that describe the deep research pipeline using the framework's primitives. The old orchestrator MUST be fully replaced — no parallel coexistence or feature-flag switching between old and new paths.

#### Workflow Definition & YAML Config

- **FR-004**: The framework MUST support defining workflows as trees of nodes, loadable from YAML configuration files, where each node is one of exactly 8 types: agent, tool, sequence, parallel, loop, conditional, subworkflow, plan_and_execute.
- **FR-005**: The framework MUST validate workflow definitions at load time, checking structural rules (unique IDs, correct children counts per type, non-overlapping parallel output keys, etc.) and reporting errors before execution begins.
- **FR-006**: The framework MUST support a declarative condition system (state conditions, LLM conditions, composite boolean conditions) for use in loop exit conditions and conditional branch selection.
- **FR-007**: The framework MUST support dot-path resolution for nested state access in conditions and input keys (e.g., accessing a sub-field of a structured agent output).

#### State Management

- **FR-008**: The framework MUST implement append-only immutable state where all inter-node communication flows through state entries with node ID, key, value, and timestamp.
- **FR-009**: The framework MUST provide two read patterns: get-latest for replace semantics and get-all for accumulated list semantics.
- **FR-010**: The framework MUST implement shared research pools with typed items, deduplication (by configurable key and by content hash), max capacity with oldest-eviction, and async-safe multi-producer writes.
- **FR-011**: Agents MUST access pools via auto-generated search/retrieval tools (not via prompt injection). The framework MUST generate search, get-recent, count, topics, and get-by-index tools for each pool an agent declares.

#### Execution Engine

- **FR-012**: The framework MUST implement an executor that walks the workflow tree, executing each node type according to its semantics (sequence: serial, parallel: concurrent, loop: do-while with min/max bounds, conditional: if/elif/else with optional default branch and gate pattern).
- **FR-013**: The executor MUST yield streaming events for all significant execution points: node start/complete/error, loop iterations, branch selections, tool calls, agent outputs, checkpoint saves, gate interactions.
- **FR-014**: The framework MUST enforce agent isolation — agents never see the full workflow state directly; the executor constructs an immutable input snapshot and receives a structured output.
- **FR-015**: The framework MUST support two-phase conversation compaction (observation masking first, then optional summarization) to manage context window pressure during agent execution loops.

#### Agent Subtypes & Templates

- **FR-016**: The framework MUST provide standard agent subtypes (coordinator, researcher, synthesizer, planner, reflector, background) with sensible defaults for model tier, output key, tools, pool writes, pool tools, output format, and output schema — all overridable per-field. Each subtype MUST include a production-quality builtin implementation (e.g., ReAct researcher with token budgeting, citation-aware synthesizer, step-by-step reflector, query coordinator, background investigator) that ships with the framework. Users may override the main prompt pair (`system_prompt`, `user_prompt_template`) via YAML; internal prompts stay internal.
- **FR-017**: *(Deferred to P2)* The framework MUST support parameterized workflow templates (BestOfN, SelfCritique, Debate, MajorityVote) that generate valid workflow trees from parameters via a template registry.
- **FR-018**: *(Deferred to P2)* The framework MUST support subworkflows with data isolation (state log always isolated, pools configurable as shared or isolated) and explicit input/output key mapping.

#### Tool System

- **FR-019**: The framework MUST define a tool protocol and a tool reference model for referencing tools by type (builtin, UC function, UC tool, enterprise) and name. All builtin tools — web search, web crawl, file search, verification/grounding, evidence pre-selection, section merging, and transform functions (majority_vote, concatenate, pick_longest, json_merge) — MUST ship with the framework.
- **FR-020**: The framework MUST resolve all tool references at workflow startup and cache them for the workflow's lifetime.
- **FR-021**: *(Deferred)* The framework MUST support tool call deduplication via a cache to prevent redundant work across loop iterations and retries.

#### Reliability & Observability

- **FR-022**: The framework MUST support per-node error handling configuration (fail, skip, retry with configurable max retries and backoff).
- **FR-023**: *(Protocol in P0, full auto-checkpoint deferred)* The framework MUST support checkpointing workflow state after leaf node completion and resuming execution from checkpoints.
- **FR-024**: The framework MUST support token budget enforcement that tracks cumulative token usage and stops execution when limits are exceeded.
- **FR-025**: *(Deferred to P2)* The framework MUST support human-in-the-loop gates on any workflow node, with configurable trigger timing (before/after), timeout, and auto-proceed behavior.
- **FR-026**: *(Deferred)* The framework MUST integrate with MLflow for tracing when available, creating spans that mirror the workflow tree structure.

#### Template Security

- **FR-027**: User-provided prompt templates MUST use a safe template renderer (not a general-purpose template engine), supporting only variable substitution, conditional blocks, iteration, and length filtering — no expression evaluation, no attribute traversal, no method calls.

#### Backward Compatibility

- **FR-028**: The Deep Research application MUST support three levels of user abstraction: Level 1 (preset steps auto-generate tree), Level 2 (template selection with parameters), Level 3 (full tree definition in YAML/JSON).
- **FR-029**: Existing orchestration config fields MUST map to equivalent workflow tree structures and execution context, preserving current behavior for all existing research modes.
- **FR-030**: *(Deferred to P2)* The framework MUST support static data flow analysis at load time to validate data dependencies, detect missing producers, and validate dot-path references against output schemas.

### Key Entities

- **WorkflowDefinition**: A complete, self-contained workflow specification — tree of nodes, pool declarations, checkpoint config, token budget, required inputs/outputs.
- **WorkflowNode**: A single node in the workflow tree — has an ID, type (one of 8), label, type-specific config, optional children, and optional gate.
- **WorkflowState**: The runtime state flowing through the tree — append-only log of state entries, shared pools, execution context, iteration counters.
- **StateEntry**: An immutable record of a single state write — node ID, key, value, timestamp.
- **PoolState**: A shared multi-producer accumulation point — typed items, dedup, max capacity, async lock.
- **WorkflowExecutor**: The engine that walks a workflow tree, executes nodes, and yields streaming events.
- **AgentInput / AgentOutput**: The isolation boundary between the executor and agents — immutable input snapshot in, structured output out.
- **ToolRef / ResearchTool**: The tool referencing and execution abstraction — type + name reference resolved to executable tool instances.
- **StreamEvent**: Base type for all execution events — node started, node completed, loop iteration, tool call, agent output, etc.
- **DataFlowGraph**: Static analysis artifact computed at load time — validates data dependencies, detects missing producers, checks dot-path references against schemas.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The framework package can be installed and used in a fresh Python project (no Deep Research code) to define and execute a multi-agent workflow from YAML, producing streaming results — verified by a standalone integration test.
- **SC-002**: All existing Deep Research E2E tests pass after the refactoring, with no user-visible behavioral changes.
- **SC-003**: A workflow defined in YAML with 6 agent subtypes, a research loop, parallel execution, and verification produces equivalent research output to the current hardcoded orchestrator.
- **SC-004**: Invalid workflow definitions (structural errors, missing producers, type mismatches) are caught at load time with clear error messages — verified by unit tests covering all validation rules in the validation matrix.
- **SC-005**: The framework's public API surface (types, executor, state, events) is fully type-annotated and usable without reading framework internals.
- **SC-006**: *(Deferred to P2)* Parameterized templates (BestOfN, SelfCritique, Debate, MajorityVote) each generate valid, executable workflow trees — verified by unit tests.
- **SC-007**: *(Deferred to P2)* Subworkflow state isolation works correctly — child workflows cannot access parent state except through declared mappings, and pool sharing respects the configured isolation mode — verified by unit tests.
- **SC-008**: Token budget enforcement stops execution when limits are exceeded — verified by a unit test with a budget-constrained workflow.

## Assumptions

- The framework and app are two separate projects in the same repository directory, each with its own package definition. The framework is independently publishable to PyPI from day one.
- The framework will use `asyncio` for concurrency (cooperative multitasking on one thread), matching the existing codebase pattern.
- The framework wraps `openai.AsyncOpenAI` directly via `FrameworkLLMClient` (no Protocol abstraction). The Deep Research app adapts its existing `LLMClient` (health tracking, fallback, OAuth) via `llm_adapter.py` to provide the `AsyncOpenAI` instance + model mapping. Constitution Principle I deviation acknowledged — justified because Databricks standardizes on OpenAI-compatible endpoints.
- The existing specialized agent functions (researcher, reflector, synthesizer, etc.) will be moved into the framework as builtin subtype implementations, maintaining current quality and sophistication.
- The safe template renderer already exists in the codebase and will be extracted into the framework package.
- MLflow tracing integration is optional — the framework works without MLflow installed but creates spans when available.

## Dependencies

- The existing Deep Research codebase provides production-tested implementations of agent nodes, tools, search, citation verification, and streaming that inform the framework's design.
- The architecture document at `specs/multi-agent-framework/architecture.md` provides the detailed technical design.
- Pydantic 2.x for data models and validation.
- `asyncio` for async execution.
- `pyyaml` for YAML config loading.

## Out of Scope

- Visual workflow editor UI (Level 3 tree editor) — the framework provides the execution engine; visual editing UI is a separate concern.
- Dynamic workflow generation (LLM-generated trees at runtime) — deferred to a later phase.
- Plugin-defined templates and custom subtype registration — deferred to a later phase.
- Database persistence of workflow definitions — the framework loads definitions from YAML/Python objects; the app layer handles DB storage.
- Frontend changes — the React frontend continues to work via the existing SSE streaming API; no frontend modifications are needed for this extraction.
