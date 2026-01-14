# 5-Agent Orchestration Design

## Overview

The Deep Research Agent uses a 5-agent architecture with step-by-step reflection. Each agent has a specific responsibility, and the Reflector makes decisions after each research step to determine whether to continue, adjust the plan, or complete.

## Agent Summary

| Agent | Purpose | Model Tier | Key Output |
|-------|---------|------------|------------|
| **Coordinator** | Query classification, complexity estimation | Simple | `QueryClassification` |
| **Background** | Quick context gathering before planning | Simple | `background_investigation_results` |
| **Planner** | Create structured research plan with N steps | Analytical | `Plan` with `PlanStep[]` |
| **Researcher** | Execute single step: web search + crawl | Analytical | `last_observation`, `sources` |
| **Reflector** | Decide: CONTINUE / ADJUST / COMPLETE | Simple | `ReflectionResult` |
| **Synthesizer** | Generate final report with citations | Complex | `final_report`, `claims` |

## Orchestration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. COORDINATOR   │  ← Query + conversation history                        │
│  │ Query classify   │  → complexity (simple/moderate/complex)                │
│  │ Simple detection │  → is_simple_query flag                                │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│     ┌─────┴─────┐                                                            │
│     │ Simple?   │                                                            │
│     └─────┬─────┘                                                            │
│       YES │ NO                                                               │
│        ↓  ↓                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 2. BACKGROUND   │  ← Original query                                       │
│  │ Quick context   │  → 2-3 search queries via LLM                           │
│  │ web search      │  → background_investigation_results                     │
│  └────────┬────────┘                                                         │
│           ↓                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. RESEARCH LOOP (max_plan_iterations = 3)                             │ │
│  │  ┌────────────────┐                                                    │ │
│  │  │ 3a. PLANNER    │  ← query + observations + background + feedback    │ │
│  │  │ Create N steps │  → Plan with PlanStep[] (preserves completed)      │ │
│  │  └───────┬────────┘                                                    │ │
│  │          ↓                                                             │ │
│  │  ┌────────────────────────────────────────────────────────────────┐    │ │
│  │  │ STEP LOOP (while has_more_steps):                              │    │ │
│  │  │  ┌─────────────────┐                                           │    │ │
│  │  │  │ 3b. RESEARCHER  │ ← current step                            │    │ │
│  │  │  │ web_search()    │ → search results                          │    │ │
│  │  │  │ web_crawl()     │ → page content, sources                   │    │ │
│  │  │  │ observation     │ → last_observation                        │    │ │
│  │  │  └───────┬─────────┘                                           │    │ │
│  │  │          ↓                                                     │    │ │
│  │  │  ┌─────────────────┐                                           │    │ │
│  │  │  │ 3c. REFLECTOR   │ ← plan + observations + sources           │    │ │
│  │  │  │ Step decision   │ → ReflectionDecision                      │    │ │
│  │  │  └───────┬─────────┘                                           │    │ │
│  │  │          │                                                     │    │ │
│  │  │    ┌─────┴─────────────────────────────┐                       │    │ │
│  │  │    │            │                      │                       │    │ │
│  │  │ CONTINUE     ADJUST               COMPLETE                     │    │ │
│  │  │    ↓            ↓                      ↓                       │    │ │
│  │  │ next step  → back to 3a           skip remaining               │    │ │
│  │  │ (advance)   (replan with           → go to 4                   │    │ │
│  │  │              completed preserved)                              │    │ │
│  │  └────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│           ↓                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. SYNTHESIZER   │  ← query + all_observations + sources                  │
│  │ Report generation│  → final_report                                        │
│  │ Citation pipeline│  → claims + citations + verdicts                       │
│  └────────┬─────────┘                                                        │
│           ↓                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 5. PERSISTENCE   │  ← state with all data                                 │
│  │ Atomic write     │  → Chat, Message, Sources, Claims persisted            │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent Details

### 1. Coordinator Agent

**File**: `src/agent/nodes/coordinator.py`

**Purpose**: Classify query complexity and detect simple queries that can be answered directly.

**Input**:
- User query
- Conversation history (last 5 messages)

**Output** (`QueryClassification`):
```python
class QueryClassification:
    complexity: str              # "simple", "moderate", "complex"
    follow_up_type: str          # "new_topic", "clarification", "complex_follow_up"
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    recommended_depth: str = "auto"
    reasoning: str
```

**Decision Logic**:
- If `is_simple_query = True` → Skip research, use direct response
- Otherwise → Sets `recommended_depth` for planning phase

**Model Tier**: Simple (Gemini)

### 2. Background Investigator Agent

**File**: `src/agent/nodes/background.py`

**Purpose**: Quick context gathering before detailed planning.

**Input**:
- Original query
- Background config (max queries, etc.)

**Process**:
1. Generates 2-3 focused search queries using LLM
2. Performs web searches via Brave API
3. Formats results into `state.background_investigation_results`
4. Does NOT crawl full page content

**Output**: `background_investigation_results` (string)

**Model Tier**: Simple

### 3. Planner Agent

**File**: `src/agent/nodes/planner.py`

**Purpose**: Create structured research plan with N steps.

**Input**:
```
- Current query
- Previous observations (from completed steps)
- Background investigation results
- Reflector feedback (if ADJUST decision)
- Completed steps (preserved during replanning)
```

**Output** (`Plan`):
```python
@dataclass
class Plan:
    id: str                          # Plan UUID
    title: str                       # Plan summary
    thought: str                     # Reasoning
    steps: list[PlanStep]            # N research steps
    has_enough_context: bool = False # Can skip to synthesis
    iteration: int = 1               # Plan iteration number

@dataclass
class PlanStep:
    id: str
    title: str                       # "Research {topic}"
    description: str                 # Details for researcher
    step_type: StepType              # RESEARCH or ANALYSIS
    needs_search: bool               # Whether to web search
    status: StepStatus = PENDING     # PENDING/IN_PROGRESS/COMPLETED/SKIPPED
    observation: str | None = None   # Filled by researcher
```

**Key Feature - Completed Step Preservation**:
When ADJUST decision occurs:
1. Planner queries `state.get_completed_steps()`
2. Only NEW steps are generated by LLM
3. Final plan = completed_steps + new_steps
4. `current_step_index` resumes from first non-completed step

**Step Limits** (per depth):
| Depth | Min Steps | Max Steps |
|-------|-----------|-----------|
| Light | 1 | 3 |
| Medium | 3 | 6 |
| Extended | 5 | 10 |

**Model Tier**: Analytical (Claude)

### 4. Researcher Agent

**File**: `src/agent/nodes/researcher.py`

**Purpose**: Execute a single research step with web search and content crawling.

**Two Modes**:

| Mode | Selection | Description |
|------|-----------|-------------|
| **Classic** | Light depth | Fixed searches/crawls per step |
| **ReAct** | Medium/Extended | LLM controls tool calls |

**Classic Mode Process**:
1. Generate search queries using LLM
2. Perform web searches (max: `depth_config.max_search_queries`)
3. Crawl top URLs (max: `depth_config.max_urls_to_crawl`)
4. Format findings into observation

**ReAct Mode Process**:
- LLM decides which searches and crawls to perform
- Stops when satisfied or budget exhausted
- See [LLM Interaction](./llm-interaction.md) for details

**Output**:
- `state.last_observation`: Formatted web findings
- `state.sources`: Updated source list

**Model Tier**: Analytical

### 5. Reflector Agent

**File**: `src/agent/nodes/reflector.py`

**Purpose**: Analyze research progress and decide whether to continue, adjust, or complete.

**Input**:
- Plan summary (all steps + status)
- All observations collected so far
- All sources found
- Remaining pending steps
- Source topics (what's been covered)

**Output** (`ReflectionResult`):
```python
@dataclass
class ReflectionResult:
    decision: ReflectionDecision       # CONTINUE, ADJUST, or COMPLETE
    reasoning: str                     # Why this decision
    suggested_changes: list[str] | None
```

**Decision Logic**:

| Decision | Condition | Action |
|----------|-----------|--------|
| **CONTINUE** | Plan on track, more coverage needed | Execute next step |
| **ADJUST** | Plan not addressing query, gaps found | Return to Planner (preserves completed) |
| **COMPLETE** | Sufficient coverage achieved | Skip remaining steps, go to Synthesizer |

**Minimum Step Enforcement**:
- Reflector cannot decide COMPLETE until `min_steps` are completed
- Configured per depth in `config/app.yaml`

**Model Tier**: Simple

### 6. Synthesizer Agent

**File**: `src/agent/nodes/synthesizer.py` (standard) or `src/agent/nodes/citation_synthesizer.py` (with citations)

**Purpose**: Generate final research report with citations.

**Two Variants**:

**Standard Synthesizer**:
- Formats observations + sources list
- Calls LLM to generate final report
- Simple `[Title](url)` style citations

**Citation-Aware Synthesizer**:
- Runs full 7-stage citation verification pipeline
- Extracts claims from generated content
- Verifies each claim against evidence
- Emits events: `claim_verified`, `verification_summary`

**Selection**: Determined by `state.enable_citation_verification` flag

**Model Tier**: Complex

## Research State

**File**: `src/agent/state.py`

The `ResearchState` is the single source of truth passed between all agents:

```python
@dataclass
class ResearchState:
    # Core identifiers
    query: str                              # User's research query
    session_id: UUID                        # Unique session identifier
    conversation_history: list[dict]        # Previous messages

    # Query routing
    query_mode: str                         # "simple", "web_search", "deep_research"
    research_depth: str                     # "auto", "light", "medium", "extended"
    effective_depth: str                    # Resolved depth after auto-selection

    # Planning state
    current_plan: Plan | None               # Research plan with steps
    plan_iterations: int                    # Times plan has been adjusted
    current_step_index: int                 # Current step being executed

    # Research progress
    last_observation: str                   # Observation from current step
    all_observations: list[str]             # All step observations
    sources: list[SourceInfo]               # Collected web sources

    # Citation verification
    evidence_pool: list[EvidenceInfo]       # Pre-selected evidence spans
    claims: list[ClaimInfo]                 # Extracted claims with verdicts
    verification_summary: VerificationSummary
    enable_citation_verification: bool      # Feature toggle

    # Reflection
    last_reflection: ReflectionResult       # Last CONTINUE/ADJUST/COMPLETE
    reflection_history: list[ReflectionResult]

    # Output
    final_report: str                       # Generated research report
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `get_current_step()` | Get step being executed |
| `has_more_steps()` | Check if more steps remain |
| `advance_step()` | Move to next step |
| `get_completed_steps()` | Steps already executed (for ADJUST preservation) |
| `resolve_depth()` | Auto → light/medium/extended based on complexity |
| `get_max_steps()` / `get_min_steps()` | Depth-aware limits |

## Query Mode Routing

The orchestrator supports 3 query modes:

### Simple Mode
- **Triggers**: Coordinator detects simple query
- **Pipeline**: Direct LLM response (Synthesizer only)
- **Memory Access**: Can use existing sources from chat
- **Timeout**: None

### Web Search Mode
- **Triggers**: User selects "web_search" mode
- **Pipeline**: 1-step plan + Researcher + Synthesizer
- **Depth**: Forced to "light"
- **Timeout**: 15 seconds → falls back to Simple mode

### Deep Research Mode
- **Triggers**: User selects "deep_research" or default
- **Pipeline**: All 5 agents
- **Depth**: Auto/light/medium/extended (from config)
- **Timeout**: 300 seconds

## Data Flow Between Agents

```
Agent          Input State                Output State Changes
─────────────────────────────────────────────────────────────
Coordinator    query,                     query_classification
               conversation_history       is_simple_query
                                          direct_response

Background     query                      background_investigation_results
               background_config          sources (partial)

Planner        query,                     current_plan
               all_observations,          plan_iterations
               background_results,        current_step_index
               completed_steps,
               reflector_feedback

Researcher     current_step,              last_observation
               state.sources,             all_observations (append)
               query                      sources (add new)

Reflector      current_plan,              last_reflection
               all_observations,          reflection_history
               sources,
               current_step_index

Synthesizer    query,                     final_report
               all_observations,          completed_at
               sources,                   claims (if citation enabled)
               plans
```

## Configuration

Agent behavior is configured in `config/app.yaml`:

```yaml
agents:
  coordinator:
    enable_clarification: true
    max_clarifying_questions: 3

  planner:
    max_plan_iterations: 3
    preserve_completed_steps: true

  researcher:
    max_search_queries: 2
    max_urls_to_crawl: 3

  reflector:
    enforce_min_steps: true

  synthesizer:
    enable_citations: true
```

See [Configuration](./configuration.md) for full details.

## See Also

- [Architecture](./architecture.md) - System overview
- [Citation Pipeline](./citation-pipeline.md) - Verification stages
- [LLM Interaction](./llm-interaction.md) - ReAct researcher pattern
