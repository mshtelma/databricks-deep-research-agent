# Implementation Plan: Plugin Architecture

**Branch**: `005-plugin-architecture` | **Date**: 2026-01-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-plugin-architecture/spec.md`

## Summary

Implement a comprehensive plugin architecture for `databricks-deep-research-agent` that enables:

1. **Built-in retrieval tools** (Vector Search, Knowledge Assistant) configurable via `app.yaml`
2. **External plugin system** via Python entry points (`deep_research.plugins`)
3. **Declarative pipeline configuration** allowing child projects to replace or customize the agent architecture
4. **Custom output types** for domain-specific structured outputs
5. **Conversation handling extensibility** for follow-up message processing
6. **Deployment utilities** as Python modules for child project deployment
7. **Parent-child extensibility** allowing child projects to install core as a pip dependency and deploy independently

This is a comprehensive implementation that fully supports building applications like **sapresalesbot** as child projects.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI 0.115+, openai 1.10+, Pydantic v2, React 18, TanStack Query 5.x, databricks-sdk 0.30+
**Storage**: Databricks Lakebase (PostgreSQL) via asyncpg, existing schema extensions
**Testing**: pytest with 3-tier hierarchy (unit, integration, complex), Vitest (frontend), Playwright (E2E)
**Target Platform**: Databricks Apps (Linux container), local development (macOS/Linux)
**Project Type**: Web application (Python backend + React frontend) + pip-installable library
**Performance Goals**: Existing research pipeline performance must not degrade; tool registry lookup <1ms
**Constraints**: Must maintain backward compatibility with existing deployments; plugins discovered at startup (not hot-reload)
**Scale/Scope**: Support 10+ registered tools, 5+ plugins, extensible to child projects with independent databases

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Clients and Workspace Integration** | ✅ PASS | All tools use WorkspaceClient; no direct API calls |
| **II. Typing-First Python** | ✅ PASS | All protocols have full type annotations |
| **III. Avoid Runtime Introspection** | ✅ PASS | Use Protocol classes, not hasattr/isinstance |
| **IV. Linting and Static Type Enforcement** | ✅ PASS | mypy strict + ruff must pass |

## Project Structure

### Documentation (this feature)

```text
specs/005-plugin-architecture/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file
├── research.md          # Phase 0 output: resolved unknowns
├── data-model.md        # Phase 1 output: entity schemas
├── quickstart.md        # Phase 1 output: developer guide
├── contracts/           # Phase 1 output: API contracts
│   ├── tool-protocol.py
│   ├── plugin-protocol.py
│   ├── pipeline-protocol.py    # NEW
│   ├── output-protocol.py      # NEW
│   ├── conversation-protocol.py # NEW
│   └── config-schema.yaml
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
# Backend restructure: src/ → src/deep_research/
src/
├── deep_research/                  # Package root (renamed from src/)
│   ├── __init__.py                 # Public API exports
│   ├── agent/
│   │   ├── orchestrator.py         # Main pipeline (uses PipelineExecutor)
│   │   ├── state.py                # ResearchState (add plugin_data)
│   │   ├── config.py               # Agent config accessors
│   │   ├── nodes/                  # Existing agents
│   │   ├── pipeline/               # NEW: Pipeline infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── config.py           # PipelineConfig, AgentConfig
│   │   │   ├── executor.py         # PipelineExecutor
│   │   │   ├── defaults.py         # DEFAULT_DEEP_RESEARCH_PIPELINE, etc.
│   │   │   └── phase.py            # PhaseInsertion, custom phase support
│   │   └── tools/                  # Refactored tools
│   │       ├── base.py             # ResearchTool protocol
│   │       ├── registry.py         # ToolRegistry class
│   │       ├── web_search.py       # Refactored to protocol
│   │       ├── web_crawl.py        # Refactored to protocol
│   │       ├── vector_search.py    # NEW: VectorSearchTool
│   │       └── knowledge_assistant.py  # NEW: KnowledgeAssistantTool
│   ├── plugins/                    # Plugin infrastructure
│   │   ├── __init__.py             # Public exports
│   │   ├── base.py                 # ResearchPlugin, ToolProvider, PromptProvider
│   │   ├── pipeline.py             # NEW: PipelineCustomizer, PhaseProvider
│   │   ├── output.py               # NEW: OutputTypeProvider
│   │   ├── conversation.py         # NEW: ConversationProvider, IntentClassifier
│   │   ├── manager.py              # PluginManager class
│   │   ├── discovery.py            # Entry point discovery
│   │   └── context.py              # ResearchContext dataclass
│   ├── output/                     # NEW: Output type infrastructure
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseOutput, SynthesisReport
│   │   ├── registry.py             # OutputTypeRegistry
│   │   └── renderer.py             # Output rendering helpers
│   ├── conversation/               # NEW: Conversation handling
│   │   ├── __init__.py
│   │   ├── handler.py              # ConversationHandler base
│   │   ├── intent.py               # IntentClassification
│   │   └── default.py              # Default handlers
│   ├── deployment/                 # NEW: Deployment utilities
│   │   ├── __init__.py
│   │   ├── lakebase.py             # wait_for_lakebase(), health checks
│   │   ├── database.py             # create_database(), ensure_exists()
│   │   ├── migrations.py           # run_migrations(), multi-path support
│   │   ├── permissions.py          # grant_to_app()
│   │   ├── app_runner.py           # Production app runner
│   │   └── cli.py                  # CLI for deployment utilities
│   ├── services/                   # Existing services
│   ├── models/                     # Pydantic models
│   │   └── source.py               # Extended with source_type
│   ├── api/                        # FastAPI routes (unchanged)
│   ├── core/                       # Config, auth (extended)
│   │   ├── app_config.py           # Add plugin, pipeline, output sections
│   │   └── factory.py              # NEW: create_app() factory
│   ├── db/                         # Database (unchanged structure)
│   │   └── migrations/
│   │       └── versions/
│   │           └── 010_source_type_field.py
│   └── main.py                     # App entry (uses create_app)

# Frontend: Add component registry and exports
frontend/
├── src/
│   ├── core/                       # Exportable core components
│   │   ├── index.ts                # Public exports for child projects
│   │   ├── components/             # Moved from src/components/
│   │   ├── hooks/                  # Moved from src/hooks/
│   │   └── plugins/                # Component registry
│   │       ├── registry.ts         # ComponentRegistry class
│   │       ├── outputTypes.ts      # Output type renderer registry
│   │       └── types.ts            # Registry type definitions
│   ├── components/                 # Re-exports from core/
│   ├── hooks/                      # Re-exports from core/
│   └── pages/                      # App-specific pages
└── package.json                    # Add exports field for @deep-research/core

# Tests
tests/
├── unit/
│   ├── agent/
│   │   ├── tools/
│   │   │   ├── test_tool_registry.py
│   │   │   ├── test_vector_search.py
│   │   │   └── test_knowledge_assistant.py
│   │   └── pipeline/               # NEW
│   │       ├── test_pipeline_config.py
│   │       ├── test_pipeline_executor.py
│   │       └── test_phase_insertion.py
│   ├── plugins/
│   │   ├── test_plugin_manager.py
│   │   ├── test_discovery.py
│   │   ├── test_pipeline_customizer.py  # NEW
│   │   └── test_output_provider.py      # NEW
│   ├── output/                     # NEW
│   │   └── test_output_registry.py
│   ├── conversation/               # NEW
│   │   └── test_intent_classifier.py
│   └── deployment/                 # NEW
│       ├── test_lakebase.py
│       └── test_migrations.py
└── integration/
    ├── test_plugin_lifecycle.py
    ├── test_custom_pipeline.py     # NEW
    └── test_sapresalesbot_pattern.py  # NEW: Validate sapresalesbot can be built
```

## Complexity Tracking

| Aspect | Design Choice | Rationale |
|--------|--------------|-----------|
| Plugin discovery | Entry points | Standard Python mechanism |
| Pipeline config | Declarative dataclass | Flexible without LangGraph complexity |
| Output types | Pydantic models | Type-safe, JSON-serializable |
| Deployment | Python modules | Reusable, testable, cross-platform |
| Frontend extension | Build-time registry | Simpler than microfrontends |

---

## Implementation Phases

### Phase 0: Research (Completed)

See `research.md` for findings on:
- Databricks Vector Search API
- Knowledge Assistant API
- Entry point discovery
- Import restructure
- Migration multi-path support

---

### Phase 1: Design Artifacts

#### 1A: Tool Infrastructure (FR-200 series)

**ResearchTool Protocol** (`src/deep_research/agent/tools/base.py`):
```python
from typing import Protocol, Any
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """JSON Schema-compatible tool definition for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

@dataclass
class ToolResult:
    """Result from tool execution."""
    content: str
    success: bool
    sources: list[dict[str, Any]] | None = None
    data: dict[str, Any] | None = None

class ResearchTool(Protocol):
    """Protocol for all research tools (core and plugin-provided)."""

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: "ResearchContext",
    ) -> ToolResult:
        """Execute the tool with validated arguments."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments, return list of errors (empty if valid)."""
        ...
```

**ToolRegistry Class** (`src/deep_research/agent/tools/registry.py`):
- `register(tool: ResearchTool) -> None`
- `get(name: str) -> ResearchTool | None`
- `list_tools() -> list[ToolDefinition]`
- `get_openai_tools() -> list[dict]` (for LLM function calling)

#### 1B: Vector Search Tool (FR-210 series)

**Configuration** (`config/app.yaml`):
```yaml
vector_search:
  enabled: false
  endpoints:
    - name: product_docs
      endpoint: vs-production
      index: product_documentation_index
      description: "Search Databricks product documentation"
      num_results: 5
      columns_to_return:
        - title
        - content
        - url
```

**VectorSearchTool** (`src/deep_research/agent/tools/vector_search.py`):
- Implements `ResearchTool` protocol
- Uses `databricks.vector_search.client.VectorSearchClient`
- Returns `ToolResult` with `sources` containing `source_type: vector_search`

#### 1C: Knowledge Assistant Tool (FR-220 series)

**Configuration** (`config/app.yaml`):
```yaml
knowledge_assistants:
  enabled: false
  endpoints:
    - name: product_expert
      endpoint_name: product-knowledge-assistant
      description: "Query product expert for detailed information"
      max_tokens: 2000
      temperature: 0.3
```

**KnowledgeAssistantTool** (`src/deep_research/agent/tools/knowledge_assistant.py`):
- Implements `ResearchTool` protocol
- Uses Databricks serving endpoint API for KA inference
- Preserves and passes through citations from KA responses

#### 1D: Plugin System (FR-230 series)

**Plugin Protocols** (`src/deep_research/plugins/base.py`):
```python
class ResearchPlugin(Protocol):
    """Base protocol for all plugins."""
    name: str
    version: str

    def initialize(self, app_config: "AppConfig") -> None: ...
    def shutdown(self) -> None: ...

class ToolProvider(Protocol):
    """Protocol for plugins that provide tools."""
    def get_tools(self, context: ResearchContext) -> list[ResearchTool]: ...

class PromptProvider(Protocol):
    """Protocol for plugins that customize prompts."""
    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]: ...
```

**PluginManager** (`src/deep_research/plugins/manager.py`):
- Discovers plugins via `deep_research.plugins` entry point group
- Initializes plugins on app startup with error isolation
- Collects tools from all ToolProvider plugins
- Collects prompt overrides from all PromptProvider plugins
- Graceful shutdown on app termination

#### 1E: Pipeline Configuration (FR-290 series) - NEW

**PipelineConfig and AgentConfig** (`src/deep_research/agent/pipeline/config.py`):

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    BACKGROUND = "background"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REFLECTOR = "reflector"
    SYNTHESIZER = "synthesizer"
    CUSTOM = "custom"

@dataclass
class AgentConfig:
    """Configuration for a single agent in the pipeline."""
    agent_type: str
    enabled: bool = True
    model_tier: str = "analytical"
    next_on_success: str | None = None
    next_on_failure: str | None = None
    loop_condition: str | None = None
    loop_back_to: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Declarative pipeline configuration."""
    name: str
    description: str
    agents: list[AgentConfig]
    start_agent: str = "coordinator"
    max_iterations: int = 15
    timeout_seconds: int = 300
```

**PipelineExecutor** (`src/deep_research/agent/pipeline/executor.py`):

```python
class PipelineExecutor:
    """Executes agents according to pipeline configuration."""

    def __init__(
        self,
        pipeline: PipelineConfig,
        plugin_manager: PluginManager,
        llm_client: LLMClient,
    ):
        self.pipeline = pipeline
        self.plugin_manager = plugin_manager
        self.llm = llm_client

    async def execute(
        self,
        state: ResearchState,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the pipeline, yielding events."""
        current_agent = self.pipeline.start_agent
        iteration = 0

        while current_agent and iteration < self.pipeline.max_iterations:
            agent_config = self._get_agent_config(current_agent)
            if not agent_config or not agent_config.enabled:
                current_agent = agent_config.next_on_success if agent_config else None
                continue

            # Check for custom phases before this agent
            for phase in self.plugin_manager.get_phases_before(current_agent):
                async for event in phase.execute(state):
                    yield event

            # Execute agent
            agent = self._get_agent_instance(agent_config)
            async for event in agent.execute(state, self.llm):
                yield event

            # Determine next agent
            current_agent = self._get_next_agent(agent_config, state)
            iteration += 1

        yield ResearchCompletedEvent(state=state)
```

**Default Pipelines** (`src/deep_research/agent/pipeline/defaults.py`):

```python
DEFAULT_DEEP_RESEARCH_PIPELINE = PipelineConfig(
    name="deep_research",
    description="Multi-agent deep research with reflection",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="background"),
        AgentConfig(agent_type="background", next_on_success="planner"),
        AgentConfig(agent_type="planner", next_on_success="researcher"),
        AgentConfig(agent_type="researcher", next_on_success="reflector"),
        AgentConfig(
            agent_type="reflector",
            next_on_success="synthesizer",
            loop_condition="decision == CONTINUE",
            loop_back_to="researcher",
            config={"adjust_goes_to": "planner"}
        ),
        AgentConfig(agent_type="synthesizer", model_tier="complex"),
    ],
    start_agent="coordinator",
)

SIMPLE_RESEARCH_PIPELINE = PipelineConfig(
    name="simple_research",
    description="Single-pass research without reflection",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="researcher"),
        AgentConfig(agent_type="researcher", next_on_success="synthesizer"),
        AgentConfig(agent_type="synthesizer"),
    ],
    start_agent="coordinator",
)

REACT_LOOP_PIPELINE = PipelineConfig(
    name="react_loop",
    description="Single ReAct agent loop (for sapresalesbot pattern)",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="researcher"),
        AgentConfig(
            agent_type="researcher",
            next_on_success="synthesizer",
            loop_condition="needs_more_research",
            loop_back_to="researcher",
            config={"max_iterations": 10, "mode": "react"}
        ),
        AgentConfig(agent_type="synthesizer"),
    ],
    start_agent="coordinator",
)
```

**PipelineCustomizer Protocol** (`src/deep_research/plugins/pipeline.py`):

```python
from typing import Protocol
from dataclasses import dataclass, field

@dataclass
class PhaseInsertion:
    """Specification for inserting a custom phase."""
    phase: "CustomPhase"
    after: str | None = None
    before: str | None = None

@dataclass
class PipelineCustomization:
    """Customizations for the pipeline."""
    insert_phases: list[PhaseInsertion] = field(default_factory=list)
    agent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    disabled_agents: list[str] = field(default_factory=list)

class PipelineCustomizer(Protocol):
    """Protocol for plugins that customize the pipeline."""
    def get_pipeline_config(self, ctx: ResearchContext) -> PipelineConfig | None:
        """Return a complete custom pipeline, or None to use default."""
        ...

    def get_pipeline_customizations(self, ctx: ResearchContext) -> PipelineCustomization | None:
        """Return customizations to apply to the default pipeline."""
        ...

class PhaseProvider(Protocol):
    """Protocol for plugins that provide custom phases."""
    def get_phases(self, ctx: ResearchContext) -> list[PhaseInsertion]:
        """Return custom phases to insert into the pipeline."""
        ...

class CustomPhase(Protocol):
    """Protocol for custom pipeline phases."""
    name: str

    async def execute(
        self,
        state: ResearchState,
        context: ResearchContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the custom phase."""
        ...
```

#### 1F: Output Type Customization (FR-300 series) - NEW

**OutputTypeProvider Protocol** (`src/deep_research/plugins/output.py`):

```python
from typing import Protocol, Any, Type
from pydantic import BaseModel

class OutputTypeProvider(Protocol):
    """Protocol for plugins that define custom output types."""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for the custom output type."""
        ...

    def get_synthesizer_config(self, ctx: ResearchContext) -> dict[str, Any]:
        """Return synthesizer configuration for this output type."""
        ...

    def get_synthesizer_prompt(self, ctx: ResearchContext) -> str | None:
        """Return custom synthesizer prompt for this output type."""
        ...
```

**Default Output Type** (`src/deep_research/output/base.py`):

```python
from pydantic import BaseModel, Field
from typing import Any

class SynthesisReport(BaseModel):
    """Default output type for research reports."""
    title: str
    executive_summary: str
    sections: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Example Custom Output Type** (for sapresalesbot pattern):

```python
class MeetingPrepOutput(BaseModel):
    """Custom output type for meeting preparation (sapresalesbot pattern)."""
    account_name: str
    executive_summary: str
    meeting_plan: MeetingPlan
    key_insights: list[KeyInsight]
    discovery_questions: DiscoveryQuestions
    attendee_briefs: list[AttendeeBrief]
    case_studies: list[CaseStudy]
    landmines: list[str]
    sources: list[Source]
```

#### 1G: Deployment Utilities (FR-310 series) - NEW

**Lakebase Utilities** (`src/deep_research/deployment/lakebase.py`):

```python
import asyncio
from databricks.sdk import WorkspaceClient

async def wait_for_lakebase(
    instance_name: str,
    timeout: int = 300,
    poll_interval: int = 10,
) -> bool:
    """Wait for Lakebase instance to be ready."""
    w = WorkspaceClient()
    elapsed = 0

    while elapsed < timeout:
        try:
            creds = w.database.generate_database_credential(
                instance_names=[instance_name]
            )
            if creds and len(creds) > 0:
                return True
        except Exception:
            pass

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Lakebase {instance_name} not ready after {timeout}s")
```

**Database Utilities** (`src/deep_research/deployment/database.py`):

```python
def create_database(instance_name: str, database_name: str) -> None:
    """Create a database on the Lakebase instance."""
    from deep_research.db.lakebase_auth import get_lakebase_credentials
    # Connect to postgres, create database if not exists
```

**Migration Utilities** (`src/deep_research/deployment/migrations.py`):

```python
def run_migrations(alembic_ini: str = "alembic.ini", revision: str = "head") -> None:
    """Run Alembic migrations with multi-path support."""
```

**App Runner** (`src/deep_research/deployment/app_runner.py`):

```python
def run(app: str, host: str = "0.0.0.0", port: int = 8000,
        run_migrations: bool = False) -> None:
    """Run the application with graceful shutdown."""
```

**CLI Entry Points** (in `pyproject.toml`):

```toml
[project.scripts]
deep-research-run = "deep_research.deployment.app_runner:main"
deep-research-migrate = "deep_research.deployment.migrations:main"
```

#### 1H: Conversation Handling (FR-320 series) - NEW

**ConversationProvider Protocol** (`src/deep_research/plugins/conversation.py`):

```python
from typing import Protocol, Any
from dataclasses import dataclass
from enum import Enum

class FollowUpIntent(str, Enum):
    QA = "qa"
    RESEARCH_UPDATE = "research_update"
    CLARIFICATION_NEEDED = "clarification_needed"

@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: FollowUpIntent
    confidence: float
    parameters: dict[str, Any]

class IntentClassifier(Protocol):
    """Protocol for intent classification."""
    async def classify(
        self,
        message: str,
        current_state: Any,
        recent_context: str | None = None,
    ) -> IntentClassification:
        ...

class ConversationProvider(Protocol):
    """Protocol for plugins that handle follow-up conversations."""

    def get_intent_classifier(self) -> IntentClassifier | None:
        """Return custom intent classifier, or None for default."""
        ...

    def get_qa_handler(self) -> "QAHandler | None":
        """Return custom QA handler for read-only questions."""
        ...

    def get_update_handler(self) -> "UpdateHandler | None":
        """Return custom handler for research/update requests."""
        ...
```

#### 1I: App Factory (FR-318) - NEW

**create_app Factory** (`src/deep_research/core/factory.py`):

```python
from fastapi import FastAPI
from pathlib import Path
from typing import Any

def create_app(
    config_path: str | Path | None = None,
    plugins: list[Any] | None = None,
    pipeline: "PipelineConfig | None" = None,
    **kwargs: Any,
) -> FastAPI:
    """
    Factory function to create a Deep Research application.

    Args:
        config_path: Path to app.yaml configuration
        plugins: Optional list of plugin instances to register
        pipeline: Optional custom pipeline configuration
        **kwargs: Additional FastAPI constructor arguments

    Returns:
        Configured FastAPI application
    """
    # Load config, init plugin manager, determine pipeline
    # Create FastAPI app with lifecycle hooks
    return app
```

#### 1J: Package Restructure (FR-240 series)

**Directory Rename**:
- `src/` → `src/deep_research/`
- Add `src/deep_research/__init__.py` with public exports

**pyproject.toml Updates**:
```toml
[project]
name = "databricks-deep-research"
version = "0.1.0"

[project.scripts]
deep-research-run = "deep_research.deployment.app_runner:main"
deep-research-migrate = "deep_research.deployment.migrations:main"

[project.entry-points."deep_research.plugins"]
# External plugins register here

[project.entry-points."deep_research.tools"]
web_search = "deep_research.agent.tools.web_search:WebSearchTool"
web_crawl = "deep_research.agent.tools.web_crawl:WebCrawlTool"
vector_search = "deep_research.agent.tools.vector_search:VectorSearchTool"
knowledge_assistant = "deep_research.agent.tools.knowledge_assistant:KnowledgeAssistantTool"
```

#### 1K: Multi-Source Citations (FR-260 series)

**Source Model Extension**:
```python
class SourceType(str, Enum):
    WEB = "web"
    VECTOR_SEARCH = "vector_search"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"
    CUSTOM = "custom"  # For plugin-provided sources

# Add to Source model
source_type: SourceType = SourceType.WEB
source_metadata: dict[str, Any] | None = None
```

---

### Phase 2: Implementation Order

```
Week 1: Package Restructure (FR-240)
├── Rename src/ to src/deep_research/
├── Update all imports
├── Update pyproject.toml with entry points and scripts
├── Verify pip install -e works
└── Run full test suite

Week 2: Tool Infrastructure (FR-200, FR-202)
├── Implement ResearchTool protocol
├── Implement ToolRegistry
├── Refactor web_search to protocol
├── Refactor web_crawl to protocol
└── Unit tests for registry

Week 3: Plugin System Core (FR-230, FR-234, FR-235)
├── Implement plugin protocols (ResearchPlugin, ToolProvider, PromptProvider)
├── Implement PluginManager
├── Implement discovery
├── Error isolation testing
└── Integration tests

Week 4: Pipeline Configuration (FR-290 series)
├── Implement PipelineConfig, AgentConfig dataclasses
├── Implement PipelineExecutor
├── Implement default pipelines (DEFAULT_DEEP_RESEARCH, SIMPLE_RESEARCH, REACT_LOOP)
├── Implement PipelineCustomizer, PhaseProvider protocols
├── Implement PhaseInsertion support
├── Refactor orchestrator to use PipelineExecutor
└── Unit + integration tests

Week 5: Output Type System (FR-300 series)
├── Implement OutputTypeProvider protocol
├── Implement OutputTypeRegistry
├── Implement SynthesisReport default output
├── Integrate with synthesizer agent
├── Update frontend for custom output types
└── Unit tests

Week 6: Vector Search & KA Tools (FR-210, FR-220)
├── Implement VectorSearchTool
├── Implement KnowledgeAssistantTool
├── Add config sections to app.yaml
├── Integration tests with real endpoints
└── Update researcher agent to use registry

Week 7: Deployment Utilities (FR-310 series)
├── Implement deep_research.deployment.lakebase
├── Implement deep_research.deployment.database
├── Implement deep_research.deployment.migrations
├── Implement deep_research.deployment.permissions
├── Implement deep-research-run CLI command
├── Implement create_app() factory
└── Unit + integration tests

Week 8: Conversation Handling (FR-320 series)
├── Implement ConversationProvider protocol
├── Implement IntentClassifier protocol
├── Implement QAHandler, UpdateHandler protocols
├── Implement default handlers
├── Integration tests
└── Documentation

Week 9: Multi-Source Citations (FR-260)
├── Add source_type to Source model
├── Database migration
├── Update evidence selector
├── Update citation display
└── E2E tests

Week 10: Child Project Support (FR-270, FR-250)
├── Document deployment workflow using Python utilities
├── Test migration multi-path
├── Create example child project structure
├── Verify independent deployment
└── Write comprehensive quickstart guide

Week 11: Frontend Extensibility (FR-280)
├── Create ComponentRegistry
├── Restructure frontend for exports
├── Add output type renderer registry
├── Update package.json exports
├── Test child frontend imports
└── Documentation

Week 12: sapresalesbot Validation & Polish
├── Validate sapresalesbot can be built as child project
├── Create sapresalesbot example plugin structure
├── Full E2E test suite
├── Performance validation
├── Documentation review
└── Release preparation
```

---

## Artifacts to Generate

| Artifact | Purpose | Generated By |
|----------|---------|--------------|
| `research.md` | Resolve unknowns before design | Phase 0 research |
| `data-model.md` | Entity schemas and relationships | Phase 1 design |
| `contracts/tool-protocol.py` | ResearchTool protocol | Phase 1A |
| `contracts/plugin-protocol.py` | Base plugin protocols | Phase 1D |
| `contracts/pipeline-protocol.py` | Pipeline protocols | Phase 1E |
| `contracts/output-protocol.py` | Output type protocol | Phase 1F |
| `contracts/conversation-protocol.py` | Conversation protocols | Phase 1H |
| `contracts/config-schema.yaml` | Configuration schema | Phase 1 |
| `quickstart.md` | Developer guide for plugins | Phase 1 complete |
| `tasks.md` | Implementation tasks | `/speckit.tasks` |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Import breakage after restructure | Comprehensive grep + test before commit |
| Pipeline executor complexity | Start with simple linear execution, add branching incrementally |
| Custom output type edge cases | Validate with real sapresalesbot output schema |
| Deployment utility failures | Comprehensive error handling, clear error messages |
| Migration conflicts in child | Clear version naming, ordering documentation |
| Frontend bundle size | Tree-shaking, lazy loading for plugin components |
| Performance regression | Benchmark tool registry and pipeline execution |

---

## Success Validation

After implementation, verify all success criteria from spec:

- [ ] SC-001: Vector Search enabled via config only
- [ ] SC-002: Knowledge Assistant enabled via config only
- [ ] SC-003: External plugins discovered on startup
- [ ] SC-004: `pip install -e .` works in clean env
- [ ] SC-005: Citations show source type badges
- [ ] SC-006: App starts despite plugin failures
- [ ] SC-007: All existing tests pass (no regressions)
- [ ] SC-008: Child project deploys via `python -m deep_research.deployment.*`
- [ ] SC-009: Child project works end-to-end
- [ ] SC-010: Child extends UI without forking
- [ ] SC-011: Child config overrides parent defaults
- [ ] SC-012: Child can define custom pipeline
- [ ] SC-013: Child can define custom output types
- [ ] SC-014: Child can implement custom conversation handlers
- [ ] SC-015: **sapresalesbot can be implemented as a child project**
