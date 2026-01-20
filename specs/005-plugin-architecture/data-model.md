# Data Model: Plugin Architecture

**Date**: 2026-01-17 | **Branch**: `005-plugin-architecture`

This document defines the data models and entity relationships for the plugin architecture feature.

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Plugin System                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐       ┌──────────────────┐                        │
│  │  ResearchPlugin  │       │   PluginManager  │                        │
│  │  (Protocol)      │◄──────│   (Class)        │                        │
│  └────────┬─────────┘       └──────────────────┘                        │
│           │                                                              │
│           │ implements                                                   │
│           ▼                                                              │
│  ┌──────────────────┐       ┌──────────────────┐                        │
│  │  ToolProvider    │       │  PromptProvider  │                        │
│  │  (Protocol)      │       │  (Protocol)      │                        │
│  └────────┬─────────┘       └──────────────────┘                        │
│           │                                                              │
│           │ provides                                                     │
│           ▼                                                              │
│  ┌──────────────────┐       ┌──────────────────┐                        │
│  │  ResearchTool    │──────►│  ToolRegistry    │                        │
│  │  (Protocol)      │       │  (Class)         │                        │
│  └──────────────────┘       └──────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          Tool Execution                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐       ┌──────────────────┐                        │
│  │  ToolDefinition  │       │  ToolResult      │                        │
│  │  (Dataclass)     │       │  (Dataclass)     │                        │
│  └──────────────────┘       └──────────────────┘                        │
│                                                                          │
│  ┌──────────────────┐                                                   │
│  │ ResearchContext  │                                                   │
│  │ (Dataclass)      │                                                   │
│  └──────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Layer (Extended)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐       ┌──────────────────┐                        │
│  │  Source          │       │  SourceType      │                        │
│  │  (SQLModel)      │◄──────│  (Enum)          │                        │
│  │  + source_type   │       │  web             │                        │
│  │  + source_meta   │       │  vector_search   │                        │
│  └──────────────────┘       │  knowledge_asst  │                        │
│                             └──────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Protocols

### ResearchTool Protocol

The base protocol that all tools must implement.

```python
from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """JSON Schema-compatible tool definition for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any]

@dataclass
class ToolResult:
    """Result from tool execution."""
    content: str
    success: bool
    sources: list[dict[str, Any]] | None = None
    data: dict[str, Any] | None = None
    error: str | None = None

@runtime_checkable
class ResearchTool(Protocol):
    """Protocol for all research tools."""

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

### ResearchContext Dataclass

Context passed to all tool executions and plugin methods.

```python
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

@dataclass
class ResearchContext:
    """Contextual information for tool execution."""

    # Identity
    chat_id: UUID
    user_id: str
    research_session_id: UUID | None = None

    # Research configuration
    research_type: str = "medium"  # light, medium, extended

    # Registries (shared across tools)
    url_registry: dict[str, Any] = field(default_factory=dict)
    evidence_registry: dict[str, Any] = field(default_factory=dict)

    # Plugin-provided data
    plugin_data: dict[str, Any] = field(default_factory=dict)

    # Tool execution tracking
    tool_call_count: int = 0
    max_tool_calls: int = 20
```

---

## Plugin Protocols

### ResearchPlugin Protocol

Base protocol for all plugins.

```python
from typing import Protocol

class ResearchPlugin(Protocol):
    """Base protocol for all plugins."""

    @property
    def name(self) -> str:
        """Unique plugin identifier."""
        ...

    @property
    def version(self) -> str:
        """Plugin version string."""
        ...

    def initialize(self, app_config: "AppConfig") -> None:
        """Initialize plugin with application configuration."""
        ...

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        ...
```

### ToolProvider Protocol

Protocol for plugins that provide tools.

```python
class ToolProvider(Protocol):
    """Protocol for plugins that provide research tools."""

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """Return list of tools provided by this plugin."""
        ...
```

### PromptProvider Protocol

Protocol for plugins that customize prompts.

```python
class PromptProvider(Protocol):
    """Protocol for plugins that customize agent prompts."""

    def get_prompt_overrides(
        self,
        context: ResearchContext
    ) -> dict[str, str]:
        """
        Return prompt overrides for agents.

        Keys are agent names: "coordinator", "planner", "researcher",
        "reflector", "synthesizer".

        Values are prompt strings or template fragments to merge.
        """
        ...
```

---

## Manager Classes

### ToolRegistry Class

Central registry for all available tools.

```python
from typing import Sequence

class ToolRegistry:
    """Registry for research tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ResearchTool] = {}

    def register(self, tool: ResearchTool) -> None:
        """Register a tool. Raises if name conflicts."""
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = tool

    def register_with_prefix(
        self,
        tool: ResearchTool,
        prefix: str
    ) -> None:
        """Register tool with prefix to avoid conflicts."""
        # Used for plugin tools that conflict with core tools
        ...

    def get(self, name: str) -> ResearchTool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions."""
        return [t.definition for t in self._tools.values()]

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.definition.name,
                    "description": t.definition.description,
                    "parameters": t.definition.parameters,
                }
            }
            for t in self._tools.values()
        ]

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
```

### PluginManager Class

Lifecycle manager for plugins.

```python
from dataclasses import dataclass, field

@dataclass
class PluginManager:
    """Manages plugin discovery, initialization, and lifecycle."""

    _plugins: list[ResearchPlugin] = field(default_factory=list)
    _tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    _initialized: bool = False

    def discover_and_load(self, app_config: "AppConfig") -> None:
        """Discover and initialize all plugins."""
        from .discovery import discover_plugins

        for plugin_cls in discover_plugins():
            try:
                plugin = plugin_cls()
                plugin.initialize(app_config)
                self._plugins.append(plugin)
                logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
            except Exception as e:
                logger.warning(f"Failed to initialize plugin: {e}")

        self._register_tools()
        self._initialized = True

    def _register_tools(self) -> None:
        """Collect and register tools from all ToolProvider plugins."""
        context = ResearchContext(...)  # Minimal context for tool collection

        for plugin in self._plugins:
            if isinstance(plugin, ToolProvider):
                for tool in plugin.get_tools(context):
                    try:
                        self._tool_registry.register(tool)
                    except ValueError:
                        # Conflict - register with prefix
                        self._tool_registry.register_with_prefix(
                            tool,
                            prefix=plugin.name
                        )

    def get_tools(
        self,
        context: ResearchContext
    ) -> list[ResearchTool]:
        """Get all tools for the given context."""
        return list(self._tool_registry._tools.values())

    def get_prompt_overrides(
        self,
        context: ResearchContext
    ) -> dict[str, str]:
        """Collect prompt overrides from all PromptProvider plugins."""
        overrides: dict[str, str] = {}

        for plugin in self._plugins:
            if isinstance(plugin, PromptProvider):
                plugin_overrides = plugin.get_prompt_overrides(context)
                overrides.update(plugin_overrides)

        return overrides

    def shutdown(self) -> None:
        """Shutdown all plugins."""
        for plugin in self._plugins:
            try:
                plugin.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {plugin.name}: {e}")

        self._plugins.clear()
        self._initialized = False
```

---

## Configuration Models

### Vector Search Configuration

```python
from pydantic import BaseModel, Field

class VectorSearchEndpointConfig(BaseModel):
    """Configuration for a single Vector Search endpoint."""

    name: str = Field(..., description="Tool name suffix")
    endpoint: str = Field(..., description="VS endpoint name")
    index: str = Field(..., description="Index name (catalog.schema.index)")
    description: str = Field(..., description="Tool description for LLM")
    num_results: int = Field(default=5, ge=1, le=100)
    columns_to_return: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None

class VectorSearchConfig(BaseModel):
    """Vector Search configuration section."""

    enabled: bool = False
    endpoints: list[VectorSearchEndpointConfig] = Field(default_factory=list)
```

### Knowledge Assistant Configuration

```python
class KnowledgeAssistantEndpointConfig(BaseModel):
    """Configuration for a single Knowledge Assistant endpoint."""

    name: str = Field(..., description="Tool name suffix")
    endpoint_name: str = Field(..., description="Serving endpoint name")
    description: str = Field(..., description="Tool description for LLM")
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class KnowledgeAssistantsConfig(BaseModel):
    """Knowledge Assistants configuration section."""

    enabled: bool = False
    endpoints: list[KnowledgeAssistantEndpointConfig] = Field(
        default_factory=list
    )
```

### Plugin Configuration

```python
class PluginConfig(BaseModel):
    """Configuration for plugin system."""

    enabled: bool = True
    discover_entry_points: bool = True
    plugin_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-plugin configuration keyed by plugin name"
    )
```

---

## Database Extensions

### Source Model Extension

```python
from enum import Enum
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB

class SourceType(str, Enum):
    """Type of source for evidence."""
    WEB = "web"
    VECTOR_SEARCH = "vector_search"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"

# Extension to existing Source model
class Source(BaseModel):
    # ... existing fields ...

    # NEW: Source type for multi-source attribution
    source_type: SourceType = Field(
        default=SourceType.WEB,
        description="Type of source (web, vector_search, knowledge_assistant)"
    )

    # NEW: Source-specific metadata
    source_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Type-specific metadata (index name, score, etc.)"
    )
```

### Migration Script

```python
# 010_source_type_field.py
"""Add source_type and source_metadata to sources table.

Revision ID: 010
Revises: 009
Create Date: 2026-01-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column(
        'sources',
        sa.Column(
            'source_type',
            sa.String(32),
            nullable=False,
            server_default='web'
        )
    )
    op.add_column(
        'sources',
        sa.Column(
            'source_metadata',
            postgresql.JSONB,
            nullable=True
        )
    )

def downgrade() -> None:
    op.drop_column('sources', 'source_metadata')
    op.drop_column('sources', 'source_type')
```

---

## Frontend Types

### Component Registry Types

```typescript
// frontend/src/core/plugins/types.ts

export interface OutputRendererProps {
  content: string;
  metadata?: Record<string, unknown>;
}

export interface PanelProps {
  chatId: string;
  sessionId?: string;
}

export interface SourceBadgeProps {
  sourceType: 'web' | 'vector_search' | 'knowledge_assistant';
  metadata?: {
    indexName?: string;
    score?: number;
    assistantName?: string;
  };
}

export interface ComponentRegistry {
  outputRenderers: Map<string, React.ComponentType<OutputRendererProps>>;
  panelComponents: Map<string, React.ComponentType<PanelProps>>;

  registerRenderer(
    outputType: string,
    component: React.ComponentType<OutputRendererProps>
  ): void;

  registerPanel(
    panelId: string,
    component: React.ComponentType<PanelProps>
  ): void;

  getRenderer(
    outputType: string
  ): React.ComponentType<OutputRendererProps> | undefined;

  getPanel(
    panelId: string
  ): React.ComponentType<PanelProps> | undefined;
}
```

### Source Type Display

```typescript
// Extended Source type
export interface Source {
  id: string;
  url: string;
  title: string;
  domain: string;
  fetchedAt: string;
  // NEW
  sourceType: 'web' | 'vector_search' | 'knowledge_assistant';
  sourceMetadata?: {
    indexName?: string;
    relevanceScore?: number;
    assistantName?: string;
    assistantCitations?: Array<{
      source: string;
      title: string;
      url?: string;
    }>;
  };
}
```

---

---

## Pipeline Configuration Models

### PipelineConfig and AgentConfig

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

### PhaseInsertion and PipelineCustomization

```python
@dataclass
class PhaseInsertion:
    """Specification for inserting a custom phase."""
    phase: "CustomPhase"
    after: str | None = None
    before: str | None = None

@dataclass
class PipelineCustomization:
    """Customizations to apply to a pipeline."""
    insert_phases: list[PhaseInsertion] = field(default_factory=list)
    agent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    disabled_agents: list[str] = field(default_factory=list)
```

### Pipeline Protocols

```python
class CustomPhase(Protocol):
    """Protocol for custom pipeline phases."""
    name: str

    async def execute(
        self,
        state: ResearchState,
        context: ResearchContext,
    ) -> AsyncGenerator[StreamEvent, None]: ...

class PipelineCustomizer(Protocol):
    """Protocol for plugins that customize the pipeline."""
    def get_pipeline_config(self, ctx: ResearchContext) -> PipelineConfig | None: ...
    def get_pipeline_customizations(self, ctx: ResearchContext) -> PipelineCustomization | None: ...

class PhaseProvider(Protocol):
    """Protocol for plugins that provide custom phases."""
    def get_phases(self, ctx: ResearchContext) -> list[PhaseInsertion]: ...
```

---

## Output Type Models

### SynthesisReport (Default)

```python
class SynthesisReport(BaseModel):
    """Default output type for research reports."""
    title: str
    executive_summary: str
    sections: list[dict[str, Any]]
    key_findings: list[str]
    sources: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### OutputTypeProvider Protocol

```python
class OutputTypeProvider(Protocol):
    """Protocol for plugins that define custom output types."""

    def get_output_schema(self) -> Type[BaseModel]:
        """Return Pydantic model for custom output."""
        ...

    def get_synthesizer_config(self, ctx: ResearchContext) -> SynthesizerConfig:
        """Return synthesizer configuration."""
        ...

    def get_synthesizer_prompt(self, ctx: ResearchContext) -> str | None:
        """Return custom synthesizer prompt."""
        ...
```

### SynthesizerConfig

```python
@dataclass
class SynthesizerConfig:
    output_type: str
    model_tier: str = "complex"
    max_tokens: int = 8000
    temperature: float = 0.7
    system_prompt_addition: str | None = None
    output_format: str = "json"
    validation_strict: bool = True
```

---

## Conversation Handling Models

### IntentClassification

```python
class FollowUpIntent(str, Enum):
    QA = "qa"
    RESEARCH_UPDATE = "research_update"
    CLARIFICATION_NEEDED = "clarification_needed"
    EXPORT = "export"
    FEEDBACK = "feedback"

@dataclass
class IntentClassification:
    intent: FollowUpIntent
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
```

### Conversation Protocols

```python
class IntentClassifier(Protocol):
    async def classify(
        self,
        message: str,
        current_state: Any,
        recent_context: str | None = None,
    ) -> IntentClassification: ...

class QAHandler(Protocol):
    async def handle(
        self,
        message: str,
        classification: IntentClassification,
        current_state: Any,
        sources: list[dict[str, Any]],
        recent_context: str | None = None,
    ) -> AnswerResponse: ...

class UpdateHandler(Protocol):
    MAX_TOOL_CALLS: int = 5

    async def handle(
        self,
        message: str,
        classification: IntentClassification,
        current_state: Any,
        sources: list[dict[str, Any]],
        tools: list[Any],
        recent_context: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...

class ConversationProvider(Protocol):
    def get_intent_classifier(self) -> IntentClassifier | None: ...
    def get_qa_handler(self) -> QAHandler | None: ...
    def get_update_handler(self) -> UpdateHandler | None: ...
```

---

## Entity Relationships (Complete)

```
                                ┌─────────────────────────────────────────┐
                                │           PluginManager                  │
                                │  - discovers plugins via entry points   │
                                │  - manages lifecycle                     │
                                │  - collects tools, prompts, pipelines   │
                                └─────────────┬───────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│    ResearchPlugin       │   │   PipelineCustomizer    │   │   OutputTypeProvider    │
│    (Base Protocol)      │   │   (Pipeline Protocol)   │   │   (Output Protocol)     │
│  - name, version        │   │  - get_pipeline_config  │   │  - get_output_schema    │
│  - initialize/shutdown  │   │  - get_customizations   │   │  - get_synth_config     │
└──────────┬──────────────┘   └───────────┬─────────────┘   └───────────┬─────────────┘
           │                              │                             │
           ▼                              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│    ToolProvider         │   │   PipelineExecutor      │   │   Synthesizer           │
│  - get_tools()          │   │  - executes agents      │   │  - uses output schema   │
└──────────┬──────────────┘   │  - handles transitions  │   │  - custom prompt        │
           │                  │  - inserts phases       │   └─────────────────────────┘
           ▼                  └─────────────────────────┘
┌─────────────────────────┐
│    ToolRegistry         │              ┌─────────────────────────┐
│  - register tools       │              │   ConversationProvider  │
│  - get_openai_tools()   │              │  - intent_classifier    │
└─────────────────────────┘              │  - qa_handler           │
                                         │  - update_handler       │
                                         └─────────────────────────┘
```

```
Pipeline Execution Flow:

PipelineConfig ────► PipelineExecutor ────► Agent Sequence
                            │
                            │ for each agent:
                            ▼
                    ┌───────────────┐
                    │ Check Phases  │ ◄─── PhaseProvider (custom phases)
                    │   (before)    │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ Execute Agent │ ◄─── AgentConfig (model, settings)
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ Check         │
                    │ Transitions   │ ◄─── next_on_success, loop_condition
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ Next Agent or │
                    │ Complete      │
                    └───────────────┘
```

---

## Validation Rules

1. **Tool names must be unique** within the registry (conflicts prefixed)
2. **Plugin names must be unique** across all loaded plugins
3. **source_type must be valid enum value** (web, vector_search, knowledge_assistant, custom)
4. **Vector Search endpoint must specify** name, endpoint, index
5. **Knowledge Assistant endpoint must specify** name, endpoint_name
6. **Plugin initialization failures** are logged but don't prevent app startup
7. **PipelineConfig.start_agent** must reference an existing agent in the agents list
8. **AgentConfig transitions** (next_on_success, loop_back_to) must reference valid agents
9. **PhaseInsertion** must specify exactly one of `after` or `before`
10. **OutputTypeProvider.get_output_schema()** must return a Pydantic BaseModel subclass
11. **Custom output types** must include a `sources` field for citation tracking
