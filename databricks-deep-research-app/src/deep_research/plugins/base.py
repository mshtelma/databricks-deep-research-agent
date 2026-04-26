"""
Plugin Protocol Definitions
===========================

This module defines the plugin protocols for extending the Deep Research Agent.

Plugins can implement one or more protocols:
- ResearchPlugin: Base lifecycle (required)
- ToolProvider: Provide custom tools
- PromptProvider: Customize agent prompts
- ExtractionConfigProvider: Customize query context extraction
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from deep_research.agent.tools.base import ResearchContext, ResearchTool


@runtime_checkable
class ResearchPlugin(Protocol):
    """
    Base protocol for all plugins.

    Every plugin must implement this protocol to participate in the
    plugin lifecycle (discovery, initialization, shutdown).
    """

    @property
    def name(self) -> str:
        """
        Unique plugin identifier.

        Used for:
        - Configuration key in app.yaml (plugins.<name>)
        - Logging and error messages
        - Tool name prefixing on conflicts
        """
        ...

    @property
    def version(self) -> str:
        """
        Plugin version string (semver recommended).

        Used for:
        - Logging on startup
        - Compatibility tracking
        """
        ...

    def initialize(self, app_config: Any) -> None:
        """
        Initialize plugin with application configuration.

        Called once on application startup after discovery.
        Plugin-specific configuration is available at:
        app_config.plugins.get(self.name, {})

        Args:
            app_config: Full application configuration (AppConfig instance)

        Raises:
            Exception: If initialization fails (logged, app continues)
        """
        ...

    def shutdown(self) -> None:
        """
        Clean up resources on application shutdown.

        Called on graceful shutdown. Should release connections,
        close files, etc.
        """
        ...


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for plugins that provide research tools.

    Implement this alongside ResearchPlugin to add custom tools
    to the researcher agent's toolkit.
    """

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """
        Return list of tools provided by this plugin.

        Called when building the tool registry. Tools may be filtered
        or customized based on context (research type, user, etc.).

        Args:
            context: Research context for tool filtering

        Returns:
            List of ResearchTool implementations
        """
        ...


@runtime_checkable
class PromptProvider(Protocol):
    """
    Protocol for plugins that customize agent prompts.

    Implement this alongside ResearchPlugin to inject domain-specific
    instructions into agent prompts.
    """

    def get_prompt_overrides(
        self,
        context: ResearchContext,
    ) -> dict[str, str]:
        """
        Return prompt customizations for agents.

        Args:
            context: Research context for prompt customization

        Returns:
            Dict mapping agent names to prompt additions/overrides.

            Keys are agent names:
            - "coordinator": Query classification agent
            - "planner": Research planning agent
            - "researcher": Step execution agent
            - "reflector": Step-by-step reflection agent
            - "synthesizer": Report generation agent

            Values are prompt strings to append to the agent's
            system prompt. Use this to inject domain-specific
            instructions, constraints, or context.

        Example:
            {
                "researcher": '''
                    When researching companies, always check:
                    - Recent news and press releases
                    - Financial reports if available
                    - Key executive changes
                ''',
                "synthesizer": '''
                    Format the final report using MEDDPICC framework:
                    - Metrics
                    - Economic Buyer
                    - Decision Criteria
                    ...
                '''
            }
        """
        ...


@dataclass
class ExtractionConfig:
    """Configuration for query context extraction.

    Provided by plugins to customize extraction behavior.
    Framework uses generic logic with this config.

    Attributes:
        system_prompt: Custom system prompt for extraction (domain-specific examples go here).
        extraction_model: Pydantic model class for structured output extraction.
        field_mapping: Mapping from extraction model fields to plugin_data keys.
            E.g., {"primary_entity": "company_name", "people": "attendees"}
    """

    system_prompt: str
    extraction_model: type[BaseModel]
    field_mapping: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ExtractionConfigProvider(Protocol):
    """Plugin protocol for providing query extraction configuration.

    Plugins implement this to customize how queries are parsed.
    Framework remains generic - all domain knowledge lives in plugin.

    Example:
        class MyPlugin:
            def get_extraction_config(self) -> ExtractionConfig | None:
                return ExtractionConfig(
                    system_prompt="Extract company info...",
                    extraction_model=MyExtractionModel,
                    field_mapping={"company": "company_name"},
                )
    """

    def get_extraction_config(self) -> ExtractionConfig | None:
        """Return extraction config or None to skip extraction."""
        ...


# =============================================================================
# Enterprise Data Source Protocols (007-enterprise-data-sources)
# =============================================================================


@dataclass
class DataSourceDefinition:
    """Definition of a queryable data source.

    Used to describe data sources provided by plugins. System and user
    data sources use the schema version (from data_source.py).
    """

    type: str  # DataSourceType value
    name: str
    description: str
    endpoint_identifier: str
    capabilities: list[str] = field(default_factory=list)
    filter_schema: dict[str, Any] | None = None
    example_queries: list[str] = field(default_factory=list)
    source: str = "plugin"


@dataclass
class SourceConstraints:
    """Constraints on which sources can be used.

    Applied by plugins to restrict source selection during research.
    """

    allowed_types: set[str] | None = None  # None = all allowed
    allowed_sources: list[str] | None = None
    required_sources: list[str] = field(default_factory=list)
    excluded_sources: list[str] = field(default_factory=list)


@dataclass
class PromptTemplateDefinition:
    """Definition of a prompt template provided by a plugin."""

    id: str
    name: str
    type: str  # 'system', 'step', 'synthesis', 'query'
    content: str
    variables: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class CustomAgentDefinition:
    """Definition of a custom agent provided by a plugin."""

    id: str
    name: str
    description: str
    system_prompt: str
    source_scope: str | None = None  # SourceScope value
    enabled_sources: list[str] = field(default_factory=list)
    disabled_sources: list[str] = field(default_factory=list)
    preset_steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FileChunk:
    """A chunk of content extracted from a file."""

    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataSourceProvider(Protocol):
    """Protocol for plugins that provide data sources (T004).

    Implement this to register data sources (Vector Search, Genie spaces,
    Knowledge Assistants) that appear in the source browser.
    """

    def get_data_sources(self, context: ResearchContext) -> list[DataSourceDefinition]:
        """Return data sources provided by this plugin.

        Called when building the source browser. Sources may be filtered
        based on context (user, research type, etc.).

        Args:
            context: Research context for source filtering

        Returns:
            List of DataSourceDefinition objects
        """
        ...

    def get_source_constraints(self, context: ResearchContext) -> SourceConstraints | None:
        """Return default constraints for this plugin's sources.

        If provided, these constraints are applied when the plugin's
        sources are active.

        Args:
            context: Research context

        Returns:
            SourceConstraints or None if no constraints
        """
        ...


@runtime_checkable
class TemplateProvider(Protocol):
    """Protocol for plugins that provide prompt templates (T005).

    Implement this to register prompt templates that users can select
    in the template library.
    """

    def get_templates(self, context: ResearchContext) -> list[PromptTemplateDefinition]:
        """Return templates provided by this plugin.

        Args:
            context: Research context for template filtering

        Returns:
            List of PromptTemplateDefinition objects
        """
        ...

    def resolve_variable(
        self, variable_name: str, context: ResearchContext
    ) -> str | None:
        """Dynamically resolve a template variable value.

        Called during template rendering for plugin-owned variables.
        Return None to use default resolution.

        Args:
            variable_name: Name of the variable to resolve
            context: Research context

        Returns:
            Resolved value or None to use default
        """
        ...


@runtime_checkable
class CustomAgentProvider(Protocol):
    """Protocol for plugins that provide custom agents (T006).

    Implement this to register custom research agents that users
    can select in the agent selector.
    """

    def get_custom_agents(self, context: ResearchContext) -> list[CustomAgentDefinition]:
        """Return custom agents provided by this plugin.

        Args:
            context: Research context for agent filtering

        Returns:
            List of CustomAgentDefinition objects
        """
        ...


@runtime_checkable
class WorkflowProviderPlugin(Protocol):
    """Protocol for plugins that provide custom research workflows.

    Plugin supplies YAML content; the app handles execution, tool resolution,
    source policy, and streaming. The YAML should reference tools by stable
    logical names (web_search, web_crawl, plugin-provided tool names).

    Example::

        class CompliancePlugin:
            name = "compliance"
            version = "1.0.0"

            def get_workflow_yaml(self, ref: str) -> str | None:
                if ref == "compliance_audit":
                    return importlib.resources.read_text(
                        "compliance.workflows", "compliance_audit.yaml"
                    )
                return None
    """

    def get_workflow_yaml(self, ref: str) -> str | None:
        """Return YAML workflow content for the given ref, or None to defer.

        Args:
            ref: Workflow reference string (e.g., "compliance_audit").

        Returns:
            Raw YAML string, or None if this plugin doesn't own that ref.
        """
        ...


@runtime_checkable
class FileProcessorProvider(Protocol):
    """Protocol for plugins that provide file processing (T007).

    Implement this to handle custom file types during file upload.
    """

    def get_supported_extensions(self) -> set[str]:
        """Return file extensions this processor handles.

        Returns:
            Set of extensions (e.g., {'.pdf', '.docx', '.xlsx'})
        """
        ...

    async def process_file(
        self, file_path: str, context: ResearchContext
    ) -> list[FileChunk]:
        """Process file into searchable chunks.

        Args:
            file_path: Path to the uploaded file
            context: Research context

        Returns:
            List of FileChunk objects
        """
        ...


@runtime_checkable
class ContextEnricher(Protocol):
    """Protocol for plugins that enrich chat memory with domain-specific context.

    Called **once per turn**, inside ``framework_orchestrator.stream_research_via_framework``,
    **after** ``ChatMemoryService.hydrate`` and ``preprocess_new_files`` complete and
    **before** the workflow starts executing. Receives a live ``ChatMemoryService``
    handle so the plugin can produce domain-shaped derivations (e.g.,
    sapresalesbot's ``AccountBrief`` from file-derived entities).

    Lifecycle contract:

    - **Execution order**: plugin registration order (deterministic).
    - **Visibility**: each enricher sees the memory state built so far — by
      preprocessing and by prior enrichers. This is intentional; downstream
      plugins may refine earlier plugins' output.
    - **Writes**: plugins must write only under their own namespace via
      ``memory.enrich_scope(plugin_name)`` (enforced at the service layer).
    - **Failure mode**: exceptions are caught and logged as
      ``CONTEXT_ENRICHER_FAILED``; the workflow continues without this
      plugin's enrichment (fail-open).
    - **Timing budget**: soft cap of 5 seconds; exceeding it logs
      ``CONTEXT_ENRICHER_TIMEOUT`` and skips the enricher.

    The ``memory`` parameter is typed as ``Any`` to avoid a hard dependency
    on the framework memory module from the plugin protocol layer; plugins
    should import
    ``deep_research.services.chat_memory_service.ChatMemoryService`` for
    type hints in their own implementation.
    """

    async def enrich_research_memory(
        self,
        memory: Any,  # ChatMemoryService — typed as Any to avoid import cycle
        context: ResearchContext,
    ) -> None:
        """Write plugin-specific context into ``memory.plugin_extensions``.

        Implementations must be idempotent: on a follow-up turn where the
        plugin's derivation has not changed, the write should produce the
        same row (upsert-by-``plugin_name``).
        """
        ...


# Combined plugin type for plugins implementing multiple protocols
class FullPlugin(ResearchPlugin, ToolProvider, PromptProvider, Protocol):
    """
    Combined protocol for plugins implementing all capabilities.

    This is a convenience type for type hints. Most plugins will
    implement ResearchPlugin plus one or both of ToolProvider
    and PromptProvider.
    """

    pass


# Type aliases
PluginList = list[ResearchPlugin]
