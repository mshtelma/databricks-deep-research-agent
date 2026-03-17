# Plugin Development Guide

This guide explains how to create plugins for the Deep Research Agent that provide custom data sources, templates, and agents.

## Overview

The plugin system allows you to extend the Deep Research Agent with:

- **Data Sources**: Custom connectors for enterprise systems
- **Templates**: Reusable prompt templates with variables
- **Custom Agents**: Specialized research assistants
- **File Processors**: Handlers for custom file formats
- **Workflows**: Custom YAML research pipelines resolved by ref

## Plugin Architecture

Plugins implement one or more **provider protocols** and are discovered at startup.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Plugin Manager                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │ DataSource    │  │ Template      │  │ CustomAgent   │       │
│  │ Provider      │  │ Provider      │  │ Provider      │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│  ┌───────────────┐  ┌───────────────┐                          │
│  │ Workflow      │  │ FileProcessor │                          │
│  │ Provider      │  │ Provider      │                          │
│  └───────────────┘  └───────────────┘                          │
│         ↓                  ↓                  ↓                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Lifecycle Events                      │  │
│  │  DataSourceQueryEvent | TemplateAppliedEvent | ...       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Provider Protocols

### DataSourceProvider

Register custom data sources that appear in the discovery list.

```python
from deep_research.plugins.base import DataSourceProvider
from deep_research.schemas.data_source import DataSourceDefinition, SourceConstraints

class MyDataSourceProvider(DataSourceProvider):
    """Provider for custom enterprise data sources."""

    def get_data_sources(self) -> list[DataSourceDefinition]:
        """Return available data sources.

        Called during discovery to list available sources.

        Returns:
            List of data source definitions.
        """
        return [
            DataSourceDefinition(
                source_id="custom:my-system",
                source_type="custom",
                name="My Enterprise System",
                description="Search our internal knowledge base",
                endpoint_identifier="https://api.internal.com/search",
                capabilities={
                    "query_types": ["keyword"],
                    "supports_filters": True,
                },
            )
        ]

    def get_source_constraints(self) -> SourceConstraints | None:
        """Return constraints when this source is active.

        Optional. Define constraints that apply when this source
        is being used (e.g., rate limits, required context).

        Returns:
            Source constraints or None.
        """
        return SourceConstraints(
            max_queries_per_step=5,
            requires_user_context=True,
        )
```

### TemplateProvider

Register reusable prompt templates.

```python
from deep_research.plugins.base import TemplateProvider
from deep_research.schemas.template import PromptTemplateDefinition, TemplateVariable

class MyTemplateProvider(TemplateProvider):
    """Provider for custom prompt templates."""

    def get_templates(self) -> list[PromptTemplateDefinition]:
        """Return available templates.

        Called when listing templates in the UI.

        Returns:
            List of template definitions.
        """
        return [
            PromptTemplateDefinition(
                template_id="security-analysis",
                name="Security Analysis Report",
                type="synthesis",
                content="""
                Generate a security analysis report for {{product}}.

                Focus on:
                - Known vulnerabilities
                - Compliance status: {{compliance_framework}}
                - Risk assessment

                Format: {{output_format}}
                """,
                variables=[
                    TemplateVariable(
                        name="product",
                        type="string",
                        required=True,
                        description="Product to analyze",
                    ),
                    TemplateVariable(
                        name="compliance_framework",
                        type="choice",
                        required=True,
                        choices=["SOC2", "HIPAA", "PCI-DSS", "GDPR"],
                        default="SOC2",
                    ),
                    TemplateVariable(
                        name="output_format",
                        type="choice",
                        choices=["detailed", "summary", "executive"],
                        default="detailed",
                    ),
                ],
                tags=["security", "compliance", "analysis"],
            )
        ]

    def resolve_variable(
        self,
        template_id: str,
        variable_name: str,
        context: dict,
    ) -> str | None:
        """Resolve a dynamic variable value.

        Called when a variable needs runtime resolution.

        Args:
            template_id: Template being rendered.
            variable_name: Variable to resolve.
            context: Current research context.

        Returns:
            Resolved value or None to use default.
        """
        if variable_name == "product" and "query" in context:
            # Extract product name from query
            return self._extract_product(context["query"])
        return None
```

### CustomAgentProvider

Register specialized research agents.

```python
from deep_research.plugins.base import CustomAgentProvider
from deep_research.schemas.custom_agent import CustomAgentDefinition, PresetStep

class MyAgentProvider(CustomAgentProvider):
    """Provider for custom research agents."""

    def get_custom_agents(self) -> list[CustomAgentDefinition]:
        """Return available custom agents.

        Called when listing agents in the selector.

        Returns:
            List of custom agent definitions.
        """
        return [
            CustomAgentDefinition(
                agent_id="security-researcher",
                name="Security Researcher",
                description="Specialized agent for security research",
                avatar_url="/static/agents/security.png",
                system_prompt="""
                You are a security research specialist. Focus on:
                - Identifying vulnerabilities and risks
                - Analyzing security implications
                - Providing actionable recommendations

                Always cite CVEs and security advisories.
                """,
                source_scope="enterprise_only",
                enabled_sources=["internal-vuln-db", "cve-search"],
                use_planner=True,
                default_depth="extended",
                output_format="structured",
                preset_steps=[
                    PresetStep(
                        title="Identify Affected Systems",
                        description="Search for systems affected by the vulnerability",
                        source_hints=["internal-vuln-db"],
                    ),
                    PresetStep(
                        title="Research CVE Details",
                        description="Get detailed CVE information",
                        source_hints=["cve-search", "web_search"],
                    ),
                    PresetStep(
                        title="Analyze Impact",
                        description="Assess business impact and risk",
                        source_hints=["genie:risk-data"],
                    ),
                ],
            )
        ]
```

### WorkflowProviderPlugin

Supply custom YAML research workflows that are resolved by a `workflow_ref` string set on a custom agent.

When a custom agent has a `workflow_ref`, the app iterates registered plugins implementing `WorkflowProviderPlugin`. The first plugin whose `get_workflow_yaml()` returns a non-`None` string wins. If no plugin claims the ref, a `ValueError` is raised (strict — no silent fallback to the default pipeline).

```python
import importlib.resources

from deep_research.plugins.base import WorkflowProviderPlugin


class ComplianceWorkflowProvider(WorkflowProviderPlugin):
    """Provides compliance-focused research workflows."""

    name = "compliance"
    version = "1.0.0"

    def get_workflow_yaml(self, ref: str) -> str | None:
        """Return YAML for compliance workflows, or None to defer."""
        if ref == "compliance_audit":
            return importlib.resources.read_text(
                "compliance.workflows", "compliance_audit.yaml"
            )
        return None
```

The YAML follows the framework workflow schema. Here is an example `compliance_audit.yaml`:

```yaml
id: compliance_audit
name: Compliance Audit Research
version: "1.0"

pools:
  shared_pool:
    capacity: 50
    dedup: true

root:
  type: sequence
  steps:
    - type: agent
      agent_type: background
      config:
        pool: shared_pool
        tools: [web_search]
        max_tool_calls: 5

    - type: plan_and_execute
      config:
        pool: shared_pool
        planner:
          agent_type: planner
        researcher:
          agent_type: researcher
          config:
            mode: react
            tools: [web_search, web_crawl]
            max_tool_calls: 15
        reflector:
          agent_type: reflector
        max_steps: 8
        min_steps: 3

    - type: agent
      agent_type: synthesizer
      config:
        pool: shared_pool
```

To associate a custom agent with this workflow, set `workflow_ref` when creating the agent:

```json
POST /api/v1/custom-agents
{
  "name": "Compliance Auditor",
  "description": "Research agent for compliance audits",
  "system_prompt": "You are a compliance research specialist...",
  "workflow_ref": "compliance_audit"
}
```

**Key behaviors:**

- `get_workflow_yaml()` returns a raw YAML string (not a file path, not a parsed object)
- Tools referenced in the YAML are automatically filtered against runtime-available tools (removed tools are logged at INFO level)
- The app handles execution, tool resolution, source policy, OBO tokens, streaming, and persistence — the plugin only supplies the YAML
- Plugin exceptions are logged and the next plugin is tried
- Returning `None` means "I don't own that ref" (not an error) — the next plugin gets a chance

### FileProcessorProvider

Handle custom file formats for upload.

```python
from deep_research.plugins.base import FileProcessorProvider
from deep_research.schemas.file_upload import FileChunk

class MyFileProcessor(FileProcessorProvider):
    """Processor for custom file formats."""

    def get_supported_extensions(self) -> list[str]:
        """Return supported file extensions.

        Returns:
            List of extensions (without dot).
        """
        return ["custom", "xyz"]

    async def process_file(
        self,
        file_path: str,
        metadata: dict,
    ) -> list[FileChunk]:
        """Process a file into searchable chunks.

        Args:
            file_path: Path to uploaded file.
            metadata: File metadata.

        Returns:
            List of file chunks for indexing.
        """
        chunks = []
        content = self._read_custom_format(file_path)

        for i, section in enumerate(content.sections):
            chunks.append(
                FileChunk(
                    chunk_index=i,
                    content=section.text,
                    metadata={
                        "section_title": section.title,
                        "page": section.page,
                    },
                )
            )

        return chunks
```

## Lifecycle Events

Plugins can subscribe to lifecycle events for observability and side effects.

### Available Events

| Event | Triggered When |
|-------|----------------|
| `DataSourceQueryEvent` | A data source is queried |
| `TemplateAppliedEvent` | A template is rendered |
| `CustomAgentSelectedEvent` | A custom agent is activated |
| `ResearchStartedEvent` | Research session begins |
| `StepCompletedEvent` | A research step completes |

### Subscribing to Events

```python
from deep_research.plugins.lifecycle.events import (
    DataSourceQueryEvent,
    TemplateAppliedEvent,
)
from deep_research.plugins.lifecycle.protocol import LifecycleSubscriber

class MyEventHandler(LifecycleSubscriber):
    """Handle lifecycle events for analytics."""

    async def on_data_source_query(self, event: DataSourceQueryEvent) -> None:
        """Called when a data source is queried.

        Args:
            event: Query event with source info and results.
        """
        # Log to analytics
        await self.analytics.log_query(
            source_type=event.source_type,
            source_name=event.source_name,
            query=event.query,
            result_count=event.result_count,
            latency_ms=event.latency_ms,
        )

    async def on_template_applied(self, event: TemplateAppliedEvent) -> None:
        """Called when a template is applied.

        Args:
            event: Template application event.
        """
        # Track template usage
        await self.analytics.log_template_use(
            template_id=event.template_id,
            user_id=event.user_id,
        )
```

## Plugin Registration

### Automatic Discovery

Place plugins in the `plugins/` directory with a `__init__.py` that exports providers:

```python
# plugins/my_plugin/__init__.py
from .providers import (
    MyDataSourceProvider,
    MyTemplateProvider,
    MyAgentProvider,
)

__all__ = [
    "MyDataSourceProvider",
    "MyTemplateProvider",
    "MyAgentProvider",
]
```

### Manual Registration

Register providers programmatically:

```python
from deep_research.plugins.manager import get_plugin_manager

manager = get_plugin_manager()

# Register providers
manager.register_data_source_provider(MyDataSourceProvider())
manager.register_template_provider(MyTemplateProvider())
manager.register_agent_provider(MyAgentProvider())

# Register event subscriber
manager.register_lifecycle_subscriber(MyEventHandler())
```

### Configuration

Configure plugins in `config/app.yaml`:

```yaml
plugins:
  enabled:
    - my_plugin
    - another_plugin

  my_plugin:
    api_endpoint: "https://api.internal.com"
    api_key: "${MY_PLUGIN_API_KEY}"
    max_results: 50
```

Access configuration in your plugin:

```python
from deep_research.core.app_config import get_app_config

class MyDataSourceProvider(DataSourceProvider):
    def __init__(self):
        config = get_app_config()
        self.api_endpoint = config.plugins.my_plugin.api_endpoint
        self.api_key = config.plugins.my_plugin.api_key
```

## Creating a Custom Tool

For data sources that need custom query logic, create a tool class.

```python
from deep_research.agent.tools.base import BaseTool, ToolDefinition, ToolResult
from deep_research.schemas.source_info import SourceInfo

class MyCustomTool(BaseTool):
    """Tool for querying my custom data source."""

    def __init__(
        self,
        source_name: str,
        api_endpoint: str,
        user_token: str,
    ):
        self.source_name = source_name
        self.api_endpoint = api_endpoint
        self.user_token = user_token

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for the LLM."""
        return ToolDefinition(
            name=f"search_{self.source_name}",
            description=f"Search {self.source_name} for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool arguments from LLM.

        Returns:
            ToolResult with sources and summary.
        """
        query = kwargs["query"]
        filters = kwargs.get("filters", {})

        # Call your API
        results = await self._search(query, filters)

        # Convert to SourceInfo
        sources = [
            SourceInfo(
                url=r["url"],
                title=r["title"],
                content=r["content"],
                source_type="custom",
                source_name=self.source_name,
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

        return ToolResult(
            sources=sources,
            summary=f"Found {len(sources)} results from {self.source_name}",
        )

    async def _search(self, query: str, filters: dict) -> list[dict]:
        """Call the custom API."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_endpoint}/search",
                json={"query": query, "filters": filters},
                headers={"Authorization": f"Bearer {self.user_token}"},
            )
            response.raise_for_status()
            return response.json()["results"]
```

### Registering the Tool

Use the tool factory to create tools dynamically:

```python
from deep_research.agent.tools.factory import register_tool_factory

@register_tool_factory("my_custom_source")
def create_my_custom_tool(
    source: DataSourceDefinition,
    user_token: str,
) -> BaseTool:
    """Factory function for MyCustomTool."""
    return MyCustomTool(
        source_name=source.name,
        api_endpoint=source.endpoint_identifier,
        user_token=user_token,
    )
```

## Testing Plugins

### Unit Tests

```python
import pytest
from deep_research.plugins.base import DataSourceProvider

class TestMyDataSourceProvider:
    """Tests for MyDataSourceProvider."""

    @pytest.fixture
    def provider(self):
        return MyDataSourceProvider()

    def test_get_data_sources(self, provider):
        """Test source discovery."""
        sources = provider.get_data_sources()

        assert len(sources) >= 1
        assert sources[0].source_id == "custom:my-system"
        assert sources[0].source_type == "custom"

    def test_get_source_constraints(self, provider):
        """Test constraint configuration."""
        constraints = provider.get_source_constraints()

        assert constraints is not None
        assert constraints.max_queries_per_step == 5
```

### Integration Tests

```python
import pytest
from deep_research.plugins.manager import get_plugin_manager

@pytest.mark.integration
class TestPluginIntegration:
    """Integration tests for plugin registration."""

    def test_plugin_discovery(self):
        """Test plugins are discovered at startup."""
        manager = get_plugin_manager()
        providers = manager.get_data_source_providers()

        # Our plugin should be discovered
        provider_types = [type(p).__name__ for p in providers]
        assert "MyDataSourceProvider" in provider_types

    async def test_data_source_query(self, mock_api):
        """Test querying through the plugin."""
        manager = get_plugin_manager()
        tool = manager.create_tool_for_source("custom:my-system")

        result = await tool.execute(query="test query")

        assert len(result.sources) > 0
        assert result.sources[0].source_type == "custom"
```

## Best Practices

### Error Handling

Always handle errors gracefully:

```python
async def execute(self, **kwargs) -> ToolResult:
    try:
        results = await self._search(kwargs["query"])
        return ToolResult(sources=self._to_sources(results))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return ToolResult(
                sources=[],
                error="Authentication failed. Please re-authenticate.",
            )
        elif e.response.status_code == 429:
            return ToolResult(
                sources=[],
                error="Rate limit exceeded. Please try again later.",
            )
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return ToolResult(
            sources=[],
            error=f"Query failed: {str(e)[:100]}",
        )
```

### Logging

Use structured logging:

```python
from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)

class MyDataSourceProvider(DataSourceProvider):
    def get_data_sources(self):
        logger.info(
            "DISCOVERING_SOURCES",
            provider=self.__class__.__name__,
        )
        sources = self._discover()
        logger.info(
            "SOURCES_DISCOVERED",
            count=len(sources),
        )
        return sources
```

### Caching

Cache expensive operations:

```python
from functools import lru_cache
from deep_research.services.discovery_cache import get_discovery_cache

class MyDataSourceProvider(DataSourceProvider):
    @lru_cache(maxsize=100, ttl=300)  # 5 minute cache
    def get_data_sources(self):
        return self._discover()
```

### Security

- Never log sensitive data (tokens, credentials)
- Use OBO authentication for user-specific access
- Validate all inputs
- Sanitize outputs before returning

## Related Documentation

- [API Reference](./api.md) - REST API endpoints
- [Data Source Configuration](./data-source-config.md) - User guide
- [Architecture](./architecture.md) - System overview

## Support

For plugin development questions:
- Open an issue on GitHub
- Check existing plugins in `plugins/` for examples
- Review the protocol definitions in `src/deep_research/plugins/base.py`
