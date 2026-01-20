# Plugin Architecture Quickstart Guide

**Date**: 2026-01-17 | **Branch**: `005-plugin-architecture`

This guide covers how to use and extend the Deep Research Agent plugin architecture.

## Table of Contents

1. [Enable Built-in Tools](#enable-built-in-tools)
2. [Create a Custom Plugin](#create-a-custom-plugin)
3. [Build a Child Project](#build-a-child-project)
4. [Extend the Frontend](#extend-the-frontend)

---

## Enable Built-in Tools

### Vector Search

Add Vector Search endpoints to query Databricks Vector Search indexes during research.

**1. Configure in `config/app.yaml`:**

```yaml
vector_search:
  enabled: true
  endpoints:
    - name: product_docs
      endpoint: vs-production-endpoint
      index: main.knowledge.product_documentation
      description: "Search Databricks product documentation for features, best practices, and technical details"
      num_results: 5
      columns_to_return:
        - title
        - content
        - url
        - category

    - name: case_studies
      endpoint: vs-production-endpoint
      index: main.knowledge.customer_success
      description: "Search customer case studies and success stories"
      num_results: 3
      columns_to_return:
        - customer_name
        - industry
        - solution
        - results
```

**2. Restart the application:**

```bash
make dev
```

**3. Use in research:**

The researcher agent automatically has access to `search_product_docs` and `search_case_studies` tools.

### Knowledge Assistant

Add Knowledge Assistant endpoints for optimized retrieval during research.

**1. Configure in `config/app.yaml`:**

```yaml
knowledge_assistants:
  enabled: true
  endpoints:
    - name: product_expert
      endpoint_name: product-knowledge-assistant
      description: "Query the product expert for detailed information about Databricks features and architecture"
      max_tokens: 2000
      temperature: 0.3

    - name: sales_intelligence
      endpoint_name: sales-knowledge-assistant
      description: "Query sales intelligence for competitive analysis and market insights"
      max_tokens: 1500
      temperature: 0.5
```

**2. Restart the application:**

The researcher agent can now use `retrieve_product_expert` and `retrieve_sales_intelligence` tools.

---

## Create a Custom Plugin

### Step 1: Project Structure

Create a new Python package:

```
my-research-plugin/
├── pyproject.toml
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       ├── plugin.py
│       ├── tools/
│       │   ├── __init__.py
│       │   └── custom_tool.py
│       └── prompts/
│           └── custom_prompts.py
└── tests/
    └── test_plugin.py
```

### Step 2: Define the Plugin

```python
# src/my_plugin/plugin.py
from typing import Any
from deep_research.plugins import ResearchPlugin, ToolProvider, PromptProvider
from deep_research.plugins.context import ResearchContext
from deep_research.agent.tools.base import ResearchTool
from .tools.custom_tool import MyCustomTool

class MyResearchPlugin:
    """Custom research plugin example."""

    name = "my_plugin"
    version = "0.1.0"

    def __init__(self) -> None:
        self._tools: list[ResearchTool] = []
        self._config: dict[str, Any] = {}

    def initialize(self, app_config: Any) -> None:
        """Initialize plugin with app configuration."""
        # Load plugin-specific config
        self._config = app_config.plugins.get("my_plugin", {})

        # Initialize tools
        if self._config.get("enable_custom_tool", True):
            self._tools.append(MyCustomTool(self._config))

    def shutdown(self) -> None:
        """Clean up resources."""
        self._tools.clear()

    # ToolProvider implementation
    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """Return available tools for this context."""
        return self._tools

    # PromptProvider implementation
    def get_prompt_overrides(
        self,
        context: ResearchContext
    ) -> dict[str, str]:
        """Return prompt customizations."""
        return {
            "researcher": self._config.get(
                "researcher_prompt_addition",
                ""
            )
        }
```

### Step 3: Implement a Custom Tool

```python
# src/my_plugin/tools/custom_tool.py
from typing import Any
from dataclasses import dataclass
from deep_research.agent.tools.base import (
    ResearchTool,
    ToolDefinition,
    ToolResult
)
from deep_research.plugins.context import ResearchContext

class MyCustomTool:
    """Example custom tool that queries an internal API."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._api_url = config.get("api_url", "https://api.example.com")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="query_internal_api",
            description="Query our internal API for proprietary data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext
    ) -> ToolResult:
        """Execute the tool."""
        query = arguments["query"]
        limit = arguments.get("limit", 10)

        try:
            # Call your internal API
            results = await self._call_api(query, limit)

            # Format results for LLM
            content = self._format_results(results)

            # Return with source tracking
            return ToolResult(
                content=content,
                success=True,
                sources=[
                    {
                        "type": "internal_api",
                        "query": query,
                        "result_count": len(results)
                    }
                ],
                data={"results": results}
            )

        except Exception as e:
            return ToolResult(
                content=f"API error: {e}",
                success=False,
                error=str(e)
            )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate tool arguments."""
        errors = []
        if "query" not in arguments:
            errors.append("Missing required argument: query")
        if "limit" in arguments:
            if not isinstance(arguments["limit"], int):
                errors.append("limit must be an integer")
            elif arguments["limit"] < 1 or arguments["limit"] > 100:
                errors.append("limit must be between 1 and 100")
        return errors

    async def _call_api(
        self,
        query: str,
        limit: int
    ) -> list[dict[str, Any]]:
        """Call the internal API."""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._api_url}/search",
                params={"q": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json()["results"]

    def _format_results(self, results: list[dict]) -> str:
        """Format results for LLM consumption."""
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"### Result {i}")
            lines.append(f"**Title**: {result.get('title', 'N/A')}")
            lines.append(f"**Content**: {result.get('content', '')[:500]}")
            lines.append("")
        return "\n".join(lines)
```

### Step 4: Register as Entry Point

```toml
# pyproject.toml
[project]
name = "my-research-plugin"
version = "0.1.0"
dependencies = [
    "databricks-deep-research>=0.1.0",
]

[project.entry-points."deep_research.plugins"]
my_plugin = "my_plugin.plugin:MyResearchPlugin"
```

### Step 5: Install and Use

```bash
# Development mode
pip install -e ./my-research-plugin

# The plugin is auto-discovered on app startup
make dev
```

---

## Build a Child Project

A "child project" is a complete application that uses Deep Research as its foundation.

### Step 1: Create Project Structure

```
my-research-agent/
├── pyproject.toml
├── Makefile
├── databricks.yml
├── app.yaml
├── alembic.ini
├── config/
│   └── app.yaml
├── src/
│   └── my_agent/
│       ├── __init__.py
│       ├── main.py          # FastAPI app
│       ├── plugin.py        # Your plugin
│       ├── tools/
│       └── db/
│           └── migrations/
│               └── versions/
├── client/                   # Frontend
│   ├── package.json
│   └── src/
└── static/                   # Built frontend
```

### Step 2: Configure Dependencies

```toml
# pyproject.toml
[project]
name = "my-research-agent"
version = "0.1.0"
dependencies = [
    "databricks-deep-research>=0.1.0",
    # Your additional dependencies
]

[project.entry-points."deep_research.plugins"]
my_agent_plugin = "my_agent.plugin:MyAgentPlugin"
```

### Step 3: Create the FastAPI App

```python
# src/my_agent/main.py
from deep_research.main import create_app
from deep_research.plugins import PluginManager
from .plugin import MyAgentPlugin

# Create app with your configuration
app = create_app(
    config_path="config/app.yaml",
    # Optional: explicitly register your plugin
    plugins=[MyAgentPlugin()]
)

# Add custom routes if needed
@app.get("/custom-endpoint")
async def custom_endpoint():
    return {"status": "ok"}
```

### Step 4: Configure Migrations

```ini
# alembic.ini
[alembic]
script_location = my_agent:db/migrations

# Include both parent and child migrations
version_locations =
    deep_research:db/migrations/versions
    %(here)s/src/my_agent/db/migrations/versions
```

### Step 5: Configure Deployment

Copy and adapt the parent's Makefile targets:

```makefile
# Makefile
TARGET ?= dev
INSTANCE = my-agent-lakebase
DATABASE = my_agent
APP_NAME = my-research-agent

.PHONY: install dev build deploy

install:
	uv sync
	cd client && bun install

dev:
	$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	uv run uvicorn src.my_agent.main:app --reload --port 8000

dev-frontend:
	cd client && bun run dev

build:
	cd client && bun run build
	rm -rf static && mv client/dist static

deploy: build
	uv pip compile pyproject.toml -o requirements.txt

	# Phase 1: Bootstrap
	LAKEBASE_DATABASE=postgres databricks bundle deploy --target $(TARGET)

	# Wait for Lakebase
	python -m deep_research.deployment.lakebase wait \
		--instance $(INSTANCE) --timeout 300

	# Create database
	python -m deep_research.deployment.database create \
		--instance $(INSTANCE) --database $(DATABASE)

	# Phase 2: Full deployment
	databricks bundle deploy --target $(TARGET)

	# Run migrations
	python -m deep_research.deployment.migrations run --alembic-ini ./alembic.ini

	# Grant permissions
	python -m deep_research.deployment.permissions grant \
		--app $(APP_NAME) --database $(DATABASE)

	@echo "Deployment complete!"
```

### Step 6: Override Configuration

```yaml
# config/app.yaml
# Inherit from parent defaults, override as needed

default_role: analytical

# Use your own endpoints
endpoints:
  my-llm-endpoint:
    endpoint_identifier: my-custom-model
    max_context_window: 128000

# Enable additional tools
vector_search:
  enabled: true
  endpoints:
    - name: my_docs
      endpoint: my-vs-endpoint
      index: my_catalog.my_schema.docs_index

# Plugin-specific config
plugins:
  my_agent_plugin:
    custom_setting: true
    api_endpoint: "https://my-api.example.com"
```

---

## Extend the Frontend

### Step 1: Install Parent Components

```json
// client/package.json
{
  "name": "my-research-agent-ui",
  "dependencies": {
    "@deep-research/core": "^0.1.0"
  }
}
```

### Step 2: Import and Extend

```typescript
// client/src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from '@deep-research/core';
import { pluginRegistry } from '@deep-research/core/plugins';
import { MyCustomPanel } from './components/MyCustomPanel';
import { MyOutputRenderer } from './components/MyOutputRenderer';

// Register custom components
pluginRegistry.registerPanel('my_custom_panel', MyCustomPanel);
pluginRegistry.registerRenderer('my_output_type', MyOutputRenderer);

// Mount the app with customizations
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App
      config={{
        title: "My Research Agent",
        logo: "/logo.png",
        // Enable custom panels
        panels: ['research', 'my_custom_panel'],
      }}
    />
  </React.StrictMode>
);
```

### Step 3: Create Custom Components

```typescript
// client/src/components/MyCustomPanel.tsx
import { PanelProps } from '@deep-research/core/plugins';

export const MyCustomPanel: React.FC<PanelProps> = ({ chatId, sessionId }) => {
  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold">My Custom Panel</h2>
      {/* Your custom UI */}
    </div>
  );
};
```

```typescript
// client/src/components/MyOutputRenderer.tsx
import { OutputRendererProps } from '@deep-research/core/plugins';
import { MarkdownRenderer } from '@deep-research/core/components';

export const MyOutputRenderer: React.FC<OutputRendererProps> = ({
  content,
  metadata
}) => {
  return (
    <div className="my-output">
      <div className="header">
        {metadata?.title && <h3>{metadata.title}</h3>}
      </div>
      <MarkdownRenderer content={content} />
    </div>
  );
};
```

### Step 4: Build and Deploy

```bash
cd client
bun run build
# Output goes to client/dist, then copied to static/

cd ..
make deploy TARGET=dev
```

---

## Configuration Reference

### Plugin Section

```yaml
plugins:
  # Enable/disable plugin system
  enabled: true

  # Auto-discover via entry points
  discover_entry_points: true

  # Per-plugin configuration
  my_plugin:
    setting1: value1
    setting2: value2
```

### Vector Search Section

```yaml
vector_search:
  enabled: false
  endpoints:
    - name: string           # Tool name suffix
      endpoint: string       # VS endpoint name
      index: string          # catalog.schema.index
      description: string    # Tool description for LLM
      num_results: 5         # Default results count
      columns_to_return: []  # Columns to include
      filters: {}            # Optional filters
```

### Knowledge Assistants Section

```yaml
knowledge_assistants:
  enabled: false
  endpoints:
    - name: string           # Tool name suffix
      endpoint_name: string  # Serving endpoint name
      description: string    # Tool description for LLM
      max_tokens: 2000       # Max response tokens
      temperature: 0.3       # Response temperature
```

---

## Troubleshooting

### Plugin Not Discovered

1. Check entry point is correctly defined in `pyproject.toml`
2. Ensure package is installed: `pip list | grep my-plugin`
3. Check app logs for discovery errors

### Tool Not Available

1. Verify tool is returned from `get_tools()`
2. Check tool definition has valid JSON Schema
3. Ensure tool name doesn't conflict with core tools

### Import Errors

1. Verify `databricks-deep-research` is installed
2. Check Python version is 3.11+
3. Run `pip install -e .` in your plugin directory

### Migration Errors

1. Verify `version_locations` in `alembic.ini`
2. Check parent package is importable
3. Run `alembic history` to see migration chain
