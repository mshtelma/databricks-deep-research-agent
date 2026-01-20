# Research: Plugin Architecture

**Date**: 2026-01-17 | **Branch**: `005-plugin-architecture`

This document captures research findings for technical unknowns identified in Phase 0 of the implementation plan.

## Research Tasks

### 1. Databricks Vector Search API

**Goal**: Verify SDK API shape for `similarity_search()` method.

**Findings**:

The Databricks SDK provides `VectorSearchClient` in `databricks.vector_search.client`:

```python
from databricks.vector_search.client import VectorSearchClient

# Initialize client (uses WorkspaceClient credentials automatically)
vsc = VectorSearchClient()

# Get index reference
index = vsc.get_index(
    endpoint_name="vs-endpoint-name",
    index_name="catalog.schema.index_name"
)

# Query the index
results = index.similarity_search(
    query_text="search query",           # For text embedding search
    # OR query_vector=[0.1, 0.2, ...],   # For raw vector search
    columns=["title", "content", "url"], # Columns to return
    filters={"category": "docs"},         # Optional filters
    num_results=10,                        # Number of results
)
```

**Response Structure**:
```python
{
    "manifest": {
        "column_count": 4,
        "columns": [
            {"name": "title", "type": "string"},
            {"name": "content", "type": "string"},
            {"name": "url", "type": "string"},
            {"name": "score", "type": "double"}
        ]
    },
    "result": {
        "row_count": 5,
        "data_array": [
            ["Product Guide", "Content here...", "https://...", 0.95],
            ["API Reference", "More content...", "https://...", 0.87],
            ...
        ]
    }
}
```

**Resolution**: API shape confirmed. Tool implementation will:
1. Initialize client lazily on first use
2. Use `similarity_search(query_text=...)` for semantic search
3. Return column-based results with relevance scores
4. Handle missing columns gracefully

---

### 2. Knowledge Assistant API

**Goal**: Verify serving endpoint API for Knowledge Assistant inference.

**Findings**:

Knowledge Assistants are deployed as Databricks serving endpoints with a chat-like interface. They use the standard Model Serving API:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

response = w.serving_endpoints.query(
    name="product-knowledge-assistant",
    dataframe_records=[
        {
            "messages": [
                {"role": "user", "content": "What is Delta Lake?"}
            ]
        }
    ]
)
```

**Response Structure**:
```python
{
    "predictions": [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Delta Lake is an open-source storage layer...",
                        "citations": [  # If KA is configured with citations
                            {
                                "source": "delta-lake-docs",
                                "title": "Delta Lake Overview",
                                "url": "https://docs.databricks.com/...",
                                "snippet": "Delta Lake provides ACID transactions..."
                            }
                        ]
                    }
                }
            ]
        }
    ]
}
```

**Resolution**: API confirmed. Tool implementation will:
1. Use `WorkspaceClient.serving_endpoints.query()`
2. Format queries as chat messages
3. Extract response content and citations
4. Pass through KA citations to citation pipeline

---

### 3. Entry Point Discovery

**Goal**: Verify `importlib.metadata.entry_points()` works for plugin discovery.

**Findings**:

Python 3.10+ provides a clean API for entry point discovery:

```python
import importlib.metadata

# Discover all plugins in the group
eps = importlib.metadata.entry_points(group="deep_research.plugins")

for ep in eps:
    print(f"Found plugin: {ep.name}")
    plugin_cls = ep.load()  # Load the class
    plugin = plugin_cls()   # Instantiate
```

**Plugin Registration** (in child project's `pyproject.toml`):
```toml
[project.entry-points."deep_research.plugins"]
my_plugin = "mypackage.plugin:MyPlugin"
```

**Error Handling**:
```python
for ep in eps:
    try:
        plugin_cls = ep.load()
        plugins.append(plugin_cls)
    except ImportError as e:
        logger.warning(f"Failed to load plugin {ep.name}: {e}")
    except Exception as e:
        logger.error(f"Plugin {ep.name} raised error: {e}")
```

**Resolution**: Entry point discovery works as expected. Implementation will:
1. Use `importlib.metadata.entry_points(group="deep_research.plugins")`
2. Catch and log load failures
3. Continue with remaining plugins on individual failures

---

### 4. Import Restructure

**Goal**: Verify imports work after `src/` → `src/deep_research/` rename.

**Findings**:

After restructure, the package layout will be:
```
src/
└── deep_research/
    ├── __init__.py
    ├── agent/
    ├── services/
    └── ...
```

**pyproject.toml Configuration**:
```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"
```

**Import Pattern Change**:
```python
# Before
from src.agent.orchestrator import run_research

# After
from deep_research.agent.orchestrator import run_research
```

**Development Installation**:
```bash
pip install -e .
# Installs "deep_research" package from src/deep_research/
```

**Migration Script** (for bulk import update):
```bash
# Find and replace all imports
find . -name "*.py" -type f -exec sed -i '' 's/from src\./from deep_research./g' {} +
find . -name "*.py" -type f -exec sed -i '' 's/import src\./import deep_research./g' {} +
```

**Resolution**: Structure confirmed. Implementation will:
1. Rename directory `src/` contents to `src/deep_research/`
2. Run bulk import update
3. Update pyproject.toml package configuration
4. Run full test suite to verify

---

### 5. Migration version_locations

**Goal**: Verify Alembic supports multiple migration paths for parent + child.

**Findings**:

Alembic supports multiple `version_locations` in `alembic.ini`:

```ini
[alembic]
script_location = myproject:db/migrations

# Multiple version locations - Alembic runs all in dependency order
version_locations =
    deep_research:db/migrations/versions
    %(here)s/src/mychildproject/db/migrations/versions
```

**Key Points**:
1. Package paths use colon notation: `packagename:path/inside/package`
2. File paths can use `%(here)s` for alembic.ini location
3. Migrations are sorted by revision timestamp/id across all locations
4. Down-revisions can reference migrations from other locations

**Child Project Setup**:
```python
# env.py modification for child project
from alembic import context
import deep_research.db.migrations  # Ensure parent migrations are importable

# Run migrations normally - Alembic handles multi-path automatically
```

**Resolution**: Multi-path migrations work. Implementation will:
1. Use package-relative paths for parent migrations
2. Document child project alembic.ini configuration
3. Ensure parent migration revision IDs don't conflict with child

---

## Spec Amendments

Based on research findings, no spec amendments required. All technical assumptions validated.

## Risk Updates

| Risk | Original Assessment | Updated Assessment |
|------|--------------------|--------------------|
| Vector Search API stability | Medium | Low - SDK API is stable |
| KA citation preservation | Medium | Low - Citations in response |
| Entry point discovery | Low | Confirmed working |
| Import restructure | Medium | Low - Standard pattern |
| Migration multi-path | Medium | Low - Alembic supports natively |

## Open Items (Resolved)

All Phase 0 unknowns have been resolved. Ready to proceed to Phase 1 design.

| Item | Status | Resolution |
|------|--------|------------|
| Vector Search API shape | Resolved | `VectorSearchClient.get_index().similarity_search()` |
| KA inference API | Resolved | `WorkspaceClient.serving_endpoints.query()` |
| Entry point mechanism | Resolved | `importlib.metadata.entry_points()` |
| Package structure | Resolved | `src/deep_research/` with setuptools find |
| Migration paths | Resolved | Alembic version_locations with package paths |
